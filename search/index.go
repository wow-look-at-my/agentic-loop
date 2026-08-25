// Package search indexes a host's conversations so they can be searched by
// word and, when an embedding model is available, by meaning.
//
// It owns one SQLite file holding an FTS5 index over message text plus the
// vectors for a semantic search over the same messages. The conversations
// themselves stay wherever the host keeps them: the index reads them through
// the Source interface, and SessionSource adapts an agentic-loop
// session.Store, so cai and the http/socket servers get searchable history
// without changing how they store anything.
//
// The index is derived and always slightly BEHIND the conversations it
// indexes. Embedding requires a network call, so it could never be part of a
// write path. The design accepts the lag instead of pretending it away: Status
// reports how far behind the index is and what the last failure was, and every
// search says which of its halves actually answered.
//
// Depth: docs/search.md.
package search

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"strconv"

	"github.com/wow-look-at-my/go-containers/set"

	_ "modernc.org/sqlite" // pure-Go SQLite driver, registered as "sqlite"
)

// Index is the handle to the index database.
type Index struct {
	sql *sql.DB
}

// Open opens (creating if necessary) the index at path, bringing its schema up to date.
func Open(ctx context.Context, path string) (*Index, error) { return open(ctx, path, "full") }

// OpenEphemeral is Open with synchronous=OFF, for a database that is deleted moments later.
func OpenEphemeral(ctx context.Context, path string) (*Index, error) { return open(ctx, path, "off") }

func open(ctx context.Context, path, synchronous string) (*Index, error) {
	dsn := "file:" + path + "?" + url.Values{
		"_pragma": {
			"journal_mode(WAL)",
			"busy_timeout(5000)",
			"synchronous(" + synchronous + ")",
		},
	}.Encode()

	sqlDB, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("search: open %q: %w", path, err)
	}
	// One connection: this is a single-writer index, and the cap removes lock contention.
	sqlDB.SetMaxOpenConns(1)

	// Schema work runs detached from the caller's context so a mid-rebuild cancellation can't drop a half.
	startup := context.WithoutCancel(ctx)
	if err := sqlDB.PingContext(startup); err != nil {
		_ = sqlDB.Close()
		return nil, fmt.Errorf("search: ping %q: %w", path, err)
	}

	idx := &Index{sql: sqlDB}
	if err := idx.applySchema(startup); err != nil {
		_ = sqlDB.Close()
		return nil, err
	}
	return idx, nil
}

// Close closes the index database.
func (i *Index) Close() error { return i.sql.Close() }

// applySchema creates both halves and rebuilds either one whose recorded
// version is not the current one, or whose tables on disk are not the shape
// that version describes. The halves are handled independently, which is the
// whole point of versioning them separately: a change to the text index must
// not cost every caller their embeddings.
func (i *Index) applySchema(ctx context.Context) error {
	if _, err := i.sql.ExecContext(ctx, metaSchema); err != nil {
		return fmt.Errorf("search: create meta: %w", err)
	}

	ftsAt, err := i.metaInt(ctx, metaFTSVersion)
	if err != nil {
		return err
	}
	ftsShaped, err := i.shapeMatches(ctx, ftsTables)
	if err != nil {
		return err
	}
	if !ftsShaped || (ftsAt != 0 && ftsAt != ftsSchemaVersion) {
		if _, err := i.sql.ExecContext(ctx, dropFTSSchema); err != nil {
			return fmt.Errorf("search: drop text index for rebuild: %w", err)
		}
	}
	if _, err := i.sql.ExecContext(ctx, ftsSchema); err != nil {
		return fmt.Errorf("search: create text index: %w", err)
	}
	if err := i.setMeta(ctx, metaFTSVersion, strconv.Itoa(ftsSchemaVersion)); err != nil {
		return err
	}

	embedAt, err := i.metaInt(ctx, metaEmbedVersion)
	if err != nil {
		return err
	}
	embedShaped, err := i.shapeMatches(ctx, embedTables)
	if err != nil {
		return err
	}
	if !embedShaped || (embedAt != 0 && embedAt != embedSchemaVersion) {
		if _, err := i.sql.ExecContext(ctx, dropEmbedSchema); err != nil {
			return fmt.Errorf("search: drop vectors for rebuild: %w", err)
		}
	}
	if _, err := i.sql.ExecContext(ctx, embedSchema); err != nil {
		return fmt.Errorf("search: create vectors: %w", err)
	}
	return i.setMeta(ctx, metaEmbedVersion, strconv.Itoa(embedSchemaVersion))
}

// shapeMatches reports whether every table in want that is PRESENT in the file
// has the columns this version of the schema gives it. A table that is absent
// matches: the CREATE below makes it.
//
// The recorded version says which shape the file is MEANT to have. It cannot
// say which shape the file actually has, because the number is not the
// library's alone: another implementation of this index writes its own
// versions into the same meta table, and one of them met version 1 with a
// different column set. The version then reads as up to date, no rebuild runs,
// and the first CREATE INDEX over a column that is not there fails -- on every
// open, forever, for a file that is derived data and free to rebuild.
func (i *Index) shapeMatches(ctx context.Context, want map[string][]string) (bool, error) {
	for table, columns := range want {
		rows, err := i.sql.QueryContext(ctx, `SELECT name FROM pragma_table_info(?)`, table)
		if err != nil {
			return false, fmt.Errorf("search: read the shape of %q: %w", table, err)
		}
		have := set.New[string]()
		for rows.Next() {
			var name string
			if err := rows.Scan(&name); err != nil {
				_ = rows.Close()
				return false, fmt.Errorf("search: scan the shape of %q: %w", table, err)
			}
			have.Add(name)
		}
		err = rows.Err()
		_ = rows.Close()
		if err != nil {
			return false, fmt.Errorf("search: read the shape of %q: %w", table, err)
		}
		if have.Len() == 0 {
			continue // the table is not there yet
		}
		for _, c := range columns {
			if !have.Contains(c) {
				return false, nil
			}
		}
	}
	return true, nil
}

// meta reads one meta value, returning "" when the key is unset.
func (i *Index) meta(ctx context.Context, key string) (string, error) {
	var v string
	err := i.sql.QueryRowContext(ctx, `SELECT value FROM meta WHERE key = ?`, key).Scan(&v)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("search: read meta %q: %w", key, err)
	}
	return v, nil
}

// metaInt reads one meta value as an integer. An unset key is 0. A key holding
// something that is not a number is a corrupt index rather than a zero: it is
// reported, because reading it as 0 would silently skip a schema rebuild.
func (i *Index) metaInt(ctx context.Context, key string) (int64, error) {
	v, err := i.meta(ctx, key)
	if err != nil || v == "" {
		return 0, err
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("search: meta %q holds %q, which is not a number: %w", key, v, err)
	}
	return n, nil
}

// setMeta writes one meta value.
func (i *Index) setMeta(ctx context.Context, key, value string) error {
	_, err := i.sql.ExecContext(ctx,
		`INSERT INTO meta (key, value) VALUES (?, ?)
		 ON CONFLICT(key) DO UPDATE SET value = excluded.value`, key, value)
	if err != nil {
		return fmt.Errorf("search: write meta %q: %w", key, err)
	}
	return nil
}

// RecordError stores the last indexing failure so Status can report why the
// index is behind. An empty message clears it.
func (i *Index) RecordError(ctx context.Context, msg string) error {
	return i.setMeta(ctx, metaLastError, msg)
}

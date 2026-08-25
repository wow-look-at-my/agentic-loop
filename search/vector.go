package search

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Vectors are stored as L2-normalized little-endian float32 BLOBs, so similarity is a plain dot product.

// float32Bytes is the width of one stored dimension.
const float32Bytes = 4

// encodeVector L2-normalizes v and returns it as a little-endian float32 BLOB.
// A zero-magnitude vector is rejected: it has no direction, so its similarity
// to everything is 0, and storing it would make a message permanently
// unfindable while still counting as embedded.
func encodeVector(v []float32) ([]byte, error) {
	if len(v) == 0 {
		return nil, fmt.Errorf("search: refusing to store an empty vector")
	}
	var sum float64
	for _, f := range v {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return nil, fmt.Errorf("search: refusing to store a vector containing %v", f)
		}
		sum += float64(f) * float64(f)
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return nil, fmt.Errorf("search: refusing to store a zero-magnitude vector")
	}

	out := make([]byte, len(v)*float32Bytes)
	for i, f := range v {
		scaled := float32(float64(f) / norm)
		binary.LittleEndian.PutUint32(out[i*float32Bytes:], math.Float32bits(scaled))
	}
	return out, nil
}

// dotBlob returns the cosine of a normalized query and a stored blob without allocating a decoded copy.
func dotBlob(query []float32, blob []byte) (score float64, ok bool) {
	if len(query) == 0 || len(blob) != len(query)*float32Bytes {
		return 0, false
	}
	var sum float64
	for i, q := range query {
		f := math.Float32frombits(binary.LittleEndian.Uint32(blob[i*float32Bytes:]))
		sum += float64(q) * float64(f)
	}
	return sum, true
}

// normalize returns a unit-length copy of v, for the query side (stored vectors
// are already normalized). It returns ok=false for a vector with no direction,
// which is what an embedding endpoint returning zeros looks like: scoring
// against it would rank the whole corpus at 0 and present that as a result.
func normalize(v []float32) (unit []float32, ok bool) {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	norm := math.Sqrt(sum)
	if len(v) == 0 || norm == 0 || math.IsNaN(norm) || math.IsInf(norm, 0) {
		return nil, false
	}
	unit = make([]float32, len(v))
	for i, f := range v {
		unit[i] = float32(float64(f) / norm)
	}
	return unit, true
}

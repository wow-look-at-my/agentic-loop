package loop

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func TestVersionNonEmpty(t *testing.T) {
	require.NotEqual(t, "", Version)

}

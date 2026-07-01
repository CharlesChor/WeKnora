package types

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestJiebaDictDirExists(t *testing.T) {
	t.Parallel()

	dictDir := t.TempDir()
	for _, name := range jiebaDictFiles {
		err := os.WriteFile(filepath.Join(dictDir, name), []byte("test"), 0o644)
		require.NoError(t, err)
	}

	require.True(t, jiebaDictDirExists(dictDir))

	require.NoError(t, os.Remove(filepath.Join(dictDir, jiebaDictFiles[0])))
	require.False(t, jiebaDictDirExists(dictDir))
}

func TestFindBundledJiebaDictDir(t *testing.T) {
	t.Parallel()

	baseDir := t.TempDir()
	dictDir := filepath.Join(baseDir, "jieba-dict")
	require.NoError(t, os.MkdirAll(dictDir, 0o755))
	for _, name := range jiebaDictFiles {
		require.NoError(t, os.WriteFile(filepath.Join(dictDir, name), []byte("test"), 0o644))
	}

	require.Equal(t, dictDir, findBundledJiebaDictDir(baseDir))
}

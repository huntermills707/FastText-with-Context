import struct
import numpy as np
from typing import Dict, List, Tuple, Optional


class FastTextContext:
    """
    Python loader for FastTextContext binary models.

    Changes vs previous version:
    - input_matrix: word-level embedding (vocab_size x dim) loaded between ngram and metadata.
    - get_word_vector: returns word_embedding + ngram_sum (OOV falls back to ngrams only).
    - get_combined_vector: averages word vectors and metadata vectors within each group,
      L2-normalises each group, sums the two, then L2-normalises the result. This gives
      words and metadata strict 50/50 directional contribution regardless of group size.
    """

    def __init__(self):
        self.dim: int = 0
        self.min_n: int = 0
        self.max_n: int = 0
        self.threshold: int = 0
        self.window_size: int = 0

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.metadata2idx: Dict[str, int] = {}
        self.idx2metadata: Dict[int, str] = {}

        self.output_matrix:   Optional[np.ndarray] = None
        self.ngram_matrix:    Optional[np.ndarray] = None
        self.input_matrix:    Optional[np.ndarray] = None   # word-level embeddings
        self.metadata_matrix: Optional[np.ndarray] = None

        self.word_codes: List[List[int]] = []
        self.word_paths: List[List[int]] = []

    def load_model(self, filename: str) -> None:
        """Load a binary model produced by the C++ train binary."""
        with open(filename, 'rb') as f:
            self.dim        = struct.unpack('i', f.read(4))[0]
            self.min_n      = struct.unpack('i', f.read(4))[0]
            self.max_n      = struct.unpack('i', f.read(4))[0]
            self.threshold  = struct.unpack('i', f.read(4))[0]
            self.window_size = struct.unpack('i', f.read(4))[0]

            vocab_size  = struct.unpack('i', f.read(4))[0]
            meta_size   = struct.unpack('i', f.read(4))[0]
            ngram_size  = struct.unpack('i', f.read(4))[0]
            output_size = struct.unpack('i', f.read(4))[0]

            self.word2idx, self.idx2word = {}, {}
            for _ in range(vocab_size):
                word_len = struct.unpack('I', f.read(4))[0]
                word     = f.read(word_len).decode('utf-8')
                idx      = struct.unpack('i', f.read(4))[0]
                self.word2idx[word] = idx
                self.idx2word[idx]  = word

            self.metadata2idx, self.idx2metadata = {}, {}
            for _ in range(meta_size):
                meta_len = struct.unpack('I', f.read(4))[0]
                meta     = f.read(meta_len).decode('utf-8')
                idx      = struct.unpack('i', f.read(4))[0]
                self.metadata2idx[meta] = idx
                self.idx2metadata[idx]  = meta

            # Matrix order (C++ save): output, ngram, input, metadata.
            self.output_matrix = np.frombuffer(
                f.read(output_size * self.dim * 4), dtype=np.float32
            ).reshape(output_size, self.dim).copy()

            self.ngram_matrix = np.frombuffer(
                f.read(ngram_size * self.dim * 4), dtype=np.float32
            ).reshape(ngram_size, self.dim).copy()

            self.input_matrix = np.frombuffer(
                f.read(vocab_size * self.dim * 4), dtype=np.float32
            ).reshape(vocab_size, self.dim).copy()

            self.metadata_matrix = (
                np.frombuffer(f.read(meta_size * self.dim * 4), dtype=np.float32)
                .reshape(meta_size, self.dim).copy()
                if meta_size > 0 else np.zeros((0, self.dim), dtype=np.float32)
            )

            self.word_codes = self._read_int_vec_vec(f, vocab_size)
            self.word_paths = self._read_int_vec_vec(f, vocab_size)

        print(f"Model loaded: {vocab_size} words, {meta_size} metadata fields, "
              f"dim={self.dim}, window={self.window_size}")

    @staticmethod
    def _read_int_vec_vec(f, size: int) -> List[List[int]]:
        result = []
        for _ in range(size):
            length = struct.unpack('I', f.read(4))[0]
            if length > 0:
                result.append(list(struct.unpack(f'{length}i', f.read(length * 4))))
            else:
                result.append([])
        return result

    def _hash(self, s: str) -> int:
        h = 14695981039346656037
        for c in s:
            h ^= ord(c)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    def _get_ngram_indices(self, word: str) -> List[int]:
        indices  = []
        bordered = '<' + word + '>'
        n_buckets = len(self.ngram_matrix)
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(bordered) - n + 1):
                indices.append(self._hash(bordered[i:i+n]) % n_buckets)
        return indices

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        """L2-normalise v in-place and return it. Zero vectors pass through unchanged."""
        n = np.linalg.norm(v)
        if n > 1e-8:
            v /= n
        return v

    def get_word_vector(self, word: str) -> np.ndarray:
        """word_embedding (if in vocab) + sum(ngram_embeddings).

        Out-of-vocabulary words fall back to ngrams only, preserving
        morphological coverage.
        """
        vec = np.zeros(self.dim, dtype=np.float32)

        if word in self.word2idx:
            vec += self.input_matrix[self.word2idx[word]]

        ngram_idx = self._get_ngram_indices(word)
        if ngram_idx:
            vec += self.ngram_matrix[ngram_idx].sum(axis=0)

        return vec

    def get_metadata_vector(self, metadata: str) -> np.ndarray:
        if metadata in self.metadata2idx:
            return self.metadata_matrix[self.metadata2idx[metadata]].copy()
        return np.zeros(self.dim, dtype=np.float32)

    def get_combined_vector(self, words: List[str],
                            metadata: Optional[List[str]] = None) -> np.ndarray:
        """Average word vectors and average metadata vectors separately, L2-normalise
        each group, sum the two normalised averages, then L2-normalise the result.

        This gives words and metadata strict 50/50 directional contribution regardless
        of how many tokens are in each group. If one group is empty its normalised
        average is the zero vector, so the result collapses to a normalised version of
        the other group alone.
        """
        # Average word vectors.
        word_avg = np.zeros(self.dim, dtype=np.float32)
        for word in words:
            word_avg += self.get_word_vector(word)
        if words:
            word_avg /= len(words)

        # Average metadata vectors (only over fields present in the vocabulary).
        meta_avg = np.zeros(self.dim, dtype=np.float32)
        if metadata:
            meta_count = 0
            for meta in metadata:
                if meta in self.metadata2idx:
                    meta_avg += self.metadata_matrix[self.metadata2idx[meta]]
                    meta_count += 1
            if meta_count > 0:
                meta_avg /= meta_count

        # Normalise each group before summing so neither dominates by magnitude.
        self._l2norm(word_avg)
        self._l2norm(meta_avg)

        combined = word_avg + meta_avg
        self._l2norm(combined)

        return combined

    def get_nearest_neighbors(self, words: List[str],
                               metadata: Optional[List[str]] = None,
                               k: int = 10) -> List[Tuple[str, float]]:
        query = self.get_combined_vector(words, metadata)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            print("Warning: query vector has near-zero magnitude.")
            return []

        results = []
        for idx, word in self.idx2word.items():
            wv   = self.get_word_vector(word)
            norm = np.linalg.norm(wv)
            if norm < 1e-8:
                continue
            sim = float(np.dot(query, wv) / (query_norm * norm))
            results.append((word, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def compare_vectors(self, words1: List[str], metadata1: Optional[List[str]],
                        words2: List[str], metadata2: Optional[List[str]]) -> dict:
        v1 = self.get_combined_vector(words1, metadata1)
        v2 = self.get_combined_vector(words2, metadata2)
        # Both vectors are L2-normalised by get_combined_vector, so the dot
        # product is the cosine similarity directly.
        cos = float(np.dot(v1, v2))
        return {
            'cosine_similarity':  cos,
            'cosine_distance':    1.0 - cos,
            'euclidean_distance': float(np.linalg.norm(v1 - v2)),
        }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fasttext_context.py <model.bin> [word]")
        sys.exit(1)

    ft = FastTextContext()
    ft.load_model(sys.argv[1])

    if len(sys.argv) > 2:
        word = sys.argv[2]
        vec  = ft.get_word_vector(word)
        print(f"\nWord vector for '{word}':")
        print(f"  Norm:          {np.linalg.norm(vec):.6f}")
        print(f"  First 10 dims: {vec[:10]}")
        for w, s in ft.get_nearest_neighbors([word], k=5):
            print(f"  {w}: {s:.4f}")

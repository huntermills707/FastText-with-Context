import struct
import numpy as np
from typing import Dict, List, Tuple, Optional


class FastTextContext:
    """
    Python loader for FastTextContext binary models.

    Composition rule (matching C++ training and inference):
      word_part = avg(word_emb_i + sum(ngrams_i))
      gate      = sigmoid(gate_bias + word_part)         [element-wise; OOV-safe via ngrams]
      meta_part = sum_k(alpha_k * gate * meta_emb_k)
      combined  = word_part + meta_part
      result    = L2_normalise(combined)

    Using word_part rather than raw word_avg for the gate signal means OOV
    words get a morphologically-informed gate via their ngram embeddings
    instead of collapsing to sigmoid(gate_bias). Only one L2 normalisation is
    applied to the final composite.
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
        self.input_matrix:    Optional[np.ndarray] = None
        self.metadata_matrix: Optional[np.ndarray] = None

        # Gated composition parameters.
        self.gate_bias: Optional[np.ndarray] = None   # shape (dim,)
        self.alpha: Optional[np.ndarray] = None        # shape (meta_size,)

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

            # Gated composition parameters (NEW in this model format).
            self.gate_bias = np.frombuffer(
                f.read(self.dim * 4), dtype=np.float32
            ).copy()

            self.alpha = (
                np.frombuffer(f.read(meta_size * 4), dtype=np.float32).copy()
                if meta_size > 0 else np.zeros(0, dtype=np.float32)
            )

            self.word_codes = self._read_int_vec_vec(f, vocab_size)
            self.word_paths = self._read_int_vec_vec(f, vocab_size)

        print(f"Model loaded: {vocab_size} words, {meta_size} metadata fields, "
              f"dim={self.dim}, window={self.window_size}")
        print(f"  gate_bias: mean={self.gate_bias.mean():.4f}, "
              f"std={self.gate_bias.std():.4f}")
        if meta_size > 0:
            print(f"  alpha: min={self.alpha.min():.4f}, max={self.alpha.max():.4f}, "
                  f"mean={self.alpha.mean():.4f}")

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

        OOV words fall back to ngrams only, preserving morphological coverage.
        """
        vec = np.zeros(self.dim, dtype=np.float32)

        if word in self.word2idx:
            vec += self.input_matrix[self.word2idx[word]]

        ngram_idx = self._get_ngram_indices(word)
        if ngram_idx:
            vec += self.ngram_matrix[ngram_idx].sum(axis=0)

        return vec

    def get_metadata_vector(self, metadata: str) -> np.ndarray:
        """Raw metadata embedding (no gating) — for diagnostic purposes."""
        if metadata in self.metadata2idx:
            return self.metadata_matrix[self.metadata2idx[metadata]].copy()
        return np.zeros(self.dim, dtype=np.float32)

    def _compute_gate(self, word_part: np.ndarray) -> np.ndarray:
        """Compute gate = sigmoid(gate_bias + word_part) element-wise."""
        if self.gate_bias is not None:
            x = np.clip(self.gate_bias + word_part, -20.0, 20.0)
        else:
            x = np.clip(word_part, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x))

    def get_combined_vector(self, words: List[str],
                            metadata: Optional[List[str]] = None) -> np.ndarray:
        """Gated composition: avg(word_vectors) + gated_metadata, then L2-normalise.

        word_part = avg(word_emb_i + sum(ngrams_i))
        gate      = sigmoid(gate_bias + word_part)      [element-wise; OOV-safe via ngrams]
        meta_part = sum_k(alpha_k * gate * meta_emb_k)
        combined  = word_part + meta_part
        result    = L2_normalise(combined)

        Matches the C++ training and inference composition exactly.
        """
        word_part = np.zeros(self.dim, dtype=np.float32)

        for word in words:
            word_part += self.get_word_vector(word)

        if words:
            word_part /= len(words)

        # Gate from word_part: covers OOV words via ngram embeddings.
        gate = self._compute_gate(word_part)

        # Gated metadata: sum_k(alpha_k * gate * meta_emb_k).
        meta_part = np.zeros(self.dim, dtype=np.float32)
        if metadata:
            for meta in metadata:
                if meta in self.metadata2idx:
                    idx = self.metadata2idx[meta]
                    a = float(self.alpha[idx]) if (self.alpha is not None and len(self.alpha) > idx) else 1.0
                    meta_part += a * gate * self.metadata_matrix[idx]

        combined = word_part + meta_part
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
        # Both vectors are L2-normalised, so dot product = cosine similarity.
        cos = float(np.dot(v1, v2))
        return {
            'cosine_similarity':  cos,
            'cosine_distance':    1.0 - cos,
            'euclidean_distance': float(np.linalg.norm(v1 - v2)),
        }

    def print_gate_stats(self, word: str) -> None:
        """Diagnostic: print the gate vector for a given word.

        Uses word_part = word_emb + ngrams, matching the actual inference path.
        OOV words will show a non-trivial gate driven by their ngram embeddings.
        """
        word_part = self.get_word_vector(word)
        gate = self._compute_gate(word_part)
        print(f"Gate for '{word}': min={gate.min():.4f} max={gate.max():.4f} "
              f"mean={gate.mean():.4f}")

    def print_alpha_stats(self) -> None:
        """Diagnostic: print per-field alpha weights sorted by magnitude."""
        if self.alpha is None or len(self.alpha) == 0:
            print("No alpha weights available.")
            return
        print("Per-field alpha weights:")
        pairs = [(self.idx2metadata.get(i, f"<{i}>"), float(self.alpha[i]))
                 for i in range(len(self.alpha))]
        for field, a in sorted(pairs, key=lambda x: abs(x[1] - 1.0), reverse=True):
            print(f"  {field}: {a:.6f}  (delta from 1.0: {a-1.0:+.6f})")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fasttext_context.py <model.bin> [word] [--meta field1 field2 ...]")
        sys.exit(1)

    ft = FastTextContext()
    ft.load_model(sys.argv[1])

    # Parse optional word and metadata arguments.
    words = []
    metadata = []
    meta_flag = False
    for arg in sys.argv[2:]:
        if arg == '--meta':
            meta_flag = True
        elif meta_flag:
            metadata.append(arg)
        else:
            words.append(arg)

    if words:
        print(f"\nWord vector for '{words[0]}':")
        vec = ft.get_word_vector(words[0])
        print(f"  Norm:          {np.linalg.norm(vec):.6f}")
        print(f"  First 10 dims: {vec[:10]}")

        ft.print_gate_stats(words[0])

        print(f"\nNearest neighbors (words={words}, metadata={metadata}):")
        for w, s in ft.get_nearest_neighbors(words, metadata or None, k=5):
            print(f"  {w}: {s:.4f}")

    ft.print_alpha_stats()

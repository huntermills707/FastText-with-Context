import struct
import numpy as np
from typing import Dict, List, Tuple, Optional


class FastTextContext:
    """
    Python implementation for loading and querying FastTextContext models.
    
    Supports:
    - Word vectors (with subword n-gram contributions)
    - Metadata vectors (author, domain, etc.)
    - Combined vectors (word + metadata)
    - Nearest neighbor search
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
        
        self.input_matrix: Optional[np.ndarray] = None      # (vocab_size, dim)
        self.output_matrix: Optional[np.ndarray] = None     # (output_size, dim)
        self.ngram_matrix: Optional[np.ndarray] = None      # (ngram_buckets, dim)
        self.metadata_matrix: Optional[np.ndarray] = None   # (meta_size, dim)
        
        self.word_codes: List[List[int]] = []
        self.word_paths: List[List[int]] = []
    
    def load_model(self, filename: str) -> None:
        """Load a binary model file saved by the C++ train binary."""
        with open(filename, 'rb') as f:
            # Read header (9 ints now, including window_size)
            self.dim = struct.unpack('i', f.read(4))[0]
            self.min_n = struct.unpack('i', f.read(4))[0]
            self.max_n = struct.unpack('i', f.read(4))[0]
            self.threshold = struct.unpack('i', f.read(4))[0]
            self.window_size = struct.unpack('i', f.read(4))[0]
            
            vocab_size = struct.unpack('i', f.read(4))[0]
            meta_size = struct.unpack('i', f.read(4))[0]
            ngram_size = struct.unpack('i', f.read(4))[0]
            output_size = struct.unpack('i', f.read(4))[0]
            
            # Read word vocabulary
            self.word2idx = {}
            self.idx2word = {}
            for _ in range(vocab_size):
                word_len = struct.unpack('I', f.read(4))[0]
                word = f.read(word_len).decode('utf-8')
                idx = struct.unpack('i', f.read(4))[0]
                self.word2idx[word] = idx
                self.idx2word[idx] = word
            
            # Read metadata vocabulary
            self.metadata2idx = {}
            self.idx2metadata = {}
            for _ in range(meta_size):
                meta_len = struct.unpack('I', f.read(4))[0]
                meta = f.read(meta_len).decode('utf-8')
                idx = struct.unpack('i', f.read(4))[0]
                self.metadata2idx[meta] = idx
                self.idx2metadata[idx] = meta
            
            # Read matrices (row-major, float32)
            self.input_matrix = np.frombuffer(
                f.read(vocab_size * self.dim * 4), dtype=np.float32
            ).reshape(vocab_size, self.dim).copy()
            
            self.output_matrix = np.frombuffer(
                f.read(output_size * self.dim * 4), dtype=np.float32
            ).reshape(output_size, self.dim).copy()
            
            self.ngram_matrix = np.frombuffer(
                f.read(ngram_size * self.dim * 4), dtype=np.float32
            ).reshape(ngram_size, self.dim).copy()
            
            self.metadata_matrix = np.frombuffer(
                f.read(meta_size * self.dim * 4), dtype=np.float32
            ).reshape(meta_size, self.dim).copy() if meta_size > 0 else np.zeros((0, self.dim), dtype=np.float32)
            
            # Read Huffman codes and paths
            self.word_codes = []
            self.word_paths = []
            for _ in range(vocab_size):
                code_len = struct.unpack('I', f.read(4))[0]
                if code_len > 0:
                    codes = list(struct.unpack(f'{code_len}i', f.read(code_len * 4)))
                else:
                    codes = []
                self.word_codes.append(codes)
            
            for _ in range(vocab_size):
                path_len = struct.unpack('I', f.read(4))[0]
                if path_len > 0:
                    path = list(struct.unpack(f'{path_len}i', f.read(path_len * 4)))
                else:
                    path = []
                self.word_paths.append(path)
        
        print(f"Model loaded: {vocab_size} words, {meta_size} metadata fields, dim={self.dim}, window={self.window_size}")
    
    def _hash(self, s: str) -> int:
        """FNV-1a hash for n-gram bucketing."""
        h = 14695981039346656037
        for c in s:
            h ^= ord(c)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h
    
    def _get_ngram_indices(self, word: str) -> List[int]:
        """Get n-gram bucket indices for a word."""
        indices = []
        bordered = '<' + word + '>'
        ngram_buckets = len(self.ngram_matrix)
        
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(bordered) - n + 1):
                ngram = bordered[i:i+n]
                h = self._hash(ngram)
                idx = h % ngram_buckets
                indices.append(idx)
        
        return indices
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get the word vector including subword n-gram contributions.
        
        Args:
            word: The word to look up
            
        Returns:
            numpy array of shape (dim,)
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        
        # Add word embedding if in vocabulary
        if word in self.word2idx:
            idx = self.word2idx[word]
            vec += self.input_matrix[idx]
        
        # Add n-gram embeddings
        ngram_indices = self._get_ngram_indices(word)
        if ngram_indices:
            vec += self.ngram_matrix[ngram_indices].sum(axis=0)
        
        return vec
    
    def get_metadata_vector(self, metadata: str) -> np.ndarray:
        """
        Get the embedding for a single metadata field.
        
        Args:
            metadata: The metadata field value (e.g., "alice", "tech")
            
        Returns:
            numpy array of shape (dim,), or zeros if not found
        """
        if metadata in self.metadata2idx:
            idx = self.metadata2idx[metadata]
            return self.metadata_matrix[idx].copy()
        return np.zeros(self.dim, dtype=np.float32)
    
    def compute_metadata_vector(self, metadata_list: List[str]) -> np.ndarray:
        """
        Compute the summed metadata vector from multiple metadata fields.
        
        Args:
            metadata_list: List of metadata field values
            
        Returns:
            numpy array of shape (dim,), summed over valid metadata
        """
        if not metadata_list:
            return np.zeros(self.dim, dtype=np.float32)
        
        vec = np.zeros(self.dim, dtype=np.float32)
        
        for meta in metadata_list:
            if meta in self.metadata2idx:
                idx = self.metadata2idx[meta]
                vec += self.metadata_matrix[idx]
        
        return vec
    
    def get_combined_vector(self, words: List[str], metadata: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute a combined vector from words and metadata.
        
        The result is normalized to unit length.
        
        Args:
            words: List of words to combine
            metadata: Optional list of metadata fields
            
        Returns:
            Normalized numpy array of shape (dim,)
        """
        combined = np.zeros(self.dim, dtype=np.float32)
        
        # Sum word vectors
        for word in words:
            combined += self.get_word_vector(word)
        
        # Add metadata vector
        if metadata:
            combined += self.compute_metadata_vector(metadata)
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined /= norm
        
        return combined
    
    def get_nearest_neighbors(self, words: List[str], metadata: Optional[List[str]] = None, 
                               k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the k nearest neighbors to a combined word+metadata query.
        
        Args:
            words: List of query words
            metadata: Optional list of metadata fields
            k: Number of neighbors to return
            
        Returns:
            List of (word, similarity_score) tuples, sorted by similarity descending
        """
        query_vec = self.get_combined_vector(words, metadata)
        
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-8:
            print("Warning: Query vector has near-zero magnitude")
            return []
        
        # Compute similarities with all words (using input_matrix only for speed)
        word_norms = np.linalg.norm(self.input_matrix, axis=1, keepdims=True)
        word_norms = np.maximum(word_norms, 1e-8)
        
        normalized_input = self.input_matrix / word_norms
        similarities = normalized_input @ query_vec
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            word = self.idx2word.get(idx, f"<idx:{idx}>")
            sim = float(similarities[idx])
            results.append((word, sim))
        
        return results
    
    def compare_vectors(self, words1: List[str], metadata1: Optional[List[str]],
                        words2: List[str], metadata2: Optional[List[str]]) -> dict:
        """
        Compare two word+metadata combinations.
        
        Returns:
            Dictionary with cosine_similarity, cosine_distance, euclidean_distance
        """
        vec1 = self.get_combined_vector(words1, metadata1)
        vec2 = self.get_combined_vector(words2, metadata2)
        
        # Vectors are already normalized from get_combined_vector
        cosine_sim = float(np.dot(vec1, vec2))
        euclidean_dist = float(np.linalg.norm(vec1 - vec2))
        
        return {
            'cosine_similarity': cosine_sim,
            'cosine_distance': 1.0 - cosine_sim,
            'euclidean_distance': euclidean_dist
        }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fasttext_context.py <model.bin>")
        print("       python fasttext_context.py <model.bin> <word>")
        sys.exit(1)
    
    model_file = sys.argv[1]
    
    ft = FastTextContext()
    ft.load_model(model_file)
    
    if len(sys.argv) > 2:
        word = sys.argv[2]
        vec = ft.get_word_vector(word)
        print(f"\nWord vector for '{word}':")
        print(f"  Shape: {vec.shape}")
        print(f"  Norm: {np.linalg.norm(vec):.6f}")
        print(f"  First 10 dims: {vec[:10]}")
        
        neighbors = ft.get_nearest_neighbors([word], k=5)
        print(f"\nTop 5 neighbors:")
        for w, sim in neighbors:
            print(f"  {w}: {sim:.4f}")

import struct
import numpy as np
from typing import Dict, List, Tuple, Optional


class FastTextContext:
    """
    Python implementation for loading and querying FastTextContext models.
    
    Supports:
    - Word vectors (with subword n-gram contributions)
    - Context vectors
    - Combined vectors (word + context)
    - Nearest neighbor search
    """
    
    def __init__(self):
        self.dim: int = 0
        self.min_n: int = 0
        self.max_n: int = 0
        self.threshold: int = 0
        
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.context2idx: Dict[str, int] = {}
        self.idx2context: Dict[int, str] = {}
        
        self.input_matrix: Optional[np.ndarray] = None      # (vocab_size, dim)
        self.output_matrix: Optional[np.ndarray] = None     # (output_size, dim)
        self.ngram_matrix: Optional[np.ndarray] = None      # (ngram_buckets, dim)
        self.context_matrix: Optional[np.ndarray] = None    # (ctx_size, dim)
        
        self.word_codes: List[List[int]] = []
        self.word_paths: List[List[int]] = []
    
    def load_model(self, filename: str) -> None:
        """Load a binary model file saved by the C++ train binary."""
        with open(filename, 'rb') as f:
            # Read header (8 ints)
            self.dim = struct.unpack('i', f.read(4))[0]
            self.min_n = struct.unpack('i', f.read(4))[0]
            self.max_n = struct.unpack('i', f.read(4))[0]
            self.threshold = struct.unpack('i', f.read(4))[0]
            
            vocab_size = struct.unpack('i', f.read(4))[0]
            ctx_size = struct.unpack('i', f.read(4))[0]
            ngram_size = struct.unpack('i', f.read(4))[0]
            output_size = struct.unpack('i', f.read(4))[0]
            
            # Read word vocabulary
            self.word2idx = {}
            self.idx2word = {}
            for _ in range(vocab_size):
                word_len = struct.unpack('I', f.read(4))[0]  # uint32_t
                word = f.read(word_len).decode('utf-8')
                idx = struct.unpack('i', f.read(4))[0]
                self.word2idx[word] = idx
                self.idx2word[idx] = word
            
            # Read context vocabulary
            self.context2idx = {}
            self.idx2context = {}
            for _ in range(ctx_size):
                ctx_len = struct.unpack('I', f.read(4))[0]
                ctx = f.read(ctx_len).decode('utf-8')
                idx = struct.unpack('i', f.read(4))[0]
                self.context2idx[ctx] = idx
                self.idx2context[idx] = ctx
            
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
            
            self.context_matrix = np.frombuffer(
                f.read(ctx_size * self.dim * 4), dtype=np.float32
            ).reshape(ctx_size, self.dim).copy() if ctx_size > 0 else np.zeros((0, self.dim), dtype=np.float32)
            
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
        
        print(f"Model loaded: {vocab_size} words, {ctx_size} contexts, dim={self.dim}")
    
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
    
    def get_context_vector(self, context: str) -> np.ndarray:
        """
        Get the embedding for a single context field.
        
        Args:
            context: The context field value (e.g., "alice", "tech")
            
        Returns:
            numpy array of shape (dim,), or zeros if not found
        """
        if context in self.context2idx:
            idx = self.context2idx[context]
            return self.context_matrix[idx].copy()
        return np.zeros(self.dim, dtype=np.float32)
    
    def compute_context_vector(self, contexts: List[str]) -> np.ndarray:
        """
        Compute the average context vector from multiple context fields.
        
        Args:
            contexts: List of context field values
            
        Returns:
            numpy array of shape (dim,), averaged over valid contexts
        """
        if not contexts:
            return np.zeros(self.dim, dtype=np.float32)
        
        vec = np.zeros(self.dim, dtype=np.float32)
        count = 0
        
        for ctx in contexts:
            if ctx in self.context2idx:
                idx = self.context2idx[ctx]
                vec += self.context_matrix[idx]
                count += 1
        
        if count > 0:
            vec /= count
        
        return vec
    
    def get_combined_vector(self, words: List[str], contexts: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute a combined vector from words and contexts.
        
        The result is normalized to unit length.
        
        Args:
            words: List of words to combine
            contexts: Optional list of context fields
            
        Returns:
            Normalized numpy array of shape (dim,)
        """
        combined = np.zeros(self.dim, dtype=np.float32)
        
        # Sum word vectors
        for word in words:
            combined += self.get_word_vector(word)
        
        # Add context vector
        if contexts:
            combined += self.compute_context_vector(contexts)
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined /= norm
        
        return combined
    
    def get_nearest_neighbors(self, words: List[str], contexts: Optional[List[str]] = None, 
                               k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the k nearest neighbors to a combined word+context query.
        
        Args:
            words: List of query words
            contexts: Optional list of context fields
            k: Number of neighbors to return
            
        Returns:
            List of (word, similarity_score) tuples, sorted by similarity descending
        """
        query_vec = self.get_combined_vector(words, contexts)
        
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-8:
            print("Warning: Query vector has near-zero magnitude")
            return []
        
        # Compute similarities with all words (using input_matrix only for speed)
        # Note: This doesn't include n-gram contributions for candidate words
        # For full accuracy, use get_word_vector for each candidate
        
        word_norms = np.linalg.norm(self.input_matrix, axis=1, keepdims=True)
        word_norms = np.maximum(word_norms, 1e-8)  # Avoid division by zero
        
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
    
    def compare_vectors(self, words1: List[str], contexts1: Optional[List[str]],
                        words2: List[str], contexts2: Optional[List[str]]) -> dict:
        """
        Compare two word+context combinations.
        
        Returns:
            Dictionary with cosine_similarity, cosine_distance, euclidean_distance
        """
        vec1 = self.get_combined_vector(words1, contexts1)
        vec2 = self.get_combined_vector(words2, contexts2)
        
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

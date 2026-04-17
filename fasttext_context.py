import struct
import numpy as np
from typing import Dict, List, Tuple, Optional


class FastTextContext:
    """
    Python loader for FastTextContext binary models (concatenation+projection architecture).

    Architecture:
      word_part    = avg(word_emb_i + sum(ngrams_i))     [d_word]
      patient_part = avg(patient_emb_i)                  [d_patient]  (zero if absent)
      encounter_part = avg(encounter_emb_i)                [d_encounter] (zero if absent)
      concat       = [word_part ; patient_part ; encounter_part]  [concat_dim]
      result       = L2_normalise(W_proj @ concat)       [d_out]

    Adding a fourth group (e.g. outcome) requires extending concat_dim and W_proj.
    """

    def __init__(self):
        self.d_word: int = 0
        self.d_patient: int = 0
        self.d_encounter: int = 0
        self.d_out: int = 0
        self.concat_dim: int = 0
        self.min_n: int = 0
        self.max_n: int = 0
        self.threshold: int = 0
        self.window_size: int = 0

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.patient2idx: Dict[str, int] = {}
        self.idx2patient: Dict[int, str] = {}
        self.encounter2idx: Dict[str, int] = {}
        self.idx2encounter: Dict[int, str] = {}

        self.input_matrix:    Optional[np.ndarray] = None   # vocab_size x d_word
        self.ngram_matrix:    Optional[np.ndarray] = None   # ngram_size x d_word
        self.output_matrix:   Optional[np.ndarray] = None   # hs_nodes x d_out
        self.W_proj:          Optional[np.ndarray] = None   # d_out x concat_dim
        self.patient_matrix:  Optional[np.ndarray] = None   # patient_size x d_patient
        self.encounter_matrix: Optional[np.ndarray] = None   # encounter_size x d_encounter

        self.word_codes: List[List[int]] = []
        self.word_paths: List[List[int]] = []

    def load_model(self, filename: str) -> None:
        """Load a binary model produced by the C++ train binary."""
        with open(filename, 'rb') as f:
            self.d_word      = struct.unpack('i', f.read(4))[0]
            self.d_patient   = struct.unpack('i', f.read(4))[0]
            self.d_encounter = struct.unpack('i', f.read(4))[0]
            self.d_out       = struct.unpack('i', f.read(4))[0]
            self.min_n       = struct.unpack('i', f.read(4))[0]
            self.max_n       = struct.unpack('i', f.read(4))[0]
            self.threshold   = struct.unpack('i', f.read(4))[0]
            self.window_size = struct.unpack('i', f.read(4))[0]

            self.concat_dim = self.d_word + self.d_patient + self.d_encounter

            vocab_size     = struct.unpack('i', f.read(4))[0]
            patient_size   = struct.unpack('i', f.read(4))[0]
            encounter_size = struct.unpack('i', f.read(4))[0]
            ngram_size     = struct.unpack('i', f.read(4))[0]
            output_size    = struct.unpack('i', f.read(4))[0]

            # Word vocabulary.
            self.word2idx, self.idx2word = {}, {}
            for _ in range(vocab_size):
                word_len = struct.unpack('I', f.read(4))[0]
                word     = f.read(word_len).decode('utf-8')
                idx      = struct.unpack('i', f.read(4))[0]
                self.word2idx[word] = idx
                self.idx2word[idx]  = word

            # Patient vocabulary.
            self.patient2idx, self.idx2patient = {}, {}
            for _ in range(patient_size):
                field_len = struct.unpack('I', f.read(4))[0]
                field     = f.read(field_len).decode('utf-8')
                idx       = struct.unpack('i', f.read(4))[0]
                self.patient2idx[field] = idx
                self.idx2patient[idx]   = field

            # Encounter vocabulary.
            self.encounter2idx, self.idx2encounter = {}, {}
            for _ in range(encounter_size):
                field_len = struct.unpack('I', f.read(4))[0]
                field     = f.read(field_len).decode('utf-8')
                idx       = struct.unpack('i', f.read(4))[0]
                self.encounter2idx[field] = idx
                self.idx2encounter[idx]   = field

            # Matrix order: output, ngram, input, W_proj, patient, encounter.
            self.output_matrix = np.frombuffer(
                f.read(output_size * self.d_out * 4), dtype=np.float32
            ).reshape(output_size, self.d_out).copy()

            self.ngram_matrix = np.frombuffer(
                f.read(ngram_size * self.d_word * 4), dtype=np.float32
            ).reshape(ngram_size, self.d_word).copy()

            self.input_matrix = np.frombuffer(
                f.read(vocab_size * self.d_word * 4), dtype=np.float32
            ).reshape(vocab_size, self.d_word).copy()

            self.W_proj = np.frombuffer(
                f.read(self.d_out * self.concat_dim * 4), dtype=np.float32
            ).reshape(self.d_out, self.concat_dim).copy()

            self.patient_matrix = (
                np.frombuffer(f.read(patient_size * self.d_patient * 4), dtype=np.float32)
                .reshape(patient_size, self.d_patient).copy()
                if patient_size > 0 else np.zeros((0, self.d_patient), dtype=np.float32)
            )

            self.encounter_matrix = (
                np.frombuffer(f.read(encounter_size * self.d_encounter * 4), dtype=np.float32)
                .reshape(encounter_size, self.d_encounter).copy()
                if encounter_size > 0 else np.zeros((0, self.d_encounter), dtype=np.float32)
            )

            self.word_codes = self._read_int_vec_vec(f, vocab_size)
            self.word_paths = self._read_int_vec_vec(f, vocab_size)

        print(f"Model loaded: {vocab_size} words, {patient_size} patient group fields, "
              f"{encounter_size} encounter group fields")
        print(f"  d_word={self.d_word} d_patient={self.d_patient} d_encounter={self.d_encounter} d_out={self.d_out} "
              f"concat_dim={self.concat_dim}")
        print(f"  W_proj shape: {self.W_proj.shape}  "
              f"(Frobenius norm: {np.linalg.norm(self.W_proj):.4f})")

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
        indices   = []
        bordered  = '<' + word + '>'
        n_buckets = len(self.ngram_matrix)
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(bordered) - n + 1):
                indices.append(self._hash(bordered[i:i+n]) % n_buckets)
        return indices

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n > 1e-8:
            v = v / n
        return v

    def get_word_vector(self, word: str) -> np.ndarray:
        """word_embedding (if in vocab) + sum(ngram_embeddings). Dimension: d_word.

        OOV words fall back to ngrams only.
        """
        vec = np.zeros(self.d_word, dtype=np.float32)

        if word in self.word2idx:
            vec += self.input_matrix[self.word2idx[word]]

        ngram_idx = self._get_ngram_indices(word)
        if ngram_idx:
            vec += self.ngram_matrix[ngram_idx].sum(axis=0)

        return vec

    def get_projected_word_vector(self, word: str) -> np.ndarray:
        """Project a word-only vector (patient/encounter regions zeroed) through W_proj.

        Returns d_out-dimensional vector. Not L2-normalised.
        """
        concat = np.zeros(self.concat_dim, dtype=np.float32)
        concat[:self.d_word] = self.get_word_vector(word)
        return self.W_proj @ concat

    def get_combined_vector(self,
                             words: List[str],
                             patient_group: Optional[List[str]] = None,
                             encounter_group: Optional[List[str]] = None) -> np.ndarray:
        """Concatenation+projection composition, L2-normalised.

        word_part    = avg(word_emb_i + sum(ngrams_i))      [d_word]
        patient_part = avg(patient_emb_i)                   [d_patient]
        encounter_part = avg(encounter_emb_i)                 [d_encounter]
        concat       = [word_part ; patient_part ; encounter_part]
        result       = L2_normalise(W_proj @ concat)        [d_out]
        """
        concat = np.zeros(self.concat_dim, dtype=np.float32)

        # Word part.
        if words:
            for w in words:
                concat[:self.d_word] += self.get_word_vector(w)
            concat[:self.d_word] /= len(words)

        # Patient part.
        if patient_group:
            n = 0
            for field in patient_group:
                if field in self.patient2idx:
                    concat[self.d_word:self.d_word+self.d_patient] += \
                        self.patient_matrix[self.patient2idx[field]]
                    n += 1
            if n > 1:
                concat[self.d_word:self.d_word+self.d_patient] /= n

        # Encounter part.
        if encounter_group:
            offset = self.d_word + self.d_patient
            n = 0
            for field in encounter_group:
                if field in self.encounter2idx:
                    concat[offset:offset+self.d_encounter] += \
                        self.encounter_matrix[self.encounter2idx[field]]
                    n += 1
            if n > 1:
                concat[offset:offset+self.d_encounter] /= n

        result = self.W_proj @ concat
        return self._l2norm(result)

    def get_group_vector(self,
                          words: Optional[List[str]] = None,
                          patient_group: Optional[List[str]] = None,
                          encounter_group: Optional[List[str]] = None) -> np.ndarray:
        """Compose any subset of groups. Absent groups are zeroed.

        Identical to get_combined_vector — exists as a named alias for clarity
        in ablation studies.
        """
        return self.get_combined_vector(words or [], patient_group, encounter_group)

    def get_nearest_neighbors(self,
                               words: List[str],
                               patient_group: Optional[List[str]] = None,
                               encounter_group: Optional[List[str]] = None,
                               k: int = 10) -> List[Tuple[str, float]]:
        """Find k nearest words using projected word-only vectors for candidates."""
        query      = self.get_combined_vector(words, patient_group, encounter_group)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            print("Warning: query vector has near-zero norm.")
            return []

        results = []
        for idx, word in self.idx2word.items():
            pv   = self.get_projected_word_vector(word)
            norm = np.linalg.norm(pv)
            if norm < 1e-8:
                continue
            sim = float(np.dot(query, pv) / (query_norm * norm))
            results.append((word, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def compare_vectors(self,
                         words1: List[str], patient1: Optional[List[str]], encounter1: Optional[List[str]],
                         words2: List[str], patient2: Optional[List[str]], encounter2: Optional[List[str]]) -> dict:
        v1 = self.get_combined_vector(words1, patient1, encounter1)
        v2 = self.get_combined_vector(words2, patient2, encounter2)
        cos = float(np.dot(v1, v2))
        return {
            'cosine_similarity':  cos,
            'cosine_distance':    1.0 - cos,
            'euclidean_distance': float(np.linalg.norm(v1 - v2)),
        }

    def print_projection_block_norms(self) -> None:
        """Diagnostic: Frobenius norm of each block within W_proj.

        The word block should dominate. Near-zero patient/encounter blocks
        indicate those groups are not learning.
        """
        if self.W_proj is None:
            print("No W_proj available.")
            return
        W_word      = self.W_proj[:, :self.d_word]
        W_patient   = self.W_proj[:, self.d_word:self.d_word+self.d_patient]
        W_encounter = self.W_proj[:, self.d_word+self.d_patient:]
        print("W_proj block Frobenius norms:")
        print(f"  W_word      [{self.d_out}x{self.d_word}]:  {np.linalg.norm(W_word):.6f}")
        print(f"  W_patient   [{self.d_out}x{self.d_patient}]:  {np.linalg.norm(W_patient):.6f}")
        print(f"  W_encounter [{self.d_out}x{self.d_encounter}]: {np.linalg.norm(W_encounter):.6f}")

    def print_patient_vocab(self) -> None:
        """Print all patient group fields."""
        print(f"Patient group fields ({len(self.patient2idx)}):")
        for field in sorted(self.patient2idx):
            print(f"  {field}")

    def print_encounter_vocab(self) -> None:
        """Print all encounter group fields."""
        print(f"Encounter group fields ({len(self.encounter2idx)}):")
        for field in sorted(self.encounter2idx):
            print(f"  {field}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fasttext_context.py <model.bin> [word] "
              "[--patient f1 f2 ...] [--encounter f1 f2 ...]")
        sys.exit(1)

    ft = FastTextContext()
    ft.load_model(sys.argv[1])

    words: List[str] = []
    patient_group: List[str] = []
    encounter_group: List[str] = []
    state = 'words'
    for arg in sys.argv[2:]:
        if arg == '--patient':     state = 'patient'
        elif arg == '--encounter': state = 'encounter'
        elif state == 'words':     words.append(arg)
        elif state == 'patient':   patient_group.append(arg)
        elif state == 'encounter': encounter_group.append(arg)

    ft.print_projection_block_norms()

    if words:
        print(f"\nWord vector for '{words[0]}' (d_word={ft.d_word}):")
        vec = ft.get_word_vector(words[0])
        print(f"  Norm:          {np.linalg.norm(vec):.6f}")
        print(f"  First 10 dims: {vec[:10]}")

        print(f"\nNearest neighbors (words={words}, patient={patient_group}, encounter={encounter_group}):")
        for w, s in ft.get_nearest_neighbors(words, patient_group or None,
                                              encounter_group or None, k=5):
            print(f"  {w}: {s:.4f}")

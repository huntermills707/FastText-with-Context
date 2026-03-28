#ifndef FASTTEXT_INFERENCE_H
#define FASTTEXT_INFERENCE_H

#include "types.h"
#include "matrix.h"
#include "vocabulary.h"
#include <string>
#include <vector>
#include <utility>

namespace fasttext {

class Inference {
public:
    Inference(const Vocabulary& vocab,
              const Matrix& input_matrix,
              const Matrix& ngram_matrix,
              const Matrix& metadata_matrix,
              const std::vector<float>& gate_bias,
              const std::vector<float>& alpha,
              int min_n, int max_n);

    // Word vector = word_embedding (if in vocab) + sum(ngram_embeddings).
    std::vector<float> getWordVector(const std::string& word) const;

    // Metadata vector for a list of metadata fields (raw, no gating).
    std::vector<float> getMetadataVector(const std::vector<std::string>& metadata) const;

    // Combined vector with gated metadata composition, L2-normalised.
    // word_part = avg(word_emb_i + sum(ngrams_i))
    // word_avg  = avg(word_emb_i)   [raw word embeddings for gate]
    // gate      = sigmoid(gate_bias + word_avg)
    // meta_part = sum_k(alpha_k * gate * meta_emb_k)
    // combined  = word_part + meta_part, then L2-normalised
    std::vector<float> getCombinedVector(const std::vector<std::string>& words,
                                         const std::vector<std::string>& metadata) const;

    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::vector<std::string>& words,
        const std::vector<std::string>& metadata,
        int k,
        const Matrix& cached_word_vectors) const;

private:
    const Vocabulary&         vocab_;
    const Matrix&             input_matrix_;
    const Matrix&             ngram_matrix_;
    const Matrix&             metadata_matrix_;
    const std::vector<float>& gate_bias_;   // learned gate bias vector (dim)
    const std::vector<float>& alpha_;       // learned per-field scalar weights (meta_size)

    int min_n_, max_n_, dim_;

    uint64_t         hash(const std::string& str) const;
    std::vector<int> getNgramIndices(const std::string& word) const;
};

} // namespace fasttext

#endif

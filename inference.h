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
              int min_n, int max_n);

    // Word vector = word_embedding (if in vocab) + sum(ngram_embeddings).
    std::vector<float> getWordVector(const std::string& word) const;

    std::vector<float> getMetadataVector(const std::vector<std::string>& metadata) const;

    // Combined vector (words + metadata), L2-normalised.
    std::vector<float> getCombinedVector(const std::vector<std::string>& words,
                                         const std::vector<std::string>& metadata) const;

    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::vector<std::string>& words,
        const std::vector<std::string>& metadata,
        int k,
        const Matrix& cached_word_vectors) const;

private:
    const Vocabulary& vocab_;
    const Matrix&     input_matrix_;    // word-level embeddings (vocab_size x dim)
    const Matrix&     ngram_matrix_;
    const Matrix&     metadata_matrix_;

    int min_n_, max_n_, dim_;

    uint64_t         hash(const std::string& str) const;
    std::vector<int> getNgramIndices(const std::string& word) const;
};

} // namespace fasttext

#endif

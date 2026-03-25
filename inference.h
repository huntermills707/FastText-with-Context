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
    Inference(const Vocabulary& vocab, const Matrix& input_matrix,
              const Matrix& ngram_matrix, const Matrix& metadata_matrix,
              int min_n, int max_n);
    
    // Get vector for a single word (includes n-gram contributions)
    std::vector<float> getWordVector(const std::string& word) const;
    
    // Get vector for metadata fields (summed)
    std::vector<float> getMetadataVector(const std::vector<std::string>& metadata) const;
    
    // Get combined vector (words + metadata, normalized)
    std::vector<float> getCombinedVector(const std::vector<std::string>& words,
                                         const std::vector<std::string>& metadata) const;
    
    // Find k nearest neighbors (parallelized)
    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::vector<std::string>& words,
        const std::vector<std::string>& metadata,
        int k) const;

private:
    const Vocabulary& vocab_;
    const Matrix& input_matrix_;
    const Matrix& ngram_matrix_;
    const Matrix& metadata_matrix_;
    
    int min_n_;
    int max_n_;
    int dim_;
    
    uint64_t hash(const std::string& str) const;
    std::vector<int> getNgramIndices(const std::string& word) const;
};

} // namespace fasttext

#endif

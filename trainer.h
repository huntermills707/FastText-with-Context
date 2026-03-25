#ifndef FASTTEXT_TRAINER_H
#define FASTTEXT_TRAINER_H

#include "types.h"
#include "matrix.h"
#include "vocabulary.h"
#include <string>
#include <random>
#include <vector>
#include <omp.h>

namespace fasttext {

class Trainer {
public:
    Trainer(int dim, int epoch, float lr, int min_n, int max_n, 
            int chunk_size, int ngram_buckets, int window_size = 20);
    
    // Main training entry point
    void train(const std::string& filename, Vocabulary& vocab,
               Matrix& input_matrix, Matrix& output_matrix,
               Matrix& ngram_matrix, Matrix& metadata_matrix);
    
private:
    int dim_;
    int epoch_;
    float lr_;
    int min_n_;
    int max_n_;
    int chunk_size_;
    int ngram_buckets_;
    int window_size_;
    
    // Thread-local storage for gradients
    std::vector<Matrix> thread_grads_;  // Gradients for output_matrix (Huffman nodes)
    std::vector<std::vector<float>> thread_input_grads_; // Gradients for input/ngram/metadata
    
    // Thread-local RNGs
    std::vector<std::mt19937> rngs_;
    
    // Hash function for n-grams
    uint64_t hash(const std::string& str) const;
    
    // N-gram utilities
    std::vector<int> getNgramIndices(const std::string& word) const;
    
    // Skip-gram forward/backward pass
    // Processes a center word and its surrounding context window
    void processSample(const StreamingSample& sample, const Vocabulary& vocab,
                       Matrix& input_matrix, Matrix& output_matrix,
                       Matrix& ngram_matrix, Matrix& metadata_matrix,
                       float current_lr, int thread_id);
    
    // Gradient merging
    void mergeGradients(Matrix& output_matrix);
    
    // File parsing
    bool parseLine(const std::string& line, StreamingSample& sample) const;
};

} // namespace fasttext

#endif

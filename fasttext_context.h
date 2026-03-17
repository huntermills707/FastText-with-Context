#ifndef FASTTEXT_CONTEXT_H
#define FASTTEXT_CONTEXT_H

#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <memory>

namespace fasttext {

struct TrainingSample {
    std::vector<std::string> context_fields;  // All context metadata
    std::vector<std::string> words;           // Tokenized sentence
};

class FastTextContext {
public:
    FastTextContext(int dim = 100, int epoch = 5, float lr = 0.05,
                   int min_n = 3, int max_n = 6, int threshold = 10,
                   int context_dim = 50);  // Separate dim for context
    
    void train(const std::string& filename);
    std::vector<float> getWordVector(const std::string& word);
    std::vector<float> getContextVector(const std::string& context_field);
    std::vector<float> getCombinedVector(const std::string& word, 
                                         const std::vector<std::string>& contexts);
    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::string& word, int k = 10);
    
private:
    // Hyperparameters
    int dim_;           // word embedding dimension
    int context_dim_;   // context embedding dimension
    int epoch_;
    float lr_;
    int min_n_;
    int max_n_;
    int threshold_;
    
    // Word embeddings
    std::unordered_map<std::string, int> word2idx_;
    std::vector<std::vector<float>> input_matrix_;
    std::vector<std::vector<float>> output_matrix_;
    std::vector<std::vector<float>> ngram_matrix_;
    
    // Context embeddings (separate space)
    std::unordered_map<std::string, int> context2idx_;
    std::vector<std::vector<float>> context_matrix_;
    
    // Vocabulary info
    std::vector<int> word_counts_;
    
    // Random
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_;
    std::normal_distribution<float> normal_;
    
    // Methods
    uint64_t hash(const std::string& str);
    std::vector<TrainingSample> parseFile(const std::string& filename);
    void buildVocab(const std::vector<TrainingSample>& samples);
    void initializeMatrices();
    void trainModel(const std::vector<TrainingSample>& samples);
    std::vector<int> getNgramIndices(const std::string& word);
    std::vector<float> computeWordVector(const std::string& word);
    std::vector<float> computeContextVector(const std::vector<std::string>& contexts);
    std::vector<float> combineVectors(const std::vector<float>& word_vec,
                                      const std::vector<float>& context_vec);
    void negativeSampling(const std::vector<float>& combined_input,
                         const std::vector<int>& labels,
                         float grad_scale);
};

} // namespace fasttext

#endif

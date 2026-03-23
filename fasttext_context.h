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
#include <queue>
#include <omp.h>
#include <cstdint>
#include <limits>

namespace fasttext {

struct StreamingSample {
    std::vector<std::string> context_fields;
    std::vector<std::string> words;
};

struct HuffmanNode {
    int word_idx;
    double frequency;
    HuffmanNode* left;
    HuffmanNode* right;
    
    HuffmanNode(int idx, double freq) 
        : word_idx(idx), frequency(freq), left(nullptr), right(nullptr) {}
};

class FastTextContext {
public:
    FastTextContext(int dim = 100, int epoch = 5, float lr = 0.05,
                   int min_n = 3, int max_n = 6, int threshold = 10,
                   int merge_interval = 100000, int chunk_size = 10000);
    
    ~FastTextContext();
    
    void trainStreaming(const std::string& filename);
    
    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);
    
    std::vector<float> getWordVector(const std::string& word);
    std::vector<float> getContextVector(const std::string& context_field);
    std::vector<float> getCombinedVector(const std::vector<std::string>& words, 
                                         const std::vector<std::string>& contexts);
    
    // Optimized nearest neighbors using direct index access and vectorized-like loops
    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::vector<std::string>& words, 
        const std::vector<std::string>& contexts, 
        int k = 10);
    
    int getDim() const { return dim_; }
    int getMinN() const { return min_n_; }
    int getMaxN() const { return max_n_; }
    int getThreshold() const { return threshold_; }
    
private:
    int dim_;
    int epoch_;
    float lr_;
    int min_n_;
    int max_n_;
    int threshold_;
    int merge_interval_;
    int chunk_size_;
    
    std::unordered_map<std::string, int> word2idx_;
    std::vector<std::vector<float>> input_matrix_;
    std::vector<std::vector<float>> output_matrix_;
    std::vector<std::vector<float>> ngram_matrix_;
    
    std::unordered_map<std::string, int> context2idx_;
    std::vector<std::vector<float>> context_matrix_;
    
    HuffmanNode* huffman_root_;
    std::vector<std::vector<int>> word_codes_;
    std::vector<std::vector<int>> word_paths_;
    std::vector<double> word_freqs_;
    
    std::vector<int> word_counts_;
    
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_;
    std::normal_distribution<float> normal_;
    
    std::vector<std::vector<std::vector<float>>> thread_local_grads_;
    int num_threads_;
    
    uint64_t hash(const std::string& str);
    void countVocabulary(const std::string& filename, 
                        std::unordered_map<std::string, int>& word_freq,
                        std::unordered_map<std::string, int>& context_freq);
    void buildVocabFromCounts(const std::unordered_map<std::string, int>& word_freq,
                             const std::unordered_map<std::string, int>& context_freq);
    void buildHuffmanTree();
    void generateCodes(HuffmanNode* node, std::vector<int>& code, 
                       std::vector<int>& path, std::vector<int>& path_nodes);
    void initializeMatrices();
    void trainModelStreaming(const std::string& filename);
    std::vector<int> getNgramIndices(const std::string& word);
    std::vector<float> computeWordVector(const std::string& word);
    std::vector<float> computeContextVector(const std::vector<std::string>& contexts);
    std::vector<float> combineVectorsAdditive(const std::vector<float>& word_vec,
                                              const std::vector<float>& context_vec);
    void hierarchicalSoftmax(const std::vector<float>& combined_input,
                            int target_word_idx,
                            float grad_scale);
    void cleanupHuffmanTree(HuffmanNode* node);
    void mergeThreadLocalGradients();
    bool parseNextSample(std::ifstream& file, StreamingSample& sample);
    
    void diagnoseVocabulary();
    void diagnoseHuffmanTree();
};

} // namespace fasttext

#endif

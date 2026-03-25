#ifndef FASTTEXT_CONTEXT_H
#define FASTTEXT_CONTEXT_H

#include "types.h"
#include "matrix.h"
#include "vocabulary.h"
#include "trainer.h"
#include "inference.h"
#include <string>
#include <memory>

namespace fasttext {

class FastTextContext {
public:
    FastTextContext(int dim = 100, int epoch = 5, float lr = 0.05,
                   int min_n = 3, int max_n = 6, int threshold = 5,
                   int chunk_size = 100000, int ngram_buckets = 2000000,
                   int window_size = 20);
    
    ~FastTextContext() = default;
    
    // Training
    void trainStreaming(const std::string& filename);
    
    // Persistence
    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);
    
    // Inference (delegated to Inference class)
    std::vector<float> getWordVector(const std::string& word);
    std::vector<float> getMetadataVector(const std::string& metadata_field);
    std::vector<float> getCombinedVector(const std::vector<std::string>& words, 
                                         const std::vector<std::string>& metadata);
    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::vector<std::string>& words, 
        const std::vector<std::string>& metadata,
        int k = 10);
    
    // Accessors
    int getDim() const { return dim_; }
    int getMinN() const { return min_n_; }
    int getMaxN() const { return max_n_; }
    int getThreshold() const { return threshold_; }
    int getNgramBuckets() const { return ngram_buckets_; }
    int getWindowSize() const { return window_size_; }

private:
    int dim_;
    int epoch_;
    float lr_;
    int min_n_;
    int max_n_;
    int threshold_;
    int chunk_size_;
    int ngram_buckets_;
    int window_size_;

    Vocabulary vocab_;
    Matrix input_matrix_;   // Word embeddings
    Matrix output_matrix_;  // Hierarchical softmax nodes
    Matrix ngram_matrix_;   // Subword embeddings
    Matrix metadata_matrix_; // Metadata embeddings
    
    std::unique_ptr<Inference> inference_;
    
    void initializeMatrices();
};

} // namespace fasttext

#endif

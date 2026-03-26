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
    FastTextContext(int dim = 100, int epoch = 5, float lr = 0.05f,
                   int min_n = 3, int max_n = 6, int threshold = 5,
                   int chunk_size = 100000, int ngram_buckets = 2000000,
                   int window_size = 20, float subsample_t = 1e-4f,
                   float grad_clip = 1.0f);

    ~FastTextContext() = default;

    void trainStreaming(const std::string& filename);
    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

    std::vector<float> getWordVector(const std::string& word);
    std::vector<float> getMetadataVector(const std::string& metadata_field);
    std::vector<float> getCombinedVector(const std::vector<std::string>& words,
                                         const std::vector<std::string>& metadata);
    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::vector<std::string>& words,
        const std::vector<std::string>& metadata,
        int k = 10);

    int getDim()          const { return dim_; }
    int getMinN()         const { return min_n_; }
    int getMaxN()         const { return max_n_; }
    int getThreshold()    const { return threshold_; }
    int getNgramBuckets() const { return ngram_buckets_; }
    int getWindowSize()   const { return window_size_; }

private:
    int   dim_, epoch_, min_n_, max_n_, threshold_, chunk_size_, ngram_buckets_, window_size_;
    float lr_, subsample_t_, grad_clip_;

    Vocabulary vocab_;
    Matrix     output_matrix_;         // hierarchical softmax nodes  (V-1 x dim)
    Matrix     input_matrix_;          // word-level input embeddings  (vocab_size x dim)
    Matrix     ngram_matrix_;          // subword n-gram embeddings    (ngram_buckets x dim)
    Matrix     metadata_matrix_;       // metadata field embeddings    (meta_size x dim)
    Matrix     cached_word_vectors_;   // precomputed for fast NN search

    std::unique_ptr<Inference> inference_;

    void initializeMatrices();
    void precomputeWordVectors();
    void makeInference();
};

} // namespace fasttext

#endif

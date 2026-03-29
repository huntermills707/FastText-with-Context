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
    // d_w:   word + n-gram embedding dimension
    // d_p:   patient group embedding dimension
    // d_pr:  provider group embedding dimension
    // d_out: output (projected) dimension — the space HS and NN search operate in
    FastTextContext(int d_w = 150, int d_p = 30, int d_pr = 15, int d_out = 150,
                   int epoch = 5, float lr = 0.05f,
                   int min_n = 3, int max_n = 8, int threshold = 5,
                   int chunk_size = 1000, int ngram_buckets = 2000000,
                   int window_size = 5, float subsample_t = 1e-4f,
                   float grad_clip = 1.0f, float weight_decay = 0.0f);

    ~FastTextContext() = default;

    void trainStreaming(const std::string& filename);
    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

    // Returns d_w-dimensional word vector (word_emb + ngrams). Not projected.
    std::vector<float> getWordVector(const std::string& word);

    // Returns d_out-dimensional combined vector (all groups concatenated, projected,
    // L2-normalised). Any absent groups produce zero contribution.
    std::vector<float> getCombinedVector(const std::vector<std::string>& words,
                                          const std::vector<std::string>& patient_meta,
                                          const std::vector<std::string>& provider_meta);

    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::vector<std::string>& words,
        const std::vector<std::string>& patient_meta,
        const std::vector<std::string>& provider_meta,
        int k = 10);

    int getDW()  const { return d_w_; }
    int getDP()  const { return d_p_; }
    int getDPr() const { return d_pr_; }
    int getDOut() const { return d_out_; }
    int getMinN()         const { return min_n_; }
    int getMaxN()         const { return max_n_; }
    int getThreshold()    const { return threshold_; }
    int getNgramBuckets() const { return ngram_buckets_; }
    int getWindowSize()   const { return window_size_; }
    int getConcatDim()    const { return d_w_ + d_p_ + d_pr_; }

private:
    int   d_w_, d_p_, d_pr_, d_out_;
    int   epoch_, min_n_, max_n_, threshold_, chunk_size_, ngram_buckets_, window_size_;
    float lr_, subsample_t_, grad_clip_, weight_decay_;

    Vocabulary vocab_;

    // Sparse matrices — Hogwild-updated during training.
    Matrix input_matrix_;    // vocab_size x d_w
    Matrix output_matrix_;   // hs_nodes x d_out
    Matrix ngram_matrix_;    // ngram_buckets x d_w

    // Dense matrices — synchronized-averaged during training.
    Matrix W_proj_;          // d_out x concat_dim
    Matrix patient_matrix_;  // patient_size x d_p
    Matrix provider_matrix_; // provider_size x d_pr

    // NN search cache: vocab_size x d_out (projected word-only vectors).
    Matrix cached_word_vectors_;

    std::unique_ptr<Inference> inference_;

    void initializeMatrices();
    void precomputeWordVectors();
    void makeInference();
};

} // namespace fasttext

#endif

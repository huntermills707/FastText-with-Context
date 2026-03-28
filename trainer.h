#ifndef FASTTEXT_TRAINER_H
#define FASTTEXT_TRAINER_H

#include "types.h"
#include "matrix.h"
#include "vocabulary.h"
#include <string>
#include <random>
#include <vector>

namespace fasttext {

class Trainer {
public:
    // grad_clip: maximum L2 norm for gradient vectors before they are applied.
    // Applied in two places: the per-node output update and the accumulated
    // center_grad before it is distributed to input/ngram/metadata matrices.
    // Set to 0.0 to disable clipping entirely.
    Trainer(int dim, int epoch, float lr, int min_n, int max_n,
            int chunk_size, int ngram_buckets, int window_size = 5,
            float grad_clip = 1.0f);

    // input_matrix: word-level embeddings (vocab_size x dim), new vs. old API.
    void train(const std::string& filename, Vocabulary& vocab,
               Matrix& input_matrix,
               Matrix& output_matrix,
               Matrix& ngram_matrix,
               Matrix& metadata_matrix);

private:
    int   dim_, epoch_, min_n_, max_n_, chunk_size_, ngram_buckets_, window_size_;
    float lr_, grad_clip_;

    // Per-thread RNGs for subsampling and window-size sampling.
    std::vector<std::mt19937> rngs_;

    // Per-thread scratch buffers to avoid repeated heap allocation inside the
    // innermost training loop.  Sized once in the constructor and reused for
    // every call to processSample / buildCenterVec / hsStep.
    struct ThreadBuffers {
        std::vector<float> center_vec;
        std::vector<float> center_grad;
        std::vector<float> out_update;
        std::vector<int>   ngram_indices;

        void resize(int dim) {
            center_vec.resize(dim);
            center_grad.resize(dim);
            out_update.resize(dim);
            // ngram_indices is variable-length; reserve a reasonable amount.
            ngram_indices.reserve(64);
        }
    };
    std::vector<ThreadBuffers> thread_bufs_;

    uint64_t         hash(const std::string& str) const;
    void             getNgramIndices(const std::string& word, std::vector<int>& out) const;
    bool             parseLine(const std::string& line, StreamingSample& sample) const;
    int              countLines(const std::string& filename) const;
    bool             checkMatrixHealth(const Matrix& m, const std::string& name, int ep) const;

    // Scales v in-place so its L2 norm does not exceed max_norm.
    // No-op if max_norm <= 0 or the norm is already within bounds.
    static void clipNorm(std::vector<float>& v, float max_norm);

    // Returns (metadata_vec, active_meta_indices).
    std::pair<std::vector<float>, std::vector<int>>
    gatherMetadataVec(const StreamingSample& sample, const Vocabulary& vocab,
                      const Matrix& metadata_matrix) const;

    // Fills out_vec and out_ngram with the composite center vector and its
    // n-gram indices.  Reuses the caller-provided vectors instead of
    // allocating new ones.
    // center_vec = word_embedding + sum(ngrams) + metadata_vec.
    void buildCenterVec(int word_idx, const std::string& word,
                        const Matrix& input_matrix, const Matrix& ngram_matrix,
                        const std::vector<float>& meta_vec,
                        std::vector<float>& out_vec,
                        std::vector<int>& out_ngram) const;

    // Single Huffman-path node forward+backward step.
    // Updates output_matrix in-place (Hogwild), accumulates into center_grad.
    // Uses scratch as a pre-allocated buffer for the output update vector.
    // Returns binary cross-entropy loss for this node.
    // Ordering matches word2vec: input gradient uses pre-update output row.
    float hsStep(int node_idx, int direction,
                 const std::vector<float>& center_vec,
                 Matrix& output_matrix,
                 std::vector<float>& center_grad,
                 std::vector<float>& scratch,
                 float lr);

    // Distributes center_grad to input_matrix, ngram_matrix, metadata_matrix (Hogwild).
    void distributeGrad(const std::vector<float>& center_grad,
                        int word_idx, const std::vector<int>& ngram_indices,
                        const std::vector<int>& active_meta,
                        Matrix& input_matrix, Matrix& ngram_matrix,
                        Matrix& metadata_matrix);

    // Parallel chunk processing (Hogwild: no gradient merge step).
    void processChunk(const std::vector<StreamingSample>& chunk,
                      const Vocabulary& vocab,
                      Matrix& input_matrix, Matrix& output_matrix,
                      Matrix& ngram_matrix, Matrix& metadata_matrix,
                      float lr, double& loss_acc, int& pred_count);

    void processSample(const StreamingSample& sample, const Vocabulary& vocab,
                       Matrix& input_matrix, Matrix& output_matrix,
                       Matrix& ngram_matrix, Matrix& metadata_matrix,
                       float lr, int thread_id,
                       double& loss_acc, int& pred_count);

    void runEpoch(int ep, int total_samples, long long& global_step,
                  long long total_steps, const std::string& filename,
                  Vocabulary& vocab,
                  Matrix& input_matrix, Matrix& output_matrix,
                  Matrix& ngram_matrix, Matrix& metadata_matrix);
};

} // namespace fasttext

#endif

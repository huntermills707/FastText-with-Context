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

    // gate_bias (dim-length) and alpha (metadata_size-length) are passed by
    // reference and updated in-place during training (Hogwild).
    void train(const std::string& filename, Vocabulary& vocab,
               Matrix& input_matrix,
               Matrix& output_matrix,
               Matrix& ngram_matrix,
               Matrix& metadata_matrix,
               std::vector<float>& gate_bias,
               std::vector<float>& alpha);

private:
    int   dim_, epoch_, min_n_, max_n_, chunk_size_, ngram_buckets_, window_size_;
    float lr_, grad_clip_;

    std::vector<std::mt19937> rngs_;

    struct ThreadBuffers {
        std::vector<float> center_vec;
        std::vector<float> center_grad;
        std::vector<float> out_update;
        std::vector<int>   ngram_indices;
        // New buffers for gated composition.
        std::vector<float> word_part;      // word_emb + sum(ngrams), before meta added
        std::vector<float> gate;           // sigmoid(gate_bias + word_part)
        std::vector<float> meta_vec;       // gated metadata sum for this center word
        std::vector<float> gate_bias_grad; // gradient w.r.t. gate_bias for this sample

        void resize(int dim) {
            center_vec.resize(dim);
            center_grad.resize(dim);
            out_update.resize(dim);
            word_part.resize(dim);
            gate.resize(dim);
            meta_vec.resize(dim);
            gate_bias_grad.resize(dim);
            ngram_indices.reserve(64);
        }
    };
    std::vector<ThreadBuffers> thread_bufs_;

    uint64_t         hash(const std::string& str) const;
    void             getNgramIndices(const std::string& word, std::vector<int>& out) const;
    bool             parseLine(const std::string& line, StreamingSample& sample) const;
    int              countLines(const std::string& filename) const;
    bool             checkMatrixHealth(const Matrix& m, const std::string& name, int ep) const;

    static void clipNorm(std::vector<float>& v, float max_norm);

    // Compute gate = sigmoid(gate_bias + word_part) element-wise.
    // word_part = word_emb + sum(ngrams), so OOV words get a morphologically-
    // informed gate signal rather than collapsing to sigmoid(gate_bias).
    void computeGate(const float* word_part,
                     const std::vector<float>& gate_bias,
                     std::vector<float>& gate) const;

    // Compute gated metadata vector: sum_k(alpha_k * gate * meta_emb_k).
    // Returns active metadata indices.
    std::vector<int> gatherGatedMetaVec(const StreamingSample& sample,
                                        const Vocabulary& vocab,
                                        const Matrix& metadata_matrix,
                                        const std::vector<float>& alpha,
                                        const std::vector<float>& gate,
                                        std::vector<float>& out_meta_vec) const;

    // Fills out_vec and out_ngram with center vector using pre-computed meta_vec.
    // center_vec = word_emb + sum(ngrams) + gated_meta_vec
    void buildCenterVec(int word_idx, const std::string& word,
                        const Matrix& input_matrix, const Matrix& ngram_matrix,
                        const std::vector<float>& meta_vec,
                        std::vector<float>& out_vec,
                        std::vector<int>& out_ngram) const;

    float hsStep(int node_idx, int direction,
                 const std::vector<float>& center_vec,
                 Matrix& output_matrix,
                 std::vector<float>& center_grad,
                 std::vector<float>& scratch,
                 float lr);

    // Distributes gradients to all parameter matrices including gate_bias and alpha.
    void distributeGrad(const std::vector<float>& center_grad,
                        int word_idx,
                        const std::vector<int>& ngram_indices,
                        const std::vector<int>& active_meta,
                        const std::vector<float>& gate,
                        const std::vector<float>& alpha,
                        const Matrix& metadata_matrix,
                        Matrix& input_matrix,
                        Matrix& ngram_matrix,
                        Matrix& metadata_matrix_mutable,
                        std::vector<float>& gate_bias,
                        std::vector<float>& alpha_mutable,
                        std::vector<float>& gate_bias_grad_buf,
                        float lr);

    void processChunk(const std::vector<StreamingSample>& chunk,
                      const Vocabulary& vocab,
                      Matrix& input_matrix, Matrix& output_matrix,
                      Matrix& ngram_matrix, Matrix& metadata_matrix,
                      std::vector<float>& gate_bias, std::vector<float>& alpha,
                      float lr, double& loss_acc, int& pred_count);

    void processSample(const StreamingSample& sample, const Vocabulary& vocab,
                       Matrix& input_matrix, Matrix& output_matrix,
                       Matrix& ngram_matrix, Matrix& metadata_matrix,
                       std::vector<float>& gate_bias, std::vector<float>& alpha,
                       float lr, int thread_id,
                       double& loss_acc, int& pred_count);

    void runEpoch(int ep, int total_samples, long long& global_step,
                  long long total_steps, const std::string& filename,
                  Vocabulary& vocab,
                  Matrix& input_matrix, Matrix& output_matrix,
                  Matrix& ngram_matrix, Matrix& metadata_matrix,
                  std::vector<float>& gate_bias, std::vector<float>& alpha);
};

} // namespace fasttext

#endif

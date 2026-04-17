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
    // d_word:  word + n-gram embedding dimension
    // d_patient:  patient group embedding dimension
    // d_encounter: encounter group embedding dimension
    // d_out: output (projected) embedding dimension — the space HS operates in
    // weight_decay: L2 regularization coefficient applied to W_proj after each
    // chunk reduce step (e.g. 1e-5). Set to 0.0 to disable.
    // Not applied to sparse embedding matrices — only to the projection.
    Trainer(int d_word, int d_patient, int d_encounter, int d_out,
            int epoch, float lr, int min_n, int max_n,
            int chunk_size, int ngram_buckets, int window_size,
            float grad_clip, float weight_decay = 0.0f);

    // Sparse shared matrices (Hogwild): input_matrix, output_matrix, ngram_matrix.
    // Dense shared matrices (synchronized averaging): W_proj, patient_matrix, encounter_matrix.
    void train(const std::string& filename, Vocabulary& vocab,
               Matrix& input_matrix,    // vocab_size x d_word  — Hogwild
               Matrix& output_matrix,   // hs_nodes x d_out  — Hogwild
               Matrix& ngram_matrix,    // ngram_buckets x d_word — Hogwild
               Matrix& W_proj,          // d_out x concat_dim — sync-averaged
               Matrix& patient_matrix,  // patient_size x d_patient — sync-averaged
               Matrix& encounter_matrix); // encounter_size x d_encounter — sync-averaged

private:
    int   d_word_, d_patient_, d_encounter_, d_out_, concat_dim_;
    int   epoch_, min_n_, max_n_, chunk_size_, ngram_buckets_, window_size_;
    float lr_, grad_clip_, weight_decay_;

    std::vector<std::mt19937> rngs_;

    struct ThreadBuffers {
        std::vector<float> concat_vec;   // concat_dim: [word_part ; patient_part ; encounter_part]
        std::vector<float> center_vec;   // d_out: projected center vector
        std::vector<float> center_grad;  // d_out: accumulated HS gradient
        std::vector<float> concat_grad;  // concat_dim: backpropagated through projection
        std::vector<float> out_update;   // d_out: scratch for HS node update clipping
        std::vector<int>   ngram_indices;

        void resize(int d_word, int d_patient, int d_encounter, int d_out) {
            int concat_dim = d_word + d_patient + d_encounter;
            concat_vec.resize(concat_dim);
            center_vec.resize(d_out);
            center_grad.resize(d_out);
            concat_grad.resize(concat_dim);
            out_update.resize(d_out);
            ngram_indices.reserve(64);
        }
    };
    std::vector<ThreadBuffers> thread_bufs_;

    // Thread-local dense parameter copies (one per thread).
    std::vector<Matrix> W_proj_local_;
    std::vector<Matrix> patient_local_;
    std::vector<Matrix> encounter_local_;

    uint64_t hash(const std::string& str) const;
    void     getNgramIndices(const std::string& word, std::vector<int>& out) const;
    bool     parseGroupedLine(const std::string& line, GroupedSample& sample) const;
    int      countLines(const std::string& filename) const;
    bool     checkMatrixHealth(const Matrix& m, const std::string& name, int ep) const;

    static void clipNorm(std::vector<float>& v, float max_norm);

    // Copy shared dense params → all thread-local copies (broadcast).
    void broadcastDenseParams(const Matrix& W_proj,
                              const Matrix& patient_matrix,
                              const Matrix& encounter_matrix);

    // Average all thread-local copies → shared dense params (reduce).
    void reduceDenseParams(Matrix& W_proj,
                           Matrix& patient_matrix,
                           Matrix& encounter_matrix);

    // Build concat_vec = [word_part ; patient_avg ; encounter_avg].
    void buildConcatVec(int word_idx, const std::string& word,
                        const GroupedSample& sample,
                        const Vocabulary& vocab,
                        const Matrix& input_matrix,
                        const Matrix& ngram_matrix,
                        const Matrix& patient_matrix,  // thread-local
                        const Matrix& encounter_matrix, // thread-local
                        std::vector<float>& concat_out,
                        std::vector<int>& ngram_out) const;

    // Distribute concat_grad to all parameter matrices.
    void distributeGrad(const std::vector<float>& concat_grad,
                        const std::vector<float>& center_grad,
                        const std::vector<float>& concat_vec,
                        int word_idx,
                        const std::vector<int>& ngram_indices,
                        const GroupedSample& sample,
                        const Vocabulary& vocab,
                        Matrix& input_matrix,    // shared Hogwild
                        Matrix& ngram_matrix,    // shared Hogwild
                        Matrix& W_proj,          // thread-local
                        Matrix& patient_matrix,  // thread-local
                        Matrix& encounter_matrix, // thread-local
                        float lr);

    // Single hierarchical-softmax step. Operates in d_out space.
    // Updates output_matrix (shared Hogwild).
    float hsStep(int node_idx, int direction,
                 const std::vector<float>& center_vec,
                 Matrix& output_matrix,
                 std::vector<float>& center_grad,
                 std::vector<float>& scratch,
                 float lr);

    void processSample(const GroupedSample& sample,
                       const Vocabulary& vocab,
                       Matrix& input_matrix, Matrix& output_matrix,
                       Matrix& ngram_matrix,
                       Matrix& W_proj,          // thread-local
                       Matrix& patient_matrix,  // thread-local
                       Matrix& encounter_matrix, // thread-local
                       float lr, int tid,
                       double& loss_acc, long long& pred_count);

    void processChunk(const std::vector<GroupedSample>& chunk,
                      const Vocabulary& vocab,
                      Matrix& input_matrix, Matrix& output_matrix,
                      Matrix& ngram_matrix,
                      Matrix& W_proj,
                      Matrix& patient_matrix,
                      Matrix& encounter_matrix,
                      float lr, double& loss_acc, long long& pred_count);

    void runEpoch(int ep, int total_samples, long long& global_step,
                  long long total_steps, const std::string& filename,
                  Vocabulary& vocab,
                  Matrix& input_matrix, Matrix& output_matrix,
                  Matrix& ngram_matrix,
                  Matrix& W_proj, Matrix& patient_matrix, Matrix& encounter_matrix);
};

} // namespace fasttext

#endif

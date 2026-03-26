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
    Trainer(int dim, int epoch, float lr, int min_n, int max_n,
            int chunk_size, int ngram_buckets, int window_size = 20);

    // input_matrix: word-level embeddings (vocab_size x dim), new vs. old API.
    void train(const std::string& filename, Vocabulary& vocab,
               Matrix& input_matrix,
               Matrix& output_matrix,
               Matrix& ngram_matrix,
               Matrix& metadata_matrix);

private:
    int   dim_, epoch_, min_n_, max_n_, chunk_size_, ngram_buckets_, window_size_;
    float lr_;

    // Per-thread RNGs for subsampling and window-size sampling.
    std::vector<std::mt19937> rngs_;

    uint64_t         hash(const std::string& str) const;
    std::vector<int> getNgramIndices(const std::string& word) const;
    bool             parseLine(const std::string& line, StreamingSample& sample) const;
    int              countLines(const std::string& filename) const;
    bool             checkMatrixHealth(const Matrix& m, const std::string& name, int ep) const;

    // Returns (metadata_vec, active_meta_indices).
    std::pair<std::vector<float>, std::vector<int>>
    gatherMetadataVec(const StreamingSample& sample, const Vocabulary& vocab,
                      const Matrix& metadata_matrix) const;

    // Returns (center_vec, ngram_indices).
    // center_vec = word_embedding + sum(ngrams) + metadata_vec.
    std::pair<std::vector<float>, std::vector<int>>
    buildCenterVec(int word_idx, const std::string& word,
                   const Matrix& input_matrix, const Matrix& ngram_matrix,
                   const std::vector<float>& meta_vec) const;

    // Single Huffman-path node forward+backward step.
    // Updates output_matrix in-place (Hogwild), accumulates into center_grad.
    // Returns binary cross-entropy loss for this node.
    // Ordering matches word2vec: input gradient uses pre-update output row.
    float hsStep(int node_idx, int direction,
                 const std::vector<float>& center_vec,
                 Matrix& output_matrix,
                 std::vector<float>& center_grad,
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

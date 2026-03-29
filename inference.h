#ifndef FASTTEXT_INFERENCE_H
#define FASTTEXT_INFERENCE_H

#include "types.h"
#include "matrix.h"
#include "vocabulary.h"
#include <string>
#include <vector>
#include <utility>

namespace fasttext {

class Inference {
public:
    Inference(const Vocabulary& vocab,
              const Matrix& input_matrix,    // vocab_size x d_w
              const Matrix& ngram_matrix,    // ngram_buckets x d_w
              const Matrix& W_proj,          // d_out x concat_dim
              const Matrix& patient_matrix,  // patient_size x d_p
              const Matrix& provider_matrix, // provider_size x d_pr
              int d_w, int d_p, int d_pr, int d_out,
              int min_n, int max_n);

    // Raw word vector = word_emb (if in vocab) + sum(ngram_embs). Dimension: d_w.
    std::vector<float> getWordVector(const std::string& word) const;

    // Project a word-only vector through W_proj (patient/provider regions zeroed).
    // Returns a d_out-dimensional vector. Not L2-normalised.
    std::vector<float> getProjectedWordVector(const std::string& word) const;

    // Gated composition: avg(word_vecs) + patient_avg + provider_avg → W_proj → L2-norm.
    // Patient and provider regions are averaged independently.
    // Returns a d_out-dimensional L2-normalised vector.
    std::vector<float> getCombinedVector(const std::vector<std::string>& words,
                                          const std::vector<std::string>& patient_meta,
                                          const std::vector<std::string>& provider_meta) const;

    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::vector<std::string>& words,
        const std::vector<std::string>& patient_meta,
        const std::vector<std::string>& provider_meta,
        int k,
        const Matrix& cached_word_vectors) const;  // vocab_size x d_out (projected)

    int getDW()  const { return d_w_; }
    int getDOut() const { return d_out_; }

private:
    const Vocabulary&  vocab_;
    const Matrix&      input_matrix_;
    const Matrix&      ngram_matrix_;
    const Matrix&      W_proj_;
    const Matrix&      patient_matrix_;
    const Matrix&      provider_matrix_;

    int d_w_, d_p_, d_pr_, d_out_, concat_dim_;
    int min_n_, max_n_;

    uint64_t         hash(const std::string& str) const;
    std::vector<int> getNgramIndices(const std::string& word) const;
};

} // namespace fasttext

#endif

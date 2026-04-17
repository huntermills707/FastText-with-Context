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
              const Matrix& input_matrix,    // vocab_size x d_word
              const Matrix& ngram_matrix,    // ngram_buckets x d_word
              const Matrix& W_proj,          // d_out x concat_dim
              const Matrix& patient_matrix,  // patient_size x d_patient
              const Matrix& encounter_matrix, // encounter_size x d_encounter
              int d_word, int d_patient, int d_encounter, int d_out,
              int min_n, int max_n);

    // Raw word vector = word_emb (if in vocab) + sum(ngram_embs). Dimension: d_word.
    std::vector<float> getWordVector(const std::string& word) const;

    // Project a word-only vector through W_proj (patient/encounter regions zeroed).
    // Returns a d_out-dimensional vector. Not L2-normalised.
    std::vector<float> getProjectedWordVector(const std::string& word) const;

    // Gated composition: avg(word_vecs) + patient_avg + encounter_avg → W_proj → L2-norm.
    // Patient and encounter regions are averaged independently.
    // Returns a d_out-dimensional L2-normalised vector.
    std::vector<float> getCombinedVector(const std::vector<std::string>& words,
                                          const std::vector<std::string>& patient_group,
                                          const std::vector<std::string>& encounter_group) const;

    std::vector<std::pair<std::string, float>> getNearestNeighbors(
        const std::vector<std::string>& words,
        const std::vector<std::string>& patient_group,
        const std::vector<std::string>& encounter_group,
        int k,
        const Matrix& cached_word_vectors) const;  // vocab_size x d_out (projected)

    int getDWord() const { return d_word_; }
    int getDOut()  const { return d_out_; }

private:
    const Vocabulary&  vocab_;
    const Matrix&      input_matrix_;
    const Matrix&      ngram_matrix_;
    const Matrix&      W_proj_;
    const Matrix&      patient_matrix_;
    const Matrix&      encounter_matrix_;

    int d_word_, d_patient_, d_encounter_, d_out_, concat_dim_;
    int min_n_, max_n_;

    uint64_t         hash(const std::string& str) const;
    std::vector<int> getNgramIndices(const std::string& word) const;
};

} // namespace fasttext

#endif

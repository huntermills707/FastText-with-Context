#include "inference.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <queue>
#include <cstring>

namespace fasttext {

Inference::Inference(const Vocabulary& vocab,
                     const Matrix& input_matrix,
                     const Matrix& ngram_matrix,
                     const Matrix& W_proj,
                     const Matrix& patient_matrix,
                     const Matrix& provider_matrix,
                     int d_w, int d_p, int d_pr, int d_out,
                     int min_n, int max_n)
    : vocab_(vocab),
      input_matrix_(input_matrix),
      ngram_matrix_(ngram_matrix),
      W_proj_(W_proj),
      patient_matrix_(patient_matrix),
      provider_matrix_(provider_matrix),
      d_w_(d_w), d_p_(d_p), d_pr_(d_pr), d_out_(d_out),
      concat_dim_(d_w + d_p + d_pr),
      min_n_(min_n), max_n_(max_n) {}

uint64_t Inference::hash(const std::string& str) const {
    uint64_t h = 14695981039346656037ULL;
    for (char c : str) { h ^= static_cast<uint64_t>(c); h *= 1099511628211ULL; }
    return h;
}

std::vector<int> Inference::getNgramIndices(const std::string& word) const {
    std::vector<int> indices;
    std::string bordered = "<" + word + ">";
    for (int n = min_n_; n <= max_n_; ++n) {
        for (size_t i = 0; i + n <= bordered.size(); ++i) {
            indices.push_back(static_cast<int>(
                hash(bordered.substr(i, n)) % ngram_matrix_.rows()));
        }
    }
    return indices;
}

// word_embedding (if in vocab) + sum(ngram_embeddings). Dimension: d_w.
// OOV words fall back to ngrams only.
std::vector<float> Inference::getWordVector(const std::string& word) const {
    std::vector<float> vec(d_w_, 0.0f);

    int word_idx = vocab_.getWordIdx(word);
    if (word_idx >= 0) {
        const float* wr = input_matrix_.row(word_idx);
        for (int j = 0; j < d_w_; ++j) vec[j] += wr[j];
    }

    for (int idx : getNgramIndices(word)) {
        const float* nr = ngram_matrix_.row(idx);
        for (int j = 0; j < d_w_; ++j) vec[j] += nr[j];
    }

    return vec;
}

// Project a word-only vector (patient/provider regions zeroed) through W_proj.
// Returns d_out-dimensional vector, not L2-normalised.
std::vector<float> Inference::getProjectedWordVector(const std::string& word) const {
    std::vector<float> concat(concat_dim_, 0.0f);
    std::vector<float> wv = getWordVector(word);
    std::copy(wv.begin(), wv.end(), concat.begin());
    // Patient and provider regions remain zero.

    std::vector<float> projected(d_out_);
    W_proj_.mulVec(concat.data(), projected.data());
    return projected;
}

// Concatenation-with-projection composition:
//   word_part = avg(word_emb_i + sum(ngrams_i))   [d_w]
//   patient_part = avg(patient_emb_i)              [d_p]  (zero if empty)
//   provider_part = avg(provider_emb_i)            [d_pr] (zero if empty)
//   concat = [word_part ; patient_part ; provider_part]  [concat_dim]
//   result = L2_normalise(W_proj * concat)         [d_out]
std::vector<float> Inference::getCombinedVector(
    const std::vector<std::string>& words,
    const std::vector<std::string>& patient_meta,
    const std::vector<std::string>& provider_meta) const {

    std::vector<float> concat(concat_dim_, 0.0f);

    // Word part: avg(word_emb_i + sum(ngrams_i)).
    float* wp = concat.data();
    for (const auto& w : words) {
        auto wv = getWordVector(w);
        for (int j = 0; j < d_w_; ++j) wp[j] += wv[j];
    }
    if (!words.empty()) {
        float inv = 1.0f / static_cast<float>(words.size());
        for (int j = 0; j < d_w_; ++j) wp[j] *= inv;
    }

    // Patient part: avg of active patient embeddings.
    float* pp = concat.data() + d_w_;
    int np = 0;
    for (const auto& field : patient_meta) {
        int idx = vocab_.getPatientIdx(field);
        if (idx < 0) continue;
        const float* row = patient_matrix_.row(idx);
        for (int j = 0; j < d_p_; ++j) pp[j] += row[j];
        ++np;
    }
    if (np > 1) {
        float inv = 1.0f / static_cast<float>(np);
        for (int j = 0; j < d_p_; ++j) pp[j] *= inv;
    }

    // Provider part: avg of active provider embeddings.
    float* prp = concat.data() + d_w_ + d_p_;
    int npr = 0;
    for (const auto& field : provider_meta) {
        int idx = vocab_.getProviderIdx(field);
        if (idx < 0) continue;
        const float* row = provider_matrix_.row(idx);
        for (int j = 0; j < d_pr_; ++j) prp[j] += row[j];
        ++npr;
    }
    if (npr > 1) {
        float inv = 1.0f / static_cast<float>(npr);
        for (int j = 0; j < d_pr_; ++j) prp[j] *= inv;
    }

    // Project: d_out = W_proj * concat.
    std::vector<float> result(d_out_);
    W_proj_.mulVec(concat.data(), result.data());

    // L2 normalise.
    float norm = 0.0f;
    for (float v : result) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > MIN_NORM)
        for (float& v : result) v /= norm;

    return result;
}

std::vector<std::pair<std::string, float>> Inference::getNearestNeighbors(
    const std::vector<std::string>& words,
    const std::vector<std::string>& patient_meta,
    const std::vector<std::string>& provider_meta,
    int k,
    const Matrix& cached_word_vectors) const {

    std::vector<float> query = getCombinedVector(words, patient_meta, provider_meta);

    float q_norm = 0.0f;
    for (float v : query) q_norm += v * v;
    q_norm = std::sqrt(q_norm);
    if (q_norm < MIN_NORM) {
        std::cerr << "Warning: query vector has near-zero norm.\n";
        return {};
    }

    const int vocab_size  = vocab_.wordSize();
    const int num_threads = omp_get_max_threads();
    // cached_word_vectors must be vocab_size x d_out (projected).
    const bool use_cache  = (cached_word_vectors.rows() == vocab_size &&
                              cached_word_vectors.cols() == d_out_);

    using PQ = std::priority_queue<std::pair<float,int>,
                                   std::vector<std::pair<float,int>>,
                                   std::greater<>>;
    std::vector<PQ> local_queues(num_threads);

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < vocab_size; ++i) {
        int tid = omp_get_thread_num();

        std::vector<float> tmp;
        const float* row;
        if (use_cache) {
            row = cached_word_vectors.row(i);
        } else {
            tmp = getProjectedWordVector(vocab_.getWord(i));
            row = tmp.data();
        }

        float dot = 0.0f, row_norm_sq = 0.0f;
        for (int j = 0; j < d_out_; ++j) {
            dot         += query[j] * row[j];
            row_norm_sq += row[j]   * row[j];
        }
        float row_norm = std::sqrt(row_norm_sq);
        if (row_norm < MIN_NORM) continue;

        float sim = dot / (q_norm * row_norm);
        auto& q   = local_queues[tid];
        if (static_cast<int>(q.size()) < k)      q.emplace(sim, i);
        else if (sim > q.top().first) { q.pop(); q.emplace(sim, i); }
    }

    std::vector<std::pair<float,int>> merged;
    merged.reserve(k * num_threads);
    for (auto& q : local_queues)
        while (!q.empty()) { merged.push_back(q.top()); q.pop(); }

    int k_actual = std::min(k, static_cast<int>(merged.size()));
    std::partial_sort(merged.begin(), merged.begin() + k_actual, merged.end(),
                      std::greater<>());

    std::vector<std::pair<std::string, float>> results;
    results.reserve(k_actual);
    for (int i = 0; i < k_actual; ++i)
        results.emplace_back(vocab_.getWord(merged[i].second), merged[i].first);

    return results;
}

} // namespace fasttext

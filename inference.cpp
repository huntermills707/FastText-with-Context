#include "inference.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <queue>

namespace fasttext {

Inference::Inference(const Vocabulary& vocab,
                     const Matrix& input_matrix,
                     const Matrix& ngram_matrix,
                     const Matrix& metadata_matrix,
                     int min_n, int max_n)
    : vocab_(vocab),
      input_matrix_(input_matrix),
      ngram_matrix_(ngram_matrix),
      metadata_matrix_(metadata_matrix),
      min_n_(min_n), max_n_(max_n),
      dim_(ngram_matrix.cols()) {}

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

// word_embedding (if in vocab) + sum(ngram_embeddings).
// Out-of-vocabulary words fall back to ngrams only, giving them a nonzero vector
// derived from morphology alone.
std::vector<float> Inference::getWordVector(const std::string& word) const {
    std::vector<float> vec(dim_, 0.0f);

    int word_idx = vocab_.getWordIdx(word);
    if (word_idx >= 0) {
        const float* wr = input_matrix_.row(word_idx);
        for (int j = 0; j < dim_; ++j) vec[j] += wr[j];
    }

    for (int idx : getNgramIndices(word)) {
        const float* nr = ngram_matrix_.row(idx);
        for (int j = 0; j < dim_; ++j) vec[j] += nr[j];
    }

    return vec;
}

std::vector<float> Inference::getMetadataVector(const std::vector<std::string>& metadata) const {
    std::vector<float> vec(dim_, 0.0f);
    for (const auto& meta : metadata) {
        int idx = vocab_.getMetadataIdx(meta);
        if (idx < 0) continue;
        const float* row = metadata_matrix_.row(idx);
        for (int j = 0; j < dim_; ++j) vec[j] += row[j];
    }
    return vec;
}

// Combined vector: average word vectors and average metadata vectors separately,
// L2-normalise each group, sum the two normalised averages, then L2-normalise
// the result. This gives words and metadata strict 50/50 directional contribution
// regardless of how many tokens are in each group.
//
// If one group is empty its normalised average is the zero vector, so the
// combined vector collapses to a normalised version of the other group alone —
// a smooth, well-defined boundary condition.
std::vector<float> Inference::getCombinedVector(const std::vector<std::string>& words,
                                                const std::vector<std::string>& metadata) const {
    auto l2norm = [&](std::vector<float>& v) {
        float n = 0.0f;
        for (float x : v) n += x * x;
        n = std::sqrt(n);
        if (n > MIN_NORM) for (float& x : v) x /= n;
    };

    // Average word vectors.
    std::vector<float> word_avg(dim_, 0.0f);
    for (const auto& word : words) {
        std::vector<float> wv = getWordVector(word);
        for (int j = 0; j < dim_; ++j) word_avg[j] += wv[j];
    }
    if (!words.empty())
        for (int j = 0; j < dim_; ++j) word_avg[j] /= static_cast<float>(words.size());

    // Average metadata vectors (only over fields present in the vocabulary).
    std::vector<float> meta_avg(dim_, 0.0f);
    int meta_count = 0;
    for (const auto& meta : metadata) {
        int idx = vocab_.getMetadataIdx(meta);
        if (idx < 0) continue;
        const float* row = metadata_matrix_.row(idx);
        for (int j = 0; j < dim_; ++j) meta_avg[j] += row[j];
        ++meta_count;
    }
    if (meta_count > 0)
        for (int j = 0; j < dim_; ++j) meta_avg[j] /= static_cast<float>(meta_count);

    // Normalise each group before summing so neither dominates by magnitude.
    l2norm(word_avg);
    l2norm(meta_avg);

    std::vector<float> combined(dim_);
    for (int j = 0; j < dim_; ++j) combined[j] = word_avg[j] + meta_avg[j];
    l2norm(combined);

    return combined;
}

std::vector<std::pair<std::string, float>> Inference::getNearestNeighbors(
    const std::vector<std::string>& words,
    const std::vector<std::string>& metadata,
    int k,
    const Matrix& cached_word_vectors) const {

    std::vector<float> query = getCombinedVector(words, metadata);

    float q_norm = 0.0f;
    for (float v : query) q_norm += v * v;
    q_norm = std::sqrt(q_norm);
    if (q_norm < MIN_NORM) {
        std::cerr << "Warning: query vector has near-zero norm.\n";
        return {};
    }

    const int vocab_size  = vocab_.wordSize();
    const int num_threads = omp_get_max_threads();
    const bool use_cache  = (cached_word_vectors.rows() == vocab_size);

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
            tmp = getWordVector(vocab_.getWord(i));
            row = tmp.data();
        }

        float dot = 0.0f, row_norm_sq = 0.0f;
        for (int j = 0; j < dim_; ++j) {
            dot        += query[j] * row[j];
            row_norm_sq += row[j]  * row[j];
        }
        float row_norm = std::sqrt(row_norm_sq);
        if (row_norm < MIN_NORM) continue;

        float sim = dot / (q_norm * row_norm);
        auto& q   = local_queues[tid];
        if (static_cast<int>(q.size()) < k)       q.emplace(sim, i);
        else if (sim > q.top().first) { q.pop();  q.emplace(sim, i); }
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

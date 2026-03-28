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
                     const std::vector<float>& gate_bias,
                     const std::vector<float>& alpha,
                     int min_n, int max_n)
    : vocab_(vocab),
      input_matrix_(input_matrix),
      ngram_matrix_(ngram_matrix),
      metadata_matrix_(metadata_matrix),
      gate_bias_(gate_bias),
      alpha_(alpha),
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
// OOV words fall back to ngrams only.
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

// Raw sum of metadata embeddings (no gating) — used for diagnostic purposes.
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

// Gated composition:
//   word_part = avg(word_emb_i + sum(ngrams_i))   [full word vectors]
//   gate      = sigmoid(gate_bias + word_part)      [element-wise; OOV-safe via ngrams]
//   meta_part = sum_k(alpha_k * gate * meta_emb_k)
//   combined  = word_part + meta_part
//   result    = L2_normalise(combined)
//
// Using word_part for the gate ensures OOV words receive a morphologically-
// informed gate signal via their ngram embeddings rather than collapsing to
// sigmoid(gate_bias). Only one L2 normalisation is applied to the final
// composite, preserving the relative word/metadata magnitudes as learned.
std::vector<float> Inference::getCombinedVector(const std::vector<std::string>& words,
                                                const std::vector<std::string>& metadata) const {
    auto l2norm = [&](std::vector<float>& v) {
        float n = 0.0f;
        for (float x : v) n += x * x;
        n = std::sqrt(n);
        if (n > MIN_NORM) for (float& x : v) x /= n;
    };

    std::vector<float> word_part(dim_, 0.0f);

    for (const auto& word : words) {
        std::vector<float> wv = getWordVector(word);
        for (int j = 0; j < dim_; ++j) word_part[j] += wv[j];
    }
    if (!words.empty()) {
        for (int j = 0; j < dim_; ++j)
            word_part[j] /= static_cast<float>(words.size());
    }

    // Compute gate = sigmoid(gate_bias + word_part).
    std::vector<float> gate(dim_);
    if (!gate_bias_.empty()) {
        for (int j = 0; j < dim_; ++j) {
            float x = gate_bias_[j] + word_part[j];
            x = std::max(-20.0f, std::min(20.0f, x));
            gate[j] = 1.0f / (1.0f + std::exp(-x));
        }
    } else {
        // Fallback if gate_bias not available (old model format): gate = 0.5.
        std::fill(gate.begin(), gate.end(), 0.5f);
    }

    // Compute gated metadata: sum_k(alpha_k * gate * meta_emb_k).
    std::vector<float> combined(word_part);
    for (const auto& meta : metadata) {
        int idx = vocab_.getMetadataIdx(meta);
        if (idx < 0) continue;
        float a = (!alpha_.empty()) ? alpha_[idx] : 1.0f;
        const float* row = metadata_matrix_.row(idx);
        for (int j = 0; j < dim_; ++j)
            combined[j] += a * gate[j] * row[j];
    }

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

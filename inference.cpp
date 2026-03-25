#include "inference.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <queue>

namespace fasttext {

Inference::Inference(const Vocabulary& vocab, const Matrix& input_matrix,
                     const Matrix& ngram_matrix, const Matrix& context_matrix,
                     int min_n, int max_n)
    : vocab_(vocab), input_matrix_(input_matrix), ngram_matrix_(ngram_matrix),
      context_matrix_(context_matrix), min_n_(min_n), max_n_(max_n),
      dim_(input_matrix.cols()) {}

uint64_t Inference::hash(const std::string& str) const {
    uint64_t h = 14695981039346656037ULL;
    for (char c : str) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

std::vector<int> Inference::getNgramIndices(const std::string& word) const {
    std::vector<int> indices;
    std::string bordered = "<" + word + ">";
    
    for (int n = min_n_; n <= max_n_; ++n) {
        for (size_t i = 0; i + n <= bordered.size(); ++i) {
            std::string ngram = bordered.substr(i, n);
            uint64_t h = hash(ngram);
            int idx = h % ngram_matrix_.rows();
            indices.push_back(idx);
        }
    }
    
    return indices;
}

std::vector<float> Inference::getWordVector(const std::string& word) const {
    std::vector<float> vec(dim_, 0.0f);
    
    // Add word embedding
    int word_idx = vocab_.getWordIdx(word);
    if (word_idx >= 0) {
        const float* row = input_matrix_.row(word_idx);
        for (int j = 0; j < dim_; ++j) {
            vec[j] += row[j];
        }
    }
    
    // Add n-gram embeddings
    auto ngram_indices = getNgramIndices(word);
    for (int idx : ngram_indices) {
        const float* ngram_row = ngram_matrix_.row(idx);
        for (int j = 0; j < dim_; ++j) {
            vec[j] += ngram_row[j];
        }
    }
    
    return vec;
}

std::vector<float> Inference::getContextVector(const std::vector<std::string>& contexts) const {
    std::vector<float> vec(dim_, 0.0f);
    
    for (const auto& ctx : contexts) {
        int ctx_idx = vocab_.getContextIdx(ctx);
        if (ctx_idx >= 0) {
            const float* row = context_matrix_.row(ctx_idx);
            for (int j = 0; j < dim_; ++j) {
                vec[j] += row[j];
            }
        }
    }

    return vec;
}

std::vector<float> Inference::getCombinedVector(const std::vector<std::string>& words,
                                                const std::vector<std::string>& contexts) const {
    std::vector<float> combined(dim_, 0.0f);
    
    // Sum word vectors
    for (const auto& word : words) {
        std::vector<float> w_vec = getWordVector(word);
        for (int j = 0; j < dim_; ++j) {
            combined[j] += w_vec[j];
        }
    }
    
    // Add context vector
    std::vector<float> ctx_vec = getContextVector(contexts);
    for (int j = 0; j < dim_; ++j) {
        combined[j] += ctx_vec[j];
    }
    
    return combined;
}

std::vector<std::pair<std::string, float>> Inference::getNearestNeighbors(
    const std::vector<std::string>& words,
    const std::vector<std::string>& contexts,
    int k) const {
    
    std::vector<float> query_vec = getCombinedVector(words, contexts);
    
    float query_norm = 0.0f;
    for (float v : query_vec) query_norm += v * v;
    query_norm = std::sqrt(query_norm);
    
    if (query_norm < MIN_NORM) {
        std::cerr << "Warning: Query vector has near-zero magnitude." << std::endl;
        return {};
    }
    
    int vocab_size = input_matrix_.rows();
    int num_threads = omp_get_max_threads();
    
    // Thread-local storage for top-k candidates
    // Using a min-heap of size k for each thread
    using PQueue = std::priority_queue<std::pair<float, int>,
                                       std::vector<std::pair<float, int>>,
                                       std::greater<>>;
    std::vector<PQueue> local_queues(num_threads);
    
    // Parallel similarity computation
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < vocab_size; ++i) {
        int tid = omp_get_thread_num();
        
        const float* row = input_matrix_.row(i);
        
        // Compute dot product and row norm in one pass
        float dot = 0.0f;
        float row_norm_sq = 0.0f;
        
        for (int j = 0; j < dim_; ++j) {
            dot += query_vec[j] * row[j];
            row_norm_sq += row[j] * row[j];
        }
        
        float row_norm = std::sqrt(row_norm_sq);
        if (row_norm < MIN_NORM) continue;
        
        float sim = dot / (query_norm * row_norm);
        
        auto& q = local_queues[tid];
        if (static_cast<int>(q.size()) < k) {
            q.emplace(sim, i);
        } else if (sim > q.top().first) {
            q.pop();
            q.emplace(sim, i);
        }
    }
    
    // Merge thread-local results
    std::vector<std::pair<float, int>> merged;
    merged.reserve(k * num_threads);
    
    for (auto& q : local_queues) {
        while (!q.empty()) {
            merged.push_back(q.top());
            q.pop();
        }
    }
    
    // Final sort (descending by similarity)
    int k_actual = std::min(k, static_cast<int>(merged.size()));
    std::partial_sort(merged.begin(), merged.begin() + k_actual, merged.end(),
                      std::greater<>());
    
    // Prepare output
    std::vector<std::pair<std::string, float>> results;
    results.reserve(k_actual);
    
    for (int i = 0; i < k_actual; ++i) {
        int idx = merged[i].second;
        float score = merged[i].first;
        results.emplace_back(vocab_.getWord(idx), score);
    }
    
    return results;
}

} // namespace fasttext

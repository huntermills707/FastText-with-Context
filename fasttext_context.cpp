#include "fasttext_context.h"
#include <iostream>

namespace fasttext {

FastTextContext::FastTextContext(int dim, int epoch, float lr, 
                                                  int min_n, int max_n, int threshold)
    : dim_(dim), epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n), 
      threshold_(threshold), rng_(std::random_device{}()), 
      uniform_(0.0, 1.0), normal_(0.0, 1.0) {}

uint64_t FastTextContext::hash(const std::string& str) {
    uint64_t h = 14695981039346656037ULL;
    for (char c : str) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

std::vector<TrainingSample> FastTextContext::parseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<TrainingSample> samples;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        TrainingSample sample;
        std::stringstream ss(line);
        std::string field;
        
        while (std::getline(ss, field, '|')) {
            if (ss.eof()) {
                std::istringstream sentence_stream(field);
                std::string word;
                while (sentence_stream >> word) {
                    sample.words.push_back(word);
                }
            } else {
                sample.context_fields.push_back(field);
            }
        }
        
        if (!sample.words.empty()) {
            samples.push_back(sample);
        }
    }
    
    file.close();
    return samples;
}

void FastTextContext::buildVocab(const std::vector<TrainingSample>& samples) {
    std::unordered_map<std::string, int> word_freq;
    std::unordered_map<std::string, int> context_freq;
    
    for (const auto& sample : samples) {
        for (const auto& word : sample.words) {
            word_freq[word]++;
        }
        for (const auto& ctx : sample.context_fields) {
            context_freq[ctx]++;
        }
    }
    
    int word_idx = 0;
    for (const auto& [word, count] : word_freq) {
        if (count >= threshold_) {
            word2idx_[word] = word_idx++;
            word_counts_.push_back(count);
        }
    }
    
    int ctx_idx = 0;
    for (const auto& [ctx, count] : context_freq) {
        context2idx_[ctx] = ctx_idx++;
    }
    
    std::cout << "Word vocabulary: " << word2idx_.size() << std::endl;
    std::cout << "Context vocabulary: " << context2idx_.size() << std::endl;
}

void FastTextContext::initializeMatrices() {
    int vocab_size = word2idx_.size();
    int context_vocab_size = context2idx_.size();
    int ngram_buckets = 2000000;
    
    // Word input matrix - small random initialization
    input_matrix_.resize(vocab_size);
    for (auto& vec : input_matrix_) {
        vec.resize(dim_);
        for (float& v : vec) {
            v = normal_(rng_) / dim_;
        }
    }
    
    // Output matrix - initialized to zero
    output_matrix_.resize(vocab_size);
    for (auto& vec : output_matrix_) {
        vec.resize(dim_);
        for (float& v : vec) {
            v = 0.0f;
        }
    }
    
    // N-gram matrix
    ngram_matrix_.resize(ngram_buckets);
    for (auto& vec : ngram_matrix_) {
        vec.resize(dim_);
        for (float& v : vec) {
            v = normal_(rng_) / dim_;
        }
    }
    
    // Context matrix
    context_matrix_.resize(context_vocab_size);
    for (auto& vec : context_matrix_) {
        vec.resize(dim_);
        for (float& v : vec) {
            v = normal_(rng_) / dim_;
        }
    }
    
    std::cout << "Word embeddings: " << vocab_size << " x " << dim_ << std::endl;
    std::cout << "Context embeddings: " << context_vocab_size << " x " << dim_ << std::endl;
    std::cout << "N-gram buckets: " << ngram_buckets << " x " << dim_ << std::endl;
}

std::vector<int> FastTextContext::getNgramIndices(const std::string& word) {
    std::vector<int> indices;
    std::string bordered = "<" + word + ">";
    
    for (int n = min_n_; n <= max_n_; ++n) {
        for (size_t i = 0; i + n <= bordered.size(); ++i) {
            std::string ngram = bordered.substr(i, n);
            uint64_t h = hash(ngram);
            int idx = h % ngram_matrix_.size();
            indices.push_back(idx);
        }
    }
    
    return indices;
}

std::vector<float> FastTextContext::computeWordVector(const std::string& word) {
    std::vector<float> vec(dim_, 0.0f);
    
    // Add word embedding
    if (word2idx_.count(word)) {
        int idx = word2idx_[word];
        for (int i = 0; i < dim_; ++i) {
            vec[i] += input_matrix_[idx][i];
        }
    }
    
    // Add n-gram embeddings
    auto ngram_indices = getNgramIndices(word);
    for (int idx : ngram_indices) {
        for (int i = 0; i < dim_; ++i) {
            vec[i] += ngram_matrix_[idx][i];
        }
    }
    
    return vec;
}

std::vector<float> FastTextContext::computeContextVector(const std::vector<std::string>& contexts) {
    std::vector<float> vec(dim_, 0.0f);
    int count = 0;
    
    for (const auto& ctx : contexts) {
        if (context2idx_.count(ctx)) {
            int idx = context2idx_[ctx];
            for (int i = 0; i < dim_; ++i) {
                vec[i] += context_matrix_[idx][i];
            }
            count++;
        }
    }
    
    if (count > 0) {
        for (float& v : vec) v /= count;  // Average
}
    
    return vec;
  }

std::vector<float> FastTextContext::combineVectors(
    const std::vector<float>& word_vec,
    const std::vector<float>& context_vec) {
    
    std::vector<float> combined(dim_);
    for (int i = 0; i < dim_; ++i) {
        combined[i] = word_vec[i] + context_vec[i];
    }
    
    return combined;
}

void FastTextContext::negativeSampling(const std::vector<float>& combined_input,
                                               const std::vector<int>& labels,
                                               float grad_scale) {
    int num_negatives = 5;
    int vocab_size = word2idx_.size();
    
    for (int label : labels) {
        // Positive sample
        for (int i = 0; i < dim_; ++i) {
            float grad = grad_scale * combined_input[i];
            output_matrix_[label][i] -= lr_ * grad;
        }
        
        // Negative samples
        for (int n = 0; n < num_negatives; ++n) {
            int neg_idx = static_cast<int>(uniform_(rng_) * vocab_size);
            if (neg_idx == label) continue;
            
            for (int i = 0; i < dim_; ++i) {
                float grad = grad_scale * combined_input[i];
                output_matrix_[neg_idx][i] -= lr_ * grad;
            }
        }
    }
}

void FastTextContext::trainModel(const std::vector<TrainingSample>& samples) {
    int total_words = 0;
    for (const auto& sample : samples) {
        total_words += sample.words.size();
    }
    
    std::cout << "Training on " << total_words << " words across " 
              << samples.size() << " samples..." << std::endl;
    
    for (int epoch = 0; epoch < epoch_; ++epoch) {
        int word_count = 0;
        
        for (const auto& sample : samples) {
            if (sample.words.empty()) continue;
            
            std::vector<float> context_vec = computeContextVector(sample.context_fields);
            
            for (size_t i = 0; i < sample.words.size(); ++i) {
                std::vector<float> word_vec = computeWordVector(sample.words[i]);
                std::vector<float> combined = combineVectors(word_vec, context_vec);
                
                std::vector<int> context;
                int window = 5;
                for (int j = std::max(0, static_cast<int>(i) - window); 
                     j < std::min(static_cast<int>(sample.words.size()), static_cast<int>(i) + window + 1); ++j) {
                    if (static_cast<size_t>(j) != i && word2idx_.count(sample.words[j])) {
                        context.push_back(word2idx_[sample.words[j]]);
                    }
                }
                
                negativeSampling(combined, context, 1.0f);
                word_count++;
            }
            
            if (word_count % 10000 == 0) {
                float progress = static_cast<float>(word_count) / total_words;
                std::cout << "\rProgress: " << (progress * 100) << "%" 
                          << " (epoch " << (epoch + 1) << "/" << epoch_ << ")" 
                          << std::flush;
            }
        }
        std::cout << std::endl;
    }
}

void FastTextContext::train(const std::string& filename) {
    auto samples = parseFile(filename);
    std::cout << "Parsed " << samples.size() << " training samples" << std::endl;
    
    buildVocab(samples);
    initializeMatrices();
    trainModel(samples);
    
    std::cout << "Training complete!" << std::endl;
}

std::vector<float> FastTextContext::getWordVector(const std::string& word) {
    return computeWordVector(word);
}

std::vector<float> FastTextContext::getContextVector(const std::string& context_field) {
    std::vector<float> vec(dim_, 0.0f);
    
    if (context2idx_.count(context_field)) {
        int idx = context2idx_[context_field];
        for (int i = 0; i < dim_; ++i) {
            vec[i] = context_matrix_[idx][i];
        }
    }
    
    return vec;
}

std::vector<float> FastTextContext::getCombinedVector(const std::string& word,
                                                               const std::vector<std::string>& contexts) {
    std::vector<float> word_vec = computeWordVector(word);
    std::vector<float> context_vec = computeContextVector(contexts);
    return combineVectors(word_vec, context_vec);
}

std::vector<std::pair<std::string, float>> FastTextContext::getNearestNeighbors(
    const std::string& word, int k) {
    
    std::vector<float> query_vec = computeWordVector(word);
    std::vector<std::pair<std::string, float>> results;
    
    // Compute query vector magnitude
    float query_norm = 0.0f;
    for (float v : query_vec) query_norm += v * v;
    query_norm = std::sqrt(query_norm);
    
    // Handle zero-norm query
    if (query_norm < 1e-8f) {
        std::cerr << "Warning: Query vector has near-zero magnitude" << std::endl;
        return results;
    }
    
    // Normalize query vector
    for (float& v : query_vec) v /= query_norm;
    
    // Compute cosine similarity with all words
    for (const auto& [w, idx] : word2idx_) {
        const std::vector<float>& word_vec = input_matrix_[idx];
        
        // Compute word vector magnitude
        float word_norm = 0.0f;
        for (float v : word_vec) word_norm += v * v;
        word_norm = std::sqrt(word_norm);
        
        // Handle zero-norm word
        if (word_norm < 1e-8f) continue;
        
        float dot = 0.0f;
        for (int i = 0; i < dim_; ++i) {
            dot += query_vec[i] * word_vec[i];
        }
        
        // Cosine similarity = dot / (norm_query * norm_word)
        float similarity = dot / word_norm;
        
        results.emplace_back(w, similarity);
    }
    
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    if (results.size() > static_cast<size_t>(k)) {
        results.resize(k);
    }
    
    return results;
}

} // namespace fasttext

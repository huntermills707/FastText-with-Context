#include "fasttext_context.h"
#include <iostream>
#include <functional>
#include <cstring>
#include <iomanip>
#include <numeric>

namespace fasttext {

FastTextContext::FastTextContext(int dim, int epoch, float lr, int min_n, int max_n, 
                                  int threshold, int merge_interval, int chunk_size)
    : dim_(dim), epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n), 
      threshold_(threshold), merge_interval_(merge_interval), chunk_size_(chunk_size),
      huffman_root_(nullptr),
      rng_(std::random_device{}()), uniform_(0.0, 1.0), normal_(0.0, 1.0),
      num_threads_(omp_get_max_threads()) {}

FastTextContext::~FastTextContext() {
    cleanupHuffmanTree(huffman_root_);
}

void FastTextContext::cleanupHuffmanTree(HuffmanNode* node) {
    if (node == nullptr) return;
    cleanupHuffmanTree(node->left);
    cleanupHuffmanTree(node->right);
    delete node;
}

uint64_t FastTextContext::hash(const std::string& str) {
    uint64_t h = 14695981039346656037ULL;
    for (char c : str) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

// DIAGNOSTIC: Check vocabulary state
void FastTextContext::diagnoseVocabulary() {
    std::cout << "\n=== VOCABULARY DIAGNOSTICS ===" << std::endl;
    std::cout << "word2idx_ size: " << word2idx_.size() << std::endl;
    std::cout << "word_freqs_ size: " << word_freqs_.size() << std::endl;
    std::cout << "word_counts_ size: " << word_counts_.size() << std::endl;
    std::cout << "Threshold: " << threshold_ << std::endl;
    
    if (!word_freqs_.empty()) {
        double min_freq = *std::min_element(word_freqs_.begin(), word_freqs_.end());
        double max_freq = *std::max_element(word_freqs_.begin(), word_freqs_.end());
        double avg_freq = std::accumulate(word_freqs_.begin(), word_freqs_.end(), 0.0) / word_freqs_.size();
        
        std::cout << "Word frequency stats:" << std::endl;
        std::cout << "  Min: " << min_freq << std::endl;
        std::cout << "  Max: " << max_freq << std::endl;
        std::cout << "  Avg: " << avg_freq << std::endl;
    }
    
    if (!word2idx_.empty()) {
        std::cout << "Sample words in vocabulary:" << std::endl;
        int count = 0;
        for (const auto& [word, idx] : word2idx_) {
            if (count++ < 5) {
                std::cout << "  '" << word << "' -> idx " << idx << std::endl;
            } else {
                break;
            }
        }
    }
    std::cout << "================================\n" << std::endl;
}

// DIAGNOSTIC: Check Huffman tree state
void FastTextContext::diagnoseHuffmanTree() {
    std::cout << "\n=== HUFFMAN TREE DIAGNOSTICS ===" << std::endl;
    std::cout << "huffman_root_: " << (huffman_root_ ? "exists" : "NULL") << std::endl;
    std::cout << "word_codes_ size: " << word_codes_.size() << std::endl;
    std::cout << "word_paths_ size: " << word_paths_.size() << std::endl;
    
    if (!word_paths_.empty()) {
        std::cout << "Path depth statistics:" << std::endl;
        size_t min_depth = word_paths_[0].size();
        size_t max_depth = word_paths_[0].size();
        size_t total_depth = 0;
        
        for (const auto& path : word_paths_) {
            min_depth = std::min(min_depth, path.size());
            max_depth = std::max(max_depth, path.size());
            total_depth += path.size();
        }
        
        std::cout << "  Min depth: " << min_depth << std::endl;
        std::cout << "  Max depth: " << max_depth << std::endl;
        std::cout << "  Avg depth: " << (total_depth / word_paths_.size()) << std::endl;
        
        // Check for depth 0 issue
        if (min_depth == 0) {
            std::cerr << "  ERROR: Some words have depth 0!" << std::endl;
            int zero_depth_count = 0;
            for (const auto& path : word_paths_) {
                if (path.size() == 0) zero_depth_count++;
            }
            std::cerr << "  Words with depth 0: " << zero_depth_count << std::endl;
        }
    }
    
    if (huffman_root_) {
        std::cout << "Root node: word_idx=" << huffman_root_->word_idx 
                  << ", frequency=" << huffman_root_->frequency << std::endl;
    }

    std::cout << "================================\n" << std::endl;
}
   

// PASS 1: Count vocabulary without storing samples

void FastTextContext::buildVocabFromCounts(const std::unordered_map<std::string, int>& word_freq,
                                          const std::unordered_map<std::string, int>& context_freq) {
    int word_idx = 0;
    int filtered_count = 0;
    
    for (const auto& [word, count] : word_freq) {
        if (count >= threshold_) {
            word2idx_[word] = word_idx++;
            word_counts_.push_back(count);
            word_freqs_.push_back(static_cast<double>(count));
        } else {
            filtered_count++;
        }
    }
    
    int ctx_idx = 0;
    for (const auto& [ctx, count] : context_freq) {
        context2idx_[ctx] = ctx_idx++;
    }
    
    std::cout << "\n=== VOCABULARY BUILD SUMMARY ===" << std::endl;
    std::cout << "Total unique words in file: " << word_freq.size() << std::endl;
    std::cout << "Words meeting threshold (" << threshold_ << "): " << word_idx << std::endl;
    std::cout << "Words filtered out: " << filtered_count << std::endl;
    std::cout << "Context fields: " << context2idx_.size() << std::endl;
    std::cout << "=================================\n" << std::endl;
    
    // Run diagnostics
    diagnoseVocabulary();
}

void FastTextContext::buildHuffmanTree() {
    int vocab_size = word2idx_.size();
    std::cout << "\n=== HUFFMAN TREE CONSTRUCTION ===" << std::endl;
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    
    if (vocab_size == 0) {
        std::cerr << "ERROR: No words in vocabulary! Cannot build Huffman tree." << std::endl;
        return;
    }
    
    if (vocab_size == 1) {
        std::cerr << "WARNING: Only 1 word in vocabulary. Huffman tree will have depth 0." << std::endl;
        word_codes_.resize(1);
        word_paths_.resize(1);
        word_codes_[0] = {};
        word_paths_[0] = {};
        return;
    }
    
    auto cmp = [](HuffmanNode* a, HuffmanNode* b) {
        return a->frequency > b->frequency;
    };
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, decltype(cmp)> pq(cmp);
    
    std::vector<HuffmanNode*> nodes(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        nodes[i] = new HuffmanNode(i, word_freqs_[i]);
        pq.push(nodes[i]);
    }
    
    std::cout << "Priority queue initialized with " << vocab_size << " leaf nodes" << std::endl;
    
    int internal_nodes = 0;
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();
        
        HuffmanNode* parent = new HuffmanNode(-1, left->frequency + right->frequency);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
        internal_nodes++;
    }
    
    huffman_root_ = pq.top();
    
    std::cout << "Tree constructed with " << internal_nodes << " internal nodes" << std::endl;
    std::cout << "Root node frequency: " << huffman_root_->frequency << std::endl;
    
    word_codes_.resize(vocab_size);
    word_paths_.resize(vocab_size);
    
    std::vector<int> code;
    std::vector<int> path;
    std::vector<int> path_nodes;
    generateCodes(huffman_root_, code, path, path_nodes);  // Changed from nodes[i] to huffman_root_
    
    std::cout << "Codes and paths generated for " << vocab_size << " words" << std::endl;
    diagnoseHuffmanTree();
    std::cout << "================================\n" << std::endl;
}

void FastTextContext::generateCodes(HuffmanNode* node, std::vector<int>& code,
                                    std::vector<int>& path, std::vector<int>& path_nodes) {
    if (node == nullptr) return;
    
    if (node->word_idx >= 0) {
        // Leaf node - store the code and path
        word_codes_[node->word_idx] = code;
        word_paths_[node->word_idx] = path;
        return;
    }
    
    int path_idx = path.size();
    path.push_back(path_idx);
    
    code.push_back(0);
    generateCodes(node->left, code, path, path_nodes);
    code.pop_back();
    
    code.push_back(1);
    generateCodes(node->right, code, path, path_nodes);
    code.pop_back();
    
    path.pop_back();
}

void FastTextContext::countVocabulary(const std::string& filename, 
                                     std::unordered_map<std::string, int>& word_freq,
                                     std::unordered_map<std::string, int>& context_freq) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    int line_count = 0;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        line_count++;
        
        std::stringstream ss(line);
        std::string field;
        
        while (std::getline(ss, field, '|')) {
            if (ss.eof()) {
                // Last field is sentence
                std::istringstream sentence_stream(field);
                std::string word;
                while (sentence_stream >> word) {
                    word_freq[word]++;
                }
            } else {
                // Context fields
                context_freq[field]++;
            }
        }
        
        if (line_count % 100000 == 0) {
            std::cout << "\rCounting vocabulary: " << line_count << " lines..." << std::flush;
        }
    }
    
    file.close();
    std::cout << "\nProcessed " << line_count << " lines for vocabulary counting" << std::endl;
}

bool FastTextContext::parseNextSample(std::ifstream& file, StreamingSample& sample) {
    std::string line;
    
    // Clear the sample for reuse
    sample.context_fields.clear();
    sample.words.clear();
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string field;
        
        while (std::getline(ss, field, '|')) {
            if (ss.eof()) {
                // Last field is sentence
                std::istringstream sentence_stream(field);
                std::string word;
                while (sentence_stream >> word) {
                    sample.words.push_back(word);
                }
            } else {
                // Context fields
                sample.context_fields.push_back(field);
            }
        }
        
        if (!sample.words.empty()) {
            return true;
        }
    }
    
    return false;
}

void FastTextContext::initializeMatrices() {
    int vocab_size = word2idx_.size();
    int context_vocab_size = context2idx_.size();
    int ngram_buckets = 2000000;
    
    thread_local_grads_.resize(num_threads_);
    int num_internal_nodes = std::max(1, vocab_size - 1);
    for (int t = 0; t < num_threads_; ++t) {
        thread_local_grads_[t].resize(num_internal_nodes);
        for (auto& vec : thread_local_grads_[t]) {
            vec.resize(dim_, 0.0f);
        }
    }
    
    input_matrix_.resize(vocab_size);
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; ++i) {
        input_matrix_[i].resize(dim_);
        for (int j = 0; j < dim_; ++j) {
            input_matrix_[i][j] = normal_(rng_) / dim_;
        }
    }
    
    output_matrix_.resize(num_internal_nodes);
    #pragma omp parallel for
    for (int i = 0; i < num_internal_nodes; ++i) {
        output_matrix_[i].resize(dim_);
        for (int j = 0; j < dim_; ++j) {
            output_matrix_[i][j] = normal_(rng_) / dim_;
        }
    }
    
    ngram_matrix_.resize(ngram_buckets);
    #pragma omp parallel for
    for (int i = 0; i < ngram_buckets; ++i) {
        ngram_matrix_[i].resize(dim_);
        for (int j = 0; j < dim_; ++j) {
            ngram_matrix_[i][j] = normal_(rng_) / dim_;
        }
    }
    
    context_matrix_.resize(context_vocab_size);
    #pragma omp parallel for
    for (int i = 0; i < context_vocab_size; ++i) {
        context_matrix_[i].resize(dim_);
        for (int j = 0; j < dim_; ++j) {
            context_matrix_[i][j] = normal_(rng_) / dim_;
        }
    }
    
    std::cout << "Word embeddings: " << vocab_size << " x " << dim_ << std::endl;
    std::cout << "Hierarchical softmax nodes: " << num_internal_nodes << " x " << dim_ << std::endl;
    std::cout << "Context embeddings: " << context_vocab_size << " x " << dim_ << std::endl;
    std::cout << "OpenMP threads: " << num_threads_ << std::endl;
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
    
    if (word2idx_.count(word)) {
        int idx = word2idx_[word];
        for (int i = 0; i < dim_; ++i) {
            vec[i] += input_matrix_[idx][i];
        }
    }
    
    auto ngram_indices = getNgramIndices(word);
    if (!ngram_indices.empty()) {
        float* reduction_buffer = new float[dim_]();
        
        #pragma omp parallel for reduction(+:reduction_buffer[:dim_])
        for (int n = 0; n < static_cast<int>(ngram_indices.size()); ++n) {
            int idx = ngram_indices[n];
            for (int i = 0; i < dim_; ++i) {
                reduction_buffer[i] += ngram_matrix_[idx][i];
            }
        }
        
        for (int i = 0; i < dim_; ++i) {
            vec[i] += reduction_buffer[i];
        }
        
        delete[] reduction_buffer;
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
        for (float& v : vec) v /= count;
    }
    
    return vec;
}

std::vector<float> FastTextContext::combineVectorsAdditive(
    const std::vector<float>& word_vec,
    const std::vector<float>& context_vec) {
    
    std::vector<float> combined(dim_);
    for (int i = 0; i < dim_; ++i) {
        combined[i] = word_vec[i] + context_vec[i];
    }
    
    return combined;
}

void FastTextContext::mergeThreadLocalGradients() {
    int num_internal_nodes = output_matrix_.size();
    
    #pragma omp parallel for
    for (int node = 0; node < num_internal_nodes; ++node) {
        for (int j = 0; j < dim_; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < num_threads_; ++t) {
                sum += thread_local_grads_[t][node][j];
            }
            output_matrix_[node][j] += sum;
        }
    }
    
    for (int t = 0; t < num_threads_; ++t) {
        for (auto& vec : thread_local_grads_[t]) {
            std::fill(vec.begin(), vec.end(), 0.0f);
        }
    }
}

void FastTextContext::hierarchicalSoftmax(const std::vector<float>& combined_input,
                                          int target_word_idx,
                                          float grad_scale) {  // grad_scale is now the current LR
    const std::vector<int>& path = word_paths_[target_word_idx];
    const std::vector<int>& code = word_codes_[target_word_idx];
    
    int thread_id = omp_get_thread_num();
    
    for (size_t i = 0; i < path.size(); ++i) {
        int node_idx = path[i];
        int direction = code[i];
        
        float dot = 0.0f;
        for (int j = 0; j < dim_; ++j) {
            dot += output_matrix_[node_idx][j] * combined_input[j];
        }
        
        dot = std::max(-20.0f, std::min(20.0f, dot));
        float sigmoid = 1.0f / (1.0f + std::exp(-dot));
        float target = (direction == 0) ? 1.0f : 0.0f;
        float error = target - sigmoid;
        
        // Use grad_scale (current LR) instead of lr_
        for (int j = 0; j < dim_; ++j) {
            thread_local_grads_[thread_id][node_idx][j] += grad_scale * error * combined_input[j];
        }
    }
}

// STREAMING TRAINING: Two-pass approach
void FastTextContext::trainStreaming(const std::string& filename) {
    std::cout << "=== Streaming Training (Two-Pass) ===" << std::endl;
    
    // PASS 1: Count vocabulary
    std::cout << "\nPass 1/2: Counting vocabulary..." << std::endl;
    std::unordered_map<std::string, int> word_freq;
    std::unordered_map<std::string, int> context_freq;
    countVocabulary(filename, word_freq, context_freq);
    
    // Build vocabulary from counts
    buildVocabFromCounts(word_freq, context_freq);
    
    // Build Huffman tree
    buildHuffmanTree();
    
    // Check if tree is valid before proceeding
    if (word_paths_.empty() || (word_paths_.size() == 1 && word_paths_[0].empty())) {
        std::cerr << "\nERROR: Huffman tree construction failed or produced invalid paths." << std::endl;
        std::cerr << "Cannot proceed with training. Check vocabulary diagnostics above." << std::endl;
        return;
    }
    
    // Initialize matrices
    initializeMatrices();
    
    // PASS 2: Train by streaming samples
    std::cout << "\nPass 2/2: Training by streaming samples..." << std::endl;
    trainModelStreaming(filename);
    
    std::cout << "\nStreaming training complete!" << std::endl;
}

void FastTextContext::trainModelStreaming(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Count total samples
    std::cout << "Counting total samples for progress tracking..." << std::endl;
    std::ifstream count_file(filename);
    int total_samples = 0;
    std::string line;
    while (std::getline(count_file, line)) {
        if (!line.empty()) total_samples++;
    }
    count_file.close();
    
    // Total training steps across all epochs
    long long total_steps = static_cast<long long>(total_samples) * epoch_;
    
    std::cout << "Total samples: " << total_samples << std::endl;
    std::cout << "Total epochs: " << epoch_ << std::endl;
    std::cout << "Total training steps: " << total_steps << std::endl;
    std::cout << "Chunk size: " << chunk_size_ << " samples" << std::endl;
    std::cout << "Merge interval: " << merge_interval_ << " samples" << std::endl;
    std::cout << "Initial LR: " << lr_ << std::endl;
    std::cout << "LR Schedule: Linear decay over " << epoch_ << " epochs" << std::endl;
    
    const int MERGE_INTERVAL = merge_interval_;
    const int CHUNK_SIZE = chunk_size_;
    const int PROGRESS_UPDATE_INTERVAL = 1000;
    const int BAR_WIDTH = 40;
    const float LR_FLOOR = lr_ * 0.0001f;
    
    long long global_step = 0;  // Track across all epochs
    
    for (int epoch = 0; epoch < epoch_; ++epoch) {
        file.clear();
        file.seekg(0, std::ios::beg);
        
        int word_count = 0;
        int samples_processed = 0;
        int last_progress_update = 0;
        
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epoch_ << " | " 
                  << std::string(BAR_WIDTH, ' ') << " | 0.00%" << std::flush;
        
        std::vector<StreamingSample> chunk;
        chunk.reserve(CHUNK_SIZE);
        StreamingSample sample;
        
        while (parseNextSample(file, sample)) {
            chunk.push_back(std::move(sample));
            
            if (chunk.size() >= CHUNK_SIZE) {
                // CORRECT: Global progress across all epochs
                float progress = static_cast<float>(global_step) / total_steps;
                float current_lr = lr_ * (1.0f - progress);
                current_lr = std::max(current_lr, LR_FLOOR);
                
                #pragma omp parallel for schedule(dynamic) reduction(+:word_count)
                for (int s = 0; s < static_cast<int>(chunk.size()); ++s) {
                    const auto& current_sample = chunk[s];
                    if (current_sample.words.empty()) continue;
                    
                    std::vector<float> context_vec = computeContextVector(current_sample.context_fields);
                    
                    for (size_t i = 0; i < current_sample.words.size(); ++i) {
                        std::string word = current_sample.words[i];
                        if (!word2idx_.count(word)) continue;
                        
                        int target_idx = word2idx_[word];
                        std::vector<float> word_vec = computeWordVector(word);
                        std::vector<float> combined = combineVectorsAdditive(word_vec, context_vec);
                        
                        hierarchicalSoftmax(combined, target_idx, current_lr);
                        word_count++;
                    }
                }
                
                global_step += chunk.size();
                samples_processed += chunk.size();
                
                if (samples_processed - last_progress_update >= PROGRESS_UPDATE_INTERVAL) {
                    #pragma omp critical
                    {
                        float epoch_progress = static_cast<float>(samples_processed) / total_samples;
                        float global_progress = static_cast<float>(global_step) / total_steps;
                        int filled_width = static_cast<int>(BAR_WIDTH * epoch_progress);
                        
                        std::cout << "\rEpoch " << (epoch + 1) << "/" << epoch_ << " | "
                                  << "[" << std::string(filled_width, '#') 
                                  << std::string(BAR_WIDTH - filled_width, ' ') << "] "
                                  << std::fixed << std::setprecision(2) << (epoch_progress * 100) << "%"
                                  << " | LR: " << std::scientific << std::setprecision(2) << current_lr
                                  << std::flush;
                        
                        last_progress_update = samples_processed;
                    }
                }
                
                if (samples_processed % MERGE_INTERVAL == 0) {
                    #pragma omp critical
                    mergeThreadLocalGradients();
                }
                
                chunk.clear();
            }
        }
        
        // Process remaining samples
        if (!chunk.empty()) {
            float progress = static_cast<float>(global_step) / total_steps;
            float current_lr = lr_ * (1.0f - progress);
            current_lr = std::max(current_lr, LR_FLOOR);
            
            #pragma omp parallel for schedule(dynamic) reduction(+:word_count)
            for (int s = 0; s < static_cast<int>(chunk.size()); ++s) {
                const auto& current_sample = chunk[s];
                if (current_sample.words.empty()) continue;
                
                std::vector<float> context_vec = computeContextVector(current_sample.context_fields);
                
                for (size_t i = 0; i < current_sample.words.size(); ++i) {
                    std::string word = current_sample.words[i];
                    if (!word2idx_.count(word)) continue;
                    
                    int target_idx = word2idx_[word];
                    std::vector<float> word_vec = computeWordVector(word);
                    std::vector<float> combined = combineVectorsAdditive(word_vec, context_vec);
                    
                    hierarchicalSoftmax(combined, target_idx, current_lr);
                    word_count++;
                }
            }
            
            global_step += chunk.size();
            samples_processed += chunk.size();
            chunk.clear();
        }
        
        #pragma omp critical
        mergeThreadLocalGradients();
        
        float final_progress = static_cast<float>(global_step) / total_steps;
        float final_lr = lr_ * (1.0f - final_progress);
        final_lr = std::max(final_lr, LR_FLOOR);
        
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epoch_ << " | " 
                  << "[" << std::string(BAR_WIDTH, '#') << "] 100.00%"
                  << " | LR: " << std::scientific << std::setprecision(2) << final_lr
                  << " | Done!" << std::endl;
    }
    
    file.close();
}

void FastTextContext::saveModel(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&min_n_), sizeof(min_n_));
    out.write(reinterpret_cast<const char*>(&max_n_), sizeof(max_n_));
    out.write(reinterpret_cast<const char*>(&threshold_), sizeof(threshold_));
    
    int vocab_size = word2idx_.size();
    int ctx_size = context2idx_.size();
    int ngram_size = ngram_matrix_.size();
    int output_size = output_matrix_.size();
    
    out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    out.write(reinterpret_cast<const char*>(&ctx_size), sizeof(ctx_size));
    out.write(reinterpret_cast<const char*>(&ngram_size), sizeof(ngram_size));
    out.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

    for (const auto& pair : word2idx_) {
        uint32_t len = pair.first.length();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(pair.first.c_str(), len);
        out.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }

    for (const auto& pair : context2idx_) {
        uint32_t len = pair.first.length();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(pair.first.c_str(), len);
        out.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }

    auto writeMatrix = [&](const std::vector<std::vector<float>>& mat) {
        for (const auto& row : mat) {
            out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
    };

    writeMatrix(input_matrix_);
    writeMatrix(output_matrix_);
    writeMatrix(ngram_matrix_);
    writeMatrix(context_matrix_);

    auto writeIntVec = [&](const std::vector<std::vector<int>>& vec) {
        for (const auto& v : vec) {
            uint32_t len = v.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (len > 0) {
                out.write(reinterpret_cast<const char*>(v.data()), len * sizeof(int));
            }
        }
    };

    writeIntVec(word_codes_);
    writeIntVec(word_paths_);

    out.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void FastTextContext::loadModel(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&min_n_), sizeof(min_n_));
    in.read(reinterpret_cast<char*>(&max_n_), sizeof(max_n_));
    in.read(reinterpret_cast<char*>(&threshold_), sizeof(threshold_));
    
    int vocab_size, ctx_size, ngram_size, output_size;
    in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    in.read(reinterpret_cast<char*>(&ctx_size), sizeof(ctx_size));
    in.read(reinterpret_cast<char*>(&ngram_size), sizeof(ngram_size));
    in.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

    input_matrix_.resize(vocab_size, std::vector<float>(dim_));
    output_matrix_.resize(output_size, std::vector<float>(dim_));
    ngram_matrix_.resize(ngram_size, std::vector<float>(dim_));
    context_matrix_.resize(ctx_size, std::vector<float>(dim_));

    word2idx_.clear();
    for (int i = 0; i < vocab_size; ++i) {
        uint32_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string word(len, '\0');
        in.read(&word[0], len);
        int idx;
        in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        word2idx_[word] = idx;
    }

    context2idx_.clear();
    for (int i = 0; i < ctx_size; ++i) {
        uint32_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string ctx(len, '\0');
        in.read(&ctx[0], len);
        int idx;
        in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        context2idx_[ctx] = idx;
    }

    auto readMatrix = [&](std::vector<std::vector<float>>& mat) {
        for (auto& row : mat) {
            in.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
        }
    };

    readMatrix(input_matrix_);
    readMatrix(output_matrix_);
    readMatrix(ngram_matrix_);
    readMatrix(context_matrix_);

    auto readIntVec = [&](std::vector<std::vector<int>>& vec) {
        vec.resize(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            uint32_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            vec[i].resize(len);
            if (len > 0) {
                in.read(reinterpret_cast<char*>(vec[i].data()), len * sizeof(int));
            }
        }
    };

    readIntVec(word_codes_);
    readIntVec(word_paths_);

    huffman_root_ = nullptr; 

    in.close();
    std::cout << "Model loaded from " << filename << std::endl;
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

std::vector<float> FastTextContext::getCombinedVector(const std::vector<std::string>& words, 
                                                      const std::vector<std::string>& contexts) {
    std::vector<float> combined_vec(dim_, 0.0f);
    
    // Sum all word vectors
    for (const auto& word : words) {
        std::vector<float> w_vec = computeWordVector(word);
        for (int i = 0; i < dim_; ++i) {
            combined_vec[i] += w_vec[i];
        }
    }
    
    // Sum all context vectors
    std::vector<float> ctx_vec = computeContextVector(contexts);
    for (int i = 0; i < dim_; ++i) {
        combined_vec[i] += ctx_vec[i];
    }
    
    // Normalize the final sum
    float norm = 0.0f;
    for (float v : combined_vec) norm += v * v;
    norm = std::sqrt(norm);
    
    if (norm > 1e-8f) {
        for (float& v : combined_vec) v /= norm;
    }
    
    return combined_vec;
}

std::vector<std::pair<std::string, float>> FastTextContext::getNearestNeighbors(
    const std::vector<std::string>& words, 
    const std::vector<std::string>& contexts, 
    int k) {
    
    // Compute the query vector by summing all inputs
    std::vector<float> query_vec = getCombinedVector(words, contexts);
    
    // Check for zero vector
    float query_norm = 0.0f;
    for (float v : query_vec) query_norm += v * v;
    query_norm = std::sqrt(query_norm);
    
    if (query_norm < 1e-8f) {
        std::cerr << "Warning: Query vector has near-zero magnitude. Check inputs." << std::endl;
        return {};
    }
    
    // Already normalized in getCombinedVector, but safe to ensure
    for (float& v : query_vec) v /= query_norm;
    
    int vocab_size = word2idx_.size();
    int num_threads = omp_get_max_threads();
    
    // Thread-local results
    std::vector<std::vector<std::pair<std::string, float>>> local_results(num_threads);
    
    // Parallel search
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < vocab_size; ++i) {
        int thread_id = omp_get_thread_num();
        auto& local = local_results[thread_id];
        
        auto it = word2idx_.begin();
        std::advance(it, i);
        const std::string& w = it->first;
        int idx = it->second;
        
        const std::vector<float>& word_vec = input_matrix_[idx];
        
        float word_norm = 0.0f;
        for (float v : word_vec) word_norm += v * v;
        word_norm = std::sqrt(word_norm);
        
        if (word_norm < 1e-8f) continue;
        
        float dot = 0.0f;
        for (int j = 0; j < dim_; ++j) {
            dot += query_vec[j] * word_vec[j];
        }
        
        float similarity = dot / word_norm;
        local.emplace_back(w, similarity);
    }
    
    // Merge and sort
    std::vector<std::pair<std::string, float>> results;
    results.reserve(vocab_size);
    for (const auto& local : local_results) {
        results.insert(results.end(), local.begin(), local.end());
    }
    
    std::partial_sort(results.begin(), results.begin() + std::min(k, static_cast<int>(results.size())), 
                      results.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
    
    if (results.size() > static_cast<size_t>(k)) {
        results.resize(k);
    }
    
    return results;
}

} // namespace fasttext

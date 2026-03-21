#include "fasttext_context.h"
#include <iostream>
#include <functional>
#include <cstring>
#include <iomanip> 

namespace fasttext {

FastTextContext::FastTextContext(int dim, int epoch, float lr, int min_n, int max_n, int threshold)
    : dim_(dim), epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n), 
      threshold_(threshold), huffman_root_(nullptr),
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
            word_freqs_.push_back(static_cast<double>(count));
        }
    }
    
    int ctx_idx = 0;
    for (const auto& [ctx, count] : context_freq) {
        context2idx_[ctx] = ctx_idx++;
    }
    
    std::cout << "Word vocabulary: " << word2idx_.size() << std::endl;
    std::cout << "Context vocabulary: " << context2idx_.size() << std::endl;
}

void FastTextContext::buildHuffmanTree() {
    int vocab_size = word2idx_.size();
    if (vocab_size == 0) return;
    
    auto cmp = [](HuffmanNode* a, HuffmanNode* b) {
        return a->frequency > b->frequency;
    };
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, decltype(cmp)> pq(cmp);
    
    std::vector<HuffmanNode*> nodes(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        nodes[i] = new HuffmanNode(i, word_freqs_[i]);
        pq.push(nodes[i]);
    }
    
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();
        
        HuffmanNode* parent = new HuffmanNode(-1, left->frequency + right->frequency);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
    }
    
    huffman_root_ = pq.top();
    
    word_codes_.resize(vocab_size);
    word_paths_.resize(vocab_size);
    
    for (int i = 0; i < vocab_size; ++i) {
        std::vector<int> code;
        std::vector<int> path;
        std::vector<int> path_nodes;
        generateCodes(nodes[i], code, path, path_nodes);
        word_codes_[i] = code;
        word_paths_[i] = path;
    }
    
    std::cout << "Huffman tree built. Max depth: " 
              << (word_codes_.empty() ? 0 : word_codes_[0].size()) << std::endl;
}

void FastTextContext::generateCodes(HuffmanNode* node, std::vector<int>& code,
                                    std::vector<int>& path, std::vector<int>& path_nodes) {
    if (node == nullptr) return;
    
    if (node->word_idx >= 0) {
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

void FastTextContext::initializeMatrices() {
    int vocab_size = word2idx_.size();
    int context_vocab_size = context2idx_.size();
    int ngram_buckets = 2000000;
    
    // Initialize thread-local gradient buffers
    thread_local_grads_.resize(num_threads_);
    int num_internal_nodes = std::max(1, vocab_size - 1);
    for (int t = 0; t < num_threads_; ++t) {
        thread_local_grads_[t].resize(num_internal_nodes);
        for (auto& vec : thread_local_grads_[t]) {
            vec.resize(dim_, 0.0f);
        }
    }
    
    // Word input matrix
    input_matrix_.resize(vocab_size);
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; ++i) {
        input_matrix_[i].resize(dim_);
        for (int j = 0; j < dim_; ++j) {
            input_matrix_[i][j] = normal_(rng_) / dim_;
        }
    }
    
    // Output matrix
    int num_internal_nodes_out = std::max(1, vocab_size - 1);
    output_matrix_.resize(num_internal_nodes_out);
    #pragma omp parallel for
    for (int i = 0; i < num_internal_nodes_out; ++i) {
        output_matrix_[i].resize(dim_);
        for (int j = 0; j < dim_; ++j) {
            output_matrix_[i][j] = normal_(rng_) / dim_;
        }
    }
    
    // N-gram matrix
    ngram_matrix_.resize(ngram_buckets);
    #pragma omp parallel for
    for (int i = 0; i < ngram_buckets; ++i) {
        ngram_matrix_[i].resize(dim_);
        for (int j = 0; j < dim_; ++j) {
            ngram_matrix_[i][j] = normal_(rng_) / dim_;
        }
    }
    
    // Context matrix
    context_matrix_.resize(context_vocab_size);
    #pragma omp parallel for
    for (int i = 0; i < context_vocab_size; ++i) {
        context_matrix_[i].resize(dim_);
        for (int j = 0; j < dim_; ++j) {
            context_matrix_[i][j] = normal_(rng_) / dim_;
        }
    }
    
    std::cout << "Word embeddings: " << vocab_size << " x " << dim_ << std::endl;
    std::cout << "Hierarchical softmax nodes: " << num_internal_nodes_out << " x " << dim_ << std::endl;
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
    
    // Add word embedding
    if (word2idx_.count(word)) {
        int idx = word2idx_[word];
        for (int i = 0; i < dim_; ++i) {
            vec[i] += input_matrix_[idx][i];
        }
    }
    
    // Add n-gram embeddings
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
                                          float grad_scale) {
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
        
        for (int j = 0; j < dim_; ++j) {
            thread_local_grads_[thread_id][node_idx][j] += lr_ * error * combined_input[j];
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
    std::cout << "Using " << num_threads_ << " OpenMP threads with thread-local gradients" << std::endl;
    std::cout << "Merge interval: every 100,000 samples" << std::endl;
    
    const int MERGE_INTERVAL = 100000;
    const int PROGRESS_UPDATE_INTERVAL = 10000;
    const int BAR_WIDTH = 40;
    
    for (int epoch = 0; epoch < epoch_; ++epoch) {
        int word_count = 0;
        int samples_processed = 0;
        int last_progress_update = 0;
        
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epoch_ << " | " 
                  << std::string(BAR_WIDTH, ' ') << " | 0.00%" << std::flush;
        
        #pragma omp parallel for schedule(static) reduction(+:word_count)
        for (int s = 0; s < static_cast<int>(samples.size()); ++s) {
            const auto& sample = samples[s];
            if (sample.words.empty()) continue;
            
            int thread_id = omp_get_thread_num();
            std::vector<float> context_vec = computeContextVector(sample.context_fields);
            
            for (size_t i = 0; i < sample.words.size(); ++i) {
                std::string word = sample.words[i];
                if (!word2idx_.count(word)) continue;
                
                int target_idx = word2idx_[word];
                std::vector<float> word_vec = computeWordVector(word);
                std::vector<float> combined = combineVectorsAdditive(word_vec, context_vec);
                
                // Accumulate gradients locally (no lock)
                hierarchicalSoftmax(combined, target_idx, 1.0f);
                
                word_count++;
            }
            
            // Track samples processed (thread-local counter)
            #pragma omp atomic
            samples_processed++;
            
            // Update progress bar periodically
            if (samples_processed - last_progress_update >= PROGRESS_UPDATE_INTERVAL) {
                #pragma omp critical
                {
                    float progress = static_cast<float>(samples_processed) / samples.size();
                    int filled_width = static_cast<int>(BAR_WIDTH * progress);
                    
                    std::cout << "\rEpoch " << (epoch + 1) << "/" << epoch_ << " | "
                              << "[" << std::string(filled_width, '#') 
                              << std::string(BAR_WIDTH - filled_width, ' ') << "] "
                              << std::fixed << std::setprecision(2) << (progress * 100) << "%"
                              << std::flush;
                    
                    last_progress_update = samples_processed;
                }
            }
            
            // Merge gradients every MERGE_INTERVAL samples
            if (samples_processed % MERGE_INTERVAL == 0) {
                #pragma omp critical
                mergeThreadLocalGradients();
            }
        }
        
        // Final merge at end of epoch
        #pragma omp critical
        mergeThreadLocalGradients();
        
        // Clear the progress line and show completion
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epoch_ << " | " 
                  << "[" << std::string(BAR_WIDTH, '#') << "] 100.00%"
                  << " | Done!" << std::endl;
    }
    
    std::cout << "\nTraining complete!" << std::endl;
}

void FastTextContext::train(const std::string& filename) {
    auto samples = parseFile(filename);
    std::cout << "Parsed " << samples.size() << " training samples" << std::endl;
    
    buildVocab(samples);
    buildHuffmanTree();
    initializeMatrices();
    trainModel(samples);

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

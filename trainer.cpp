#include "trainer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>  // FIXED: Added for setprecision
#include <omp.h>

namespace fasttext {

Trainer::Trainer(int dim, int epoch, float lr, int min_n, int max_n,
                 int chunk_size, int ngram_buckets)
    : dim_(dim), epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n),
      chunk_size_(chunk_size), ngram_buckets_(ngram_buckets) {
    
    int num_threads = omp_get_max_threads();
    rngs_.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        rngs_[i].seed(std::random_device{}() + i);
    }
}

uint64_t Trainer::hash(const std::string& str) const {
    uint64_t h = 14695981039346656037ULL;
    for (char c : str) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

std::vector<int> Trainer::getNgramIndices(const std::string& word) const {
    std::vector<int> indices;
    std::string bordered = "<" + word + ">";
    
    for (int n = min_n_; n <= max_n_; ++n) {
        for (size_t i = 0; i + n <= bordered.size(); ++i) {
            std::string ngram = bordered.substr(i, n);
            uint64_t h = hash(ngram);
            int idx = h % ngram_buckets_;  // Use member variable
            indices.push_back(idx);
        }
    }
    
    return indices;
}

bool Trainer::parseLine(const std::string& line, StreamingSample& sample) const {
    sample.context_fields.clear();
    sample.words.clear();
    
    if (line.empty()) return false;
    
    std::stringstream ss(line);
    std::string field;
    std::vector<std::string> all_fields;
    
    while (std::getline(ss, field, '|')) {
        all_fields.push_back(field);
    }
    
    if (all_fields.empty()) return false;
    
    // Last field is sentence, rest are context
    for (size_t i = 0; i < all_fields.size() - 1; ++i) {
        sample.context_fields.push_back(all_fields[i]);
    }
    
    // Tokenize last field
    std::istringstream sentence_stream(all_fields.back());
    std::string word;
    while (sentence_stream >> word) {
        sample.words.push_back(word);
    }
    
    return !sample.words.empty();
}

void Trainer::processSample(const StreamingSample& sample, const Vocabulary& vocab,
                            Matrix& input_matrix, Matrix& output_matrix,
                            Matrix& ngram_matrix, Matrix& context_matrix,
                            float current_lr, int thread_id) {
    
    if (sample.words.empty()) return;
    
    // Compute context vector and track active context indices
    std::vector<float> context_vec(dim_, 0.0f);
    std::vector<int> active_ctx_indices;
    
    for (const auto& ctx : sample.context_fields) {
        int ctx_idx = vocab.getContextIdx(ctx);
        if (ctx_idx >= 0) {
            const float* ctx_row = context_matrix.row(ctx_idx);
            for (int j = 0; j < dim_; ++j) {
                context_vec[j] += ctx_row[j];
            }
            active_ctx_indices.push_back(ctx_idx);
        }
    }
     
    // Process each word in the sample
    for (const auto& word : sample.words) {
        int word_idx = vocab.getWordIdx(word);
        if (word_idx < 0) continue;
        
        // Compute word vector (input + n-grams)
        std::vector<float> word_vec(dim_, 0.0f);
        
        // Add input matrix contribution
        const float* input_row = input_matrix.row(word_idx);
        for (int j = 0; j < dim_; ++j) {
            word_vec[j] += input_row[j];
        }
        
        // Add n-gram contributions
        std::vector<int> ngram_indices = getNgramIndices(word);
        for (int idx : ngram_indices) {
            const float* ngram_row = ngram_matrix.row(idx);
            for (int j = 0; j < dim_; ++j) {
                word_vec[j] += ngram_row[j];
            }
        }
        
        // Combined vector = word + context
        std::vector<float> combined(dim_);
        for (int j = 0; j < dim_; ++j) {
            combined[j] = word_vec[j] + context_vec[j];
        }
        
        // Hierarchical softmax forward/backward pass
        const std::vector<int>& path = vocab.word_paths_[word_idx];
        const std::vector<int>& code = vocab.word_codes_[word_idx];
        
        if (path.empty()) continue;
        
        // Accumulate gradients for input matrices (to be applied later)
        std::vector<float> input_grad(dim_, 0.0f);
        
        for (size_t i = 0; i < path.size(); ++i) {
            int node_idx = path[i];
            int direction = code[i];
            
            // Compute dot product
            float dot = 0.0f;
            const float* output_row = output_matrix.row(node_idx);
            for (int j = 0; j < dim_; ++j) {
                dot += output_row[j] * combined[j];
            }
            
            // Clamp and sigmoid
            dot = std::max(-20.0f, std::min(20.0f, dot));
            float sigmoid = 1.0f / (1.0f + std::exp(-dot));
            float target = (direction == 0) ? 1.0f : 0.0f;
            float error = target - sigmoid;
            
            // Gradient for output matrix (accumulated in thread-local storage)
            for (int j = 0; j < dim_; ++j) {
                thread_grads_[thread_id].at(node_idx, j) += current_lr * error * combined[j];
            }
            
            // Gradient flowing back to combined vector
            for (int j = 0; j < dim_; ++j) {
                input_grad[j] += current_lr * error * output_row[j];
            }
        }
        
        // Accumulate word gradient
        for (int j = 0; j < dim_; ++j) {
            thread_input_grads_[thread_id][word_idx * dim_ + j] += input_grad[j];
        }
        
        // Distribute gradient to active contexts
        float grad_scale = 1.0f;
        for (int ctx_idx : active_ctx_indices) {
            int global_ctx_idx = vocab.wordSize() + ctx_idx;
            for (int j = 0; j < dim_; ++j) {
                thread_input_grads_[thread_id][global_ctx_idx * dim_ + j] += input_grad[j] * grad_scale;
            }
        }
    }
}

void Trainer::mergeGradients(Matrix& output_matrix) {
    int num_threads = thread_grads_.size();
    int num_nodes = output_matrix.rows();
    
    // Merge output matrix gradients
    for (int node = 0; node < num_nodes; ++node) {
        for (int j = 0; j < dim_; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < num_threads; ++t) {
                sum += thread_grads_[t].at(node, j);
            }
            output_matrix.at(node, j) += sum;
        }
    }
    
    // Clear thread-local output gradients
    for (int t = 0; t < num_threads; ++t) {
        thread_grads_[t].zero();
    }
}

void Trainer::train(const std::string& filename, Vocabulary& vocab,
                    Matrix& input_matrix, Matrix& output_matrix,
                    Matrix& ngram_matrix, Matrix& context_matrix) {
    
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
    
    long long total_steps = static_cast<long long>(total_samples) * epoch_;
    
    std::cout << "Total samples: " << total_samples << std::endl;
    std::cout << "Total epochs: " << epoch_ << std::endl;
    std::cout << "Total training steps: " << total_steps << std::endl;
    std::cout << "LR Schedule: Linear decay over " << epoch_ << " epochs" << std::endl;
    
    const int CHUNK_SIZE = chunk_size_;
    const int BAR_WIDTH = 40;
    const float LR_FLOOR = lr_ * 0.0001f;
    
    // Initialize thread-local gradient storage
    int num_threads = omp_get_max_threads();
    int num_nodes = vocab.huffmanNodes();
    
    // FIXED: Buffer size should be (total unique entities) * dim
    int total_entities = vocab.wordSize() + vocab.contextSize();
    thread_grads_.resize(num_threads, Matrix(num_nodes, dim_));
    thread_input_grads_.resize(num_threads, std::vector<float>(
        total_entities * dim_, 0.0f));
    
    long long global_step = 0;

    for (int epoch = 0; epoch < epoch_; ++epoch) {
        file.clear();
        file.seekg(0, std::ios::beg);
        
        int samples_processed = 0;
        int last_progress_update = 0;
        
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epoch_ << " | " 
                  << std::string(BAR_WIDTH, ' ') << " | 0.00%" << std::flush;
        
        std::vector<StreamingSample> chunk;
        chunk.reserve(CHUNK_SIZE);
        StreamingSample sample;
        
        while (std::getline(file, line)) {
            if (parseLine(line, sample)) {
                chunk.push_back(std::move(sample));
            }
            
            // Process chunk immediately when full
            if (chunk.size() >= CHUNK_SIZE) {
                float progress = static_cast<float>(global_step) / total_steps;
                float current_lr = lr_ * (1.0f - progress);
                current_lr = std::max(current_lr, LR_FLOOR);
                
                #pragma omp parallel for schedule(dynamic)
                for (int s = 0; s < static_cast<int>(chunk.size()); ++s) {
                    const auto& current_sample = chunk[s];
                    int thread_id = omp_get_thread_num();
                    processSample(current_sample, vocab, input_matrix, output_matrix,
                                  ngram_matrix, context_matrix, current_lr, thread_id);
                }
                
                global_step += chunk.size();
                samples_processed += chunk.size();
                
                // Progress update
                if (samples_processed - last_progress_update >= 1000) {
                    #pragma omp critical
                    {
                        float epoch_progress = static_cast<float>(samples_processed) / total_samples;
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
                
                // MERGE GRADIENTS IMMEDIATELY AFTER PROCESSING CHUNK
                #pragma omp critical
                mergeGradients(output_matrix);
                
                chunk.clear();
            }
        }
        
        // Process remaining samples
        if (!chunk.empty()) {
            float progress = static_cast<float>(global_step) / total_steps;
            float current_lr = lr_ * (1.0f - progress);
            current_lr = std::max(current_lr, LR_FLOOR);
            
            #pragma omp parallel for schedule(dynamic)
            for (int s = 0; s < static_cast<int>(chunk.size()); ++s) {
                const auto& current_sample = chunk[s];
                int thread_id = omp_get_thread_num();
                processSample(current_sample, vocab, input_matrix, output_matrix,
                              ngram_matrix, context_matrix, current_lr, thread_id);
            }
            
            global_step += chunk.size();
            samples_processed += chunk.size();
            
            // MERGE GRADIENTS FOR REMAINING SAMPLES
            #pragma omp critical
            mergeGradients(output_matrix);
            
            chunk.clear();
        }
        
        // No need for extra merge here as it's done after every chunk/remainder
        
        float final_progress = static_cast<float>(global_step) / total_steps;
        float final_lr = lr_ * (1.0f - final_progress);
        final_lr = std::max(final_lr, LR_FLOOR);
        
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epoch_ << " | " 
                  << "[" << std::string(BAR_WIDTH, '#') << "] 100.00%"
                  << " | LR: " << std::scientific << std::setprecision(2) << final_lr
                  << " | Done!" << std::endl;
    }
    
    file.close();
    std::cout << "\nTraining complete!" << std::endl;
}

} // namespace fasttext

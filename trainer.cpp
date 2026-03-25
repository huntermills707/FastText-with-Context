#include "trainer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>

namespace fasttext {

Trainer::Trainer(int dim, int epoch, float lr, int min_n, int max_n,
                 int chunk_size, int ngram_buckets, int window_size)
    : dim_(dim), epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n),
      chunk_size_(chunk_size), ngram_buckets_(ngram_buckets), window_size_(window_size) {
    
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
            int idx = h % ngram_buckets_;
            indices.push_back(idx);
        }
    }
    
    return indices;
}

bool Trainer::parseLine(const std::string& line, StreamingSample& sample) const {
    sample.metadata_fields.clear();
    sample.words.clear();
    
    if (line.empty()) return false;
    
    std::stringstream ss(line);
    std::string field;
    std::vector<std::string> all_fields;
    
    while (std::getline(ss, field, '|')) {
        all_fields.push_back(field);
    }
    
    if (all_fields.empty()) return false;
    
    // Last field is sentence, rest are metadata
    for (size_t i = 0; i < all_fields.size() - 1; ++i) {
        sample.metadata_fields.push_back(all_fields[i]);
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
                            Matrix& ngram_matrix, Matrix& metadata_matrix,
                            float current_lr, int thread_id) {
    
    const std::vector<std::string>& words = sample.words;
    if (words.empty()) return;
    
    // Pre-compute metadata vector (shared across all center words in this sample)
    std::vector<float> metadata_vec(dim_, 0.0f);
    std::vector<int> active_metadata_indices;
    
    for (const auto& meta : sample.metadata_fields) {
        int meta_idx = vocab.getMetadataIdx(meta);
        if (meta_idx >= 0) {
            const float* meta_row = metadata_matrix.row(meta_idx);
            for (int j = 0; j < dim_; ++j) {
                metadata_vec[j] += meta_row[j];
            }
            active_metadata_indices.push_back(meta_idx);
        }
    }
    
    // Skip-gram: For each word in the sentence, treat it as center word
    // and predict surrounding words in the window
    for (int center_pos = 0; center_pos < static_cast<int>(words.size()); ++center_pos) {
        const std::string& center_word = words[center_pos];
        int center_word_idx = vocab.getWordIdx(center_word);
        if (center_word_idx < 0) continue;
        
        // Compute center word vector (n-grams + metadata)
        std::vector<float> center_vec(dim_, 0.0f);
        
        // Add input matrix contribution
        const float* input_row = input_matrix.row(center_word_idx);
        for (int j = 0; j < dim_; ++j) {
            center_vec[j] += input_row[j];
        }
        
        // Add n-gram contributions
        std::vector<int> ngram_indices = getNgramIndices(center_word);
        for (int idx : ngram_indices) {
            const float* ngram_row = ngram_matrix.row(idx);
            for (int j = 0; j < dim_; ++j) {
                center_vec[j] += ngram_row[j];
            }
        }
        
        // Add metadata contribution
        for (int j = 0; j < dim_; ++j) {
            center_vec[j] += metadata_vec[j];
        }
        
        // Accumulate gradients for center word (to be applied later)
        std::vector<float> center_grad(dim_, 0.0f);
        
        // Define window bounds
        int window_start = std::max(0, center_pos - window_size_);
        int window_end = std::min(static_cast<int>(words.size()), center_pos + window_size_ + 1);
        
        // For each word in the window (excluding center), predict it via HS
        for (int context_pos = window_start; context_pos < window_end; ++context_pos) {
            if (context_pos == center_pos) continue;  // Skip center word
            
            const std::string& context_word = words[context_pos];
            int context_word_idx = vocab.getWordIdx(context_word);
            if (context_word_idx < 0) continue;
            
            // Get Huffman path for this context word
            const std::vector<int>& path = vocab.word_paths_[context_word_idx];
            const std::vector<int>& code = vocab.word_codes_[context_word_idx];
            
            if (path.empty()) continue;
            
            // HS forward/backward pass: predict context_word from center_vec
            for (size_t i = 0; i < path.size(); ++i) {
                int node_idx = path[i];
                int direction = code[i];
                
                // Compute dot product
                float dot = 0.0f;
                const float* output_row = output_matrix.row(node_idx);
                for (int j = 0; j < dim_; ++j) {
                    dot += output_row[j] * center_vec[j];
                }
                
                // Clamp and sigmoid
                dot = std::max(-20.0f, std::min(20.0f, dot));
                float sigmoid = 1.0f / (1.0f + std::exp(-dot));
                float target = (direction == 0) ? 1.0f : 0.0f;
                float error = target - sigmoid;
                
                // Gradient for output matrix (accumulated in thread-local storage)
                for (int j = 0; j < dim_; ++j) {
                    thread_grads_[thread_id].at(node_idx, j) += current_lr * error * center_vec[j];
                }
                
                // Gradient flowing back to center vector
                for (int j = 0; j < dim_; ++j) {
                    center_grad[j] += current_lr * error * output_row[j];
                }
            }
        }
        
        // Accumulate center word gradient
        for (int j = 0; j < dim_; ++j) {
            thread_input_grads_[thread_id][center_word_idx * dim_ + j] += center_grad[j];
        }
        
        // Distribute gradient to active metadata (only if metadata was present)
        if (!active_metadata_indices.empty()) {
            for (int meta_idx : active_metadata_indices) {
                int global_meta_idx = vocab.wordSize() + meta_idx;
                for (int j = 0; j < dim_; ++j) {
                    thread_input_grads_[thread_id][global_meta_idx * dim_ + j] += center_grad[j];
                }
            }
        }
    }
}

void Trainer::mergeGradients(Matrix& output_matrix) {
    int num_threads = thread_grads_.size();
    int num_nodes = output_matrix.rows();
    
    // Merge output matrix gradients (Huffman nodes)
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
                    Matrix& ngram_matrix, Matrix& metadata_matrix) {
    
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
    
    // Estimate total steps: samples * avg_words_per_sample * window_pairs
    // Simplified: just use samples * epoch for now
    long long total_steps = static_cast<long long>(total_samples) * epoch_;
    
    std::cout << "Total samples: " << total_samples << std::endl;
    std::cout << "Total epochs: " << epoch_ << std::endl;
    std::cout << "Total training steps: " << total_steps << std::endl;
    std::cout << "Window size: " << window_size_ << " (Skip-gram)" << std::endl;
    std::cout << "LR Schedule: Linear decay over " << epoch_ << " epochs" << std::endl;
    
    const int CHUNK_SIZE = chunk_size_;
    const int BAR_WIDTH = 40;
    const float LR_FLOOR = lr_ * 0.0001f;
    
    // Initialize thread-local gradient storage
    int num_threads = omp_get_max_threads();
    int num_nodes = vocab.huffmanNodes();
    
    // Buffer size should be (total unique entities) * dim
    int total_entities = vocab.wordSize() + vocab.metadataSize();
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
                                  ngram_matrix, metadata_matrix, current_lr, thread_id);
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
                
                // Also apply input/metadata gradients
                #pragma omp critical
                {
                    for (int t = 0; t < num_threads; ++t) {
                        for (size_t i = 0; i < thread_input_grads_[t].size(); ++i) {
                            input_matrix.data()[i] += thread_input_grads_[t][i];
                        }
                        // Clear thread-local input gradients
                        std::fill(thread_input_grads_[t].begin(), thread_input_grads_[t].end(), 0.0f);
                    }
                }
                
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
                              ngram_matrix, metadata_matrix, current_lr, thread_id);
            }
            
            global_step += chunk.size();
            samples_processed += chunk.size();
            
            // MERGE GRADIENTS FOR REMAINING SAMPLES
            #pragma omp critical
            mergeGradients(output_matrix);
            
            // Apply input/metadata gradients
            #pragma omp critical
            {
                for (int t = 0; t < num_threads; ++t) {
                    for (size_t i = 0; i < thread_input_grads_[t].size(); ++i) {
                        input_matrix.data()[i] += thread_input_grads_[t][i];
                    }
                    std::fill(thread_input_grads_[t].begin(), thread_input_grads_[t].end(), 0.0f);
                }
            }
            
            chunk.clear();
        }
        
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

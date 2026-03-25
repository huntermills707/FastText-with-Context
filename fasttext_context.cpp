#include "fasttext_context.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <omp.h>

namespace fasttext {

FastTextContext::FastTextContext(int dim, int epoch, float lr, int min_n, int max_n,
                                 int threshold, int chunk_size, int ngram_buckets)
    : dim_(dim), epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n),
      threshold_(threshold), chunk_size_(chunk_size), ngram_buckets_(ngram_buckets) {}

void FastTextContext::initializeMatrices() {
    int vocab_size = vocab_.wordSize();
    int context_size = vocab_.contextSize();
    int num_huffman_nodes = vocab_.huffmanNodes();
    
    std::cout << "\n=== MATRIX INITIALIZATION ===" << std::endl;
    std::cout << "Word embeddings: " << vocab_size << " x " << dim_ << std::endl;
    std::cout << "Output (HS) nodes: " << num_huffman_nodes << " x " << dim_ << std::endl;
    std::cout << "Context embeddings: " << context_size << " x " << dim_ << std::endl;
    std::cout << "N-gram buckets: " << ngram_buckets_ << " x " << dim_ << std::endl;
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
    
    input_matrix_.resize(vocab_size, dim_);
    output_matrix_.resize(num_huffman_nodes, dim_);
    context_matrix_.resize(context_size, dim_);
    ngram_matrix_.resize(ngram_buckets_, dim_);

    // Initialize with thread-local RNGs to avoid race conditions
    int num_threads = omp_get_max_threads();
    std::vector<std::mt19937> thread_rngs(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        thread_rngs[t].seed(std::random_device{}() + t);
    }
    
    float init_scale = 1.0f / dim_;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937& rng = thread_rngs[tid];
        
        // Input matrix
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < input_matrix_.rows(); ++i) {
            for (int64_t j = 0; j < input_matrix_.cols(); ++j) {
                std::normal_distribution<float> dist(0.0, 1.0);
                input_matrix_.at(i, j) = dist(rng) * init_scale;
            }
        }
        
        // Output matrix
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < output_matrix_.rows(); ++i) {
            for (int64_t j = 0; j < output_matrix_.cols(); ++j) {
                std::normal_distribution<float> dist(0.0, 1.0);
                output_matrix_.at(i, j) = dist(rng) * init_scale;
            }
        }
        
        // Context matrix
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < context_matrix_.rows(); ++i) {
            for (int64_t j = 0; j < context_matrix_.cols(); ++j) {
                std::normal_distribution<float> dist(0.0, 1.0);
                context_matrix_.at(i, j) = dist(rng) * init_scale;
            }
        }
        
        // N-gram matrix
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < ngram_matrix_.rows(); ++i) {
            for (int64_t j = 0; j < ngram_matrix_.cols(); ++j) {
                std::normal_distribution<float> dist(0.0, 1.0);
                ngram_matrix_.at(i, j) = dist(rng) * init_scale;
            }
        }
    }
    
    std::cout << "=============================\n" << std::endl;
}

void FastTextContext::trainStreaming(const std::string& filename) {
    std::cout << "=== FastText Context Streaming Training ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Input:           " << filename << std::endl;
    std::cout << "  Dimension:       " << dim_ << std::endl;
    std::cout << "  Epochs:          " << epoch_ << std::endl;
    std::cout << "  LR:              " << lr_ << std::endl;
    std::cout << "  N-grams:         " << min_n_ << "-" << max_n_ << std::endl;
    std::cout << "  Threshold:       " << threshold_ << std::endl;
    std::cout << "  Chunk Size:      " << chunk_size_ << std::endl;
    std::cout << "  N-gram Buckets:  " << ngram_buckets_ << std::endl;
    std::cout << std::endl;
    
    // PASS 1: Count vocabulary
    std::cout << "Building vocabulary..." << std::endl;
    
    std::unordered_map<std::string, int> word_freq;
    std::unordered_map<std::string, int> context_freq;
    
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
        std::vector<std::string> fields;
        
        while (std::getline(ss, field, '|')) {
            fields.push_back(field);
        }
        
        if (fields.empty()) continue;
        
        // Last field is sentence
        std::istringstream sentence_stream(fields.back());
        std::string word;
        while (sentence_stream >> word) {
            word_freq[word]++;
        }
        
        // Rest are context fields
        for (size_t i = 0; i < fields.size() - 1; ++i) {
            context_freq[fields[i]]++;
        }
        
        if (line_count % 100000 == 0) {
            std::cout << "\rCounting vocabulary: " << line_count << " lines..." << std::flush;
        }
    }
    file.close();
    
    std::cout << "\nProcessed " << line_count << " lines for vocabulary counting" << std::endl;
    
    // Build vocabulary
    vocab_ = Vocabulary(threshold_);
    vocab_.buildFromCounts(word_freq, context_freq);
    
    // Build Huffman tree
    vocab_.buildHuffmanTree();
    
    // Check validity
    if (vocab_.wordSize() == 0) {
        throw std::runtime_error("No words in vocabulary after filtering. Lower the threshold.");
    }
    
    // Initialize matrices
    initializeMatrices();
    
    // PASS 2: Train
    std::cout << "Training..." << std::endl;
    Trainer trainer(dim_, epoch_, lr_, min_n_, max_n_, chunk_size_, ngram_buckets_);
    trainer.train(filename, vocab_, input_matrix_, output_matrix_, ngram_matrix_, context_matrix_);
    
    // Initialize inference engine
    inference_ = std::make_unique<Inference>(vocab_, input_matrix_, ngram_matrix_, 
                                              context_matrix_, min_n_, max_n_);
    
    std::cout << "\nTraining complete!" << std::endl;
}

void FastTextContext::saveModel(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write header
    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&min_n_), sizeof(min_n_));
    out.write(reinterpret_cast<const char*>(&max_n_), sizeof(max_n_));
    out.write(reinterpret_cast<const char*>(&threshold_), sizeof(threshold_));
    
    int vocab_size = vocab_.wordSize();
    int ctx_size = vocab_.contextSize();
    int ngram_size = ngram_matrix_.rows();
    int output_size = output_matrix_.rows();
    
    out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    out.write(reinterpret_cast<const char*>(&ctx_size), sizeof(ctx_size));
    out.write(reinterpret_cast<const char*>(&ngram_size), sizeof(ngram_size));
    out.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));
    
    // Write word vocabulary
    for (int i = 0; i < vocab_size; ++i) {
        const std::string& word = vocab_.getWord(i);
        uint32_t len = word.length();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(word.c_str(), len);
        out.write(reinterpret_cast<const char*>(&i), sizeof(i));
    }
    
    // Write context vocabulary (FIXED: Complete serialization)
    for (int i = 0; i < ctx_size; ++i) {
        const std::string& ctx = vocab_.getContext(i);
        uint32_t len = ctx.length();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(ctx.c_str(), len);
        out.write(reinterpret_cast<const char*>(&i), sizeof(i));
    }
    
    // Write matrices
    input_matrix_.save(out);
    output_matrix_.save(out);
    ngram_matrix_.save(out);
    context_matrix_.save(out);
    
    // Write Huffman codes and paths
    auto writeIntVec = [&](const std::vector<std::vector<int>>& vec) {
        for (const auto& v : vec) {
            uint32_t len = v.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (len > 0) {
                out.write(reinterpret_cast<const char*>(v.data()), len * sizeof(int));
            }
        }
    };
    
    writeIntVec(vocab_.word_codes_);
    writeIntVec(vocab_.word_paths_);
    
    out.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void FastTextContext::loadModel(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Read header
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&min_n_), sizeof(min_n_));
    in.read(reinterpret_cast<char*>(&max_n_), sizeof(max_n_));
    in.read(reinterpret_cast<char*>(&threshold_), sizeof(threshold_));
    
    int vocab_size, ctx_size, ngram_size, output_size;
    in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    in.read(reinterpret_cast<char*>(&ctx_size), sizeof(ctx_size));
    in.read(reinterpret_cast<char*>(&ngram_size), sizeof(ngram_size));
    in.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));
    
    // Resize matrices
    input_matrix_.resize(vocab_size, dim_);
    output_matrix_.resize(output_size, dim_);
    ngram_matrix_.resize(ngram_size, dim_);
    context_matrix_.resize(ctx_size, dim_);
    
    // Read word vocabulary (FIXED: Populate vocab_ properly)
    for (int i = 0; i < vocab_size; ++i) {
        uint32_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string word(len, '\0');
        in.read(&word[0], len);
        int idx;
        in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        vocab_.addWord(idx, word);
    }
    
    // Read context vocabulary (FIXED: Complete deserialization)
    for (int i = 0; i < ctx_size; ++i) {
        uint32_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string ctx(len, '\0');
        in.read(&ctx[0], len);
        int idx;
        in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        vocab_.addContext(idx, ctx);
    }
    
    // Read matrices
    input_matrix_.load(in);
    output_matrix_.load(in);
    ngram_matrix_.load(in);
    context_matrix_.load(in);
    
    // Read Huffman codes and paths
    auto readIntVec = [&](std::vector<std::vector<int>>& vec, int size) {
        vec.resize(size);
        for (int i = 0; i < size; ++i) {
            uint32_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            vec[i].resize(len);
            if (len > 0) {
                in.read(reinterpret_cast<char*>(vec[i].data()), len * sizeof(int));
            }
        }
    };
    
    readIntVec(vocab_.word_codes_, vocab_size);
    readIntVec(vocab_.word_paths_, vocab_size);
    
    in.close();
    
    // Initialize inference engine
    inference_ = std::make_unique<Inference>(vocab_, input_matrix_, ngram_matrix_,
                                              context_matrix_, min_n_, max_n_);
    
    std::cout << "Model loaded from " << filename << std::endl;
}

std::vector<float> FastTextContext::getWordVector(const std::string& word) {
    if (!inference_) {
        throw std::runtime_error("Model not initialized. Train or load a model first.");
    }
    return inference_->getWordVector(word);
}

std::vector<float> FastTextContext::getContextVector(const std::string& context_field) {
    if (!inference_) {
        throw std::runtime_error("Model not initialized. Train or load a model first.");
    }
    return inference_->getContextVector({context_field});
}

std::vector<float> FastTextContext::getCombinedVector(const std::vector<std::string>& words,
                                                       const std::vector<std::string>& contexts) {
    if (!inference_) {
        throw std::runtime_error("Model not initialized. Train or load a model first.");
    }
    return inference_->getCombinedVector(words, contexts);
}

std::vector<std::pair<std::string, float>> FastTextContext::getNearestNeighbors(
    const std::vector<std::string>& words,
    const std::vector<std::string>& contexts,
    int k) {
    if (!inference_) {
        throw std::runtime_error("Model not initialized. Train or load a model first.");
    }
    return inference_->getNearestNeighbors(words, contexts, k);
}

} // namespace fasttext

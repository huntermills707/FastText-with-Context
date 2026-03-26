#include "fasttext_context.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <omp.h>
#include <cmath>

namespace fasttext {

FastTextContext::FastTextContext(int dim, int epoch, float lr,
                                 int min_n, int max_n, int threshold,
                                 int chunk_size, int ngram_buckets,
                                 int window_size, float subsample_t)
    : dim_(dim), epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n),
      threshold_(threshold), chunk_size_(chunk_size), ngram_buckets_(ngram_buckets),
      window_size_(window_size), subsample_t_(subsample_t) {}

void FastTextContext::makeInference() {
    inference_ = std::make_unique<Inference>(vocab_, input_matrix_, ngram_matrix_,
                                             metadata_matrix_, min_n_, max_n_);
}

void FastTextContext::initializeMatrices() {
    const int V  = vocab_.wordSize();
    const int M  = vocab_.metadataSize();
    const int HS = vocab_.huffmanNodes();

    output_matrix_.resize(HS,           dim_);
    input_matrix_.resize(V,             dim_);
    ngram_matrix_.resize(ngram_buckets_, dim_);
    metadata_matrix_.resize(M,           dim_);

    const int T     = omp_get_max_threads();
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim_));
    std::vector<std::mt19937> rngs(T);
    for (int t = 0; t < T; ++t) rngs[t].seed(std::random_device{}() + t);

    auto initMat = [&](Matrix& mat) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < mat.rows(); ++i) {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            int tid = omp_get_thread_num();
            for (int64_t j = 0; j < mat.cols(); ++j)
                mat.at(i, j) = dist(rngs[tid]) * scale;
        }
    };

    initMat(output_matrix_);
    initMat(input_matrix_);
    initMat(ngram_matrix_);
    initMat(metadata_matrix_);

    std::cout << "Matrices: hs_nodes=" << HS << " vocab=" << V
              << " ngram_buckets=" << ngram_buckets_ << " meta=" << M
              << " dim=" << dim_ << " threads=" << T << std::endl;
}

void FastTextContext::precomputeWordVectors() {
    const int V = vocab_.wordSize();
    if (V == 0) { std::cout << "No words to precompute.\n"; return; }
    std::cout << "Precomputing " << V << " word vectors..." << std::endl;

    cached_word_vectors_.resize(V, dim_);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < V; ++i) {
        std::vector<float> vec = inference_->getWordVector(vocab_.getWord(i));
        bool valid = true;
        for (float v : vec) if (!std::isfinite(v)) { valid = false; break; }

        for (int j = 0; j < dim_; ++j)
            cached_word_vectors_.at(i, j) = valid ? vec[j] : 0.0f;
    }
    std::cout << "Word vector cache ready.\n";
}

void FastTextContext::trainStreaming(const std::string& filename) {
    std::cout << "Building vocabulary from " << filename << "..." << std::endl;

    std::unordered_map<std::string, int> word_freq, meta_freq;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);

    std::string line;
    int lc = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        ++lc;
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(ss, field, '|')) fields.push_back(field);
        if (fields.empty()) continue;

        std::istringstream sent(fields.back());
        std::string word;
        while (sent >> word) word_freq[word]++;
        for (size_t i = 0; i + 1 < fields.size(); ++i) meta_freq[fields[i]]++;

        if (lc % 100000 == 0)
            std::cout << "\rCounting: " << lc << " lines" << std::flush;
    }
    file.close();
    std::cout << "\nProcessed " << lc << " lines.\n";

    vocab_ = Vocabulary(threshold_);
    vocab_.buildFromCounts(word_freq, meta_freq);
    vocab_.buildHuffmanTree();
    vocab_.computeDiscardProbs(subsample_t_);

    if (vocab_.wordSize() == 0)
        throw std::runtime_error("No words after filtering. Lower -threshold.");

    initializeMatrices();
    makeInference();

    Trainer trainer(dim_, epoch_, lr_, min_n_, max_n_,
                    chunk_size_, ngram_buckets_, window_size_);
    trainer.train(filename, vocab_, input_matrix_, output_matrix_,
                  ngram_matrix_, metadata_matrix_);

    precomputeWordVectors();
    std::cout << "\nTraining complete." << std::endl;
}

// Helper lambdas for serialising/deserialising vector<vector<int>>.
static void writeVecVec(std::ostream& out, const std::vector<std::vector<int>>& vv) {
    for (const auto& v : vv) {
        uint32_t len = static_cast<uint32_t>(v.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len > 0) out.write(reinterpret_cast<const char*>(v.data()), len * sizeof(int));
    }
}

static void readVecVec(std::istream& in, std::vector<std::vector<int>>& vv, int size) {
    vv.resize(size);
    for (int i = 0; i < size; ++i) {
        uint32_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        vv[i].resize(len);
        if (len > 0) in.read(reinterpret_cast<char*>(vv[i].data()), len * sizeof(int));
    }
}

void FastTextContext::saveModel(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open for writing: " + filename);

    out.write(reinterpret_cast<const char*>(&dim_),        sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&min_n_),      sizeof(min_n_));
    out.write(reinterpret_cast<const char*>(&max_n_),      sizeof(max_n_));
    out.write(reinterpret_cast<const char*>(&threshold_),  sizeof(threshold_));
    out.write(reinterpret_cast<const char*>(&window_size_),sizeof(window_size_));

    int vs = vocab_.wordSize(), ms = vocab_.metadataSize();
    int ng = static_cast<int>(ngram_matrix_.rows());
    int os = static_cast<int>(output_matrix_.rows());
    out.write(reinterpret_cast<const char*>(&vs), sizeof(vs));
    out.write(reinterpret_cast<const char*>(&ms), sizeof(ms));
    out.write(reinterpret_cast<const char*>(&ng), sizeof(ng));
    out.write(reinterpret_cast<const char*>(&os), sizeof(os));

    for (int i = 0; i < vs; ++i) {
        const std::string& w = vocab_.getWord(i);
        uint32_t len = static_cast<uint32_t>(w.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(w.c_str(), len);
        out.write(reinterpret_cast<const char*>(&i), sizeof(i));
    }
    for (int i = 0; i < ms; ++i) {
        const std::string& m = vocab_.getMetadata(i);
        uint32_t len = static_cast<uint32_t>(m.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(m.c_str(), len);
        out.write(reinterpret_cast<const char*>(&i), sizeof(i));
    }

    // Matrix order: output, ngram, input (NEW), metadata.
    output_matrix_.save(out);
    ngram_matrix_.save(out);
    input_matrix_.save(out);
    metadata_matrix_.save(out);

    writeVecVec(out, vocab_.word_codes_);
    writeVecVec(out, vocab_.word_paths_);

    out.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void FastTextContext::loadModel(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open for reading: " + filename);

    in.read(reinterpret_cast<char*>(&dim_),        sizeof(dim_));
    in.read(reinterpret_cast<char*>(&min_n_),      sizeof(min_n_));
    in.read(reinterpret_cast<char*>(&max_n_),      sizeof(max_n_));
    in.read(reinterpret_cast<char*>(&threshold_),  sizeof(threshold_));
    in.read(reinterpret_cast<char*>(&window_size_),sizeof(window_size_));

    int vs, ms, ng, os;
    in.read(reinterpret_cast<char*>(&vs), sizeof(vs));
    in.read(reinterpret_cast<char*>(&ms), sizeof(ms));
    in.read(reinterpret_cast<char*>(&ng), sizeof(ng));
    in.read(reinterpret_cast<char*>(&os), sizeof(os));

    output_matrix_.resize(os, dim_);
    ngram_matrix_.resize(ng, dim_);
    input_matrix_.resize(vs, dim_);
    metadata_matrix_.resize(ms, dim_);

    for (int i = 0; i < vs; ++i) {
        uint32_t len; in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string word(len, '\0'); in.read(&word[0], len);
        int idx;      in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        vocab_.addWord(idx, word);
    }
    for (int i = 0; i < ms; ++i) {
        uint32_t len; in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string meta(len, '\0'); in.read(&meta[0], len);
        int idx;      in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        vocab_.addMetadata(idx, meta);
    }

    output_matrix_.load(in);
    ngram_matrix_.load(in);
    input_matrix_.load(in);
    metadata_matrix_.load(in);

    readVecVec(in, vocab_.word_codes_, vs);
    readVecVec(in, vocab_.word_paths_, vs);

    in.close();
    makeInference();
    precomputeWordVectors();
    std::cout << "Model loaded from " << filename << std::endl;
}

std::vector<float> FastTextContext::getWordVector(const std::string& word) {
    if (!inference_) throw std::runtime_error("Model not initialised.");

    int idx = vocab_.getWordIdx(word);
    if (idx >= 0 && cached_word_vectors_.rows() > idx) {
        const float* row = cached_word_vectors_.row(idx);
        std::vector<float> result(row, row + dim_);
        for (float v : result)
            if (!std::isfinite(v)) return inference_->getWordVector(word);
        return result;
    }
    return inference_->getWordVector(word);
}

std::vector<float> FastTextContext::getMetadataVector(const std::string& field) {
    if (!inference_) throw std::runtime_error("Model not initialised.");
    return inference_->getMetadataVector({field});
}

std::vector<float> FastTextContext::getCombinedVector(const std::vector<std::string>& words,
                                                       const std::vector<std::string>& metadata) {
    if (!inference_) throw std::runtime_error("Model not initialised.");

    std::vector<float> combined(dim_, 0.0f);
    for (const auto& w : words) {
        std::vector<float> wv = getWordVector(w);
        for (int j = 0; j < dim_; ++j) combined[j] += wv[j];
    }
    std::vector<float> mv = inference_->getMetadataVector(metadata);
    for (int j = 0; j < dim_; ++j) combined[j] += mv[j];

    float norm = 0.0f;
    for (float v : combined) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > MIN_NORM)
        for (int j = 0; j < dim_; ++j) combined[j] /= norm;

    return combined;
}

std::vector<std::pair<std::string, float>> FastTextContext::getNearestNeighbors(
    const std::vector<std::string>& words,
    const std::vector<std::string>& metadata,
    int k) {
    if (!inference_) throw std::runtime_error("Model not initialised.");
    return inference_->getNearestNeighbors(words, metadata, k, cached_word_vectors_);
}

} // namespace fasttext

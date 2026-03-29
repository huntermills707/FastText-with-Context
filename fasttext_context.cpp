#include "fasttext_context.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <omp.h>
#include <cmath>

namespace fasttext {

FastTextContext::FastTextContext(int d_w, int d_p, int d_pr, int d_out,
                                 int epoch, float lr,
                                 int min_n, int max_n, int threshold,
                                 int chunk_size, int ngram_buckets,
                                 int window_size, float subsample_t, float grad_clip, float weight_decay)
    : d_w_(d_w), d_p_(d_p), d_pr_(d_pr), d_out_(d_out),
      epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n),
      threshold_(threshold), chunk_size_(chunk_size), ngram_buckets_(ngram_buckets),
      window_size_(window_size), subsample_t_(subsample_t), grad_clip_(grad_clip), weight_decay_(weight_decay) {}

void FastTextContext::makeInference() {
    inference_ = std::make_unique<Inference>(
        vocab_, input_matrix_, ngram_matrix_,
        W_proj_, patient_matrix_, provider_matrix_,
        d_w_, d_p_, d_pr_, d_out_, min_n_, max_n_);
}

void FastTextContext::initializeMatrices() {
    const int V  = vocab_.wordSize();
    const int NP = vocab_.patientSize();
    const int PR = vocab_.providerSize();
    const int HS = vocab_.huffmanNodes();
    const int CD = d_w_ + d_p_ + d_pr_;

    output_matrix_.resize(HS,            d_out_);
    input_matrix_.resize(V,              d_w_);
    ngram_matrix_.resize(ngram_buckets_, d_w_);
    W_proj_.resize(d_out_,              CD);
    patient_matrix_.resize(NP,           d_p_);
    provider_matrix_.resize(PR,          d_pr_);

    const int T = omp_get_max_threads();
    std::vector<std::mt19937> rngs(T);
    for (int t = 0; t < T; ++t) rngs[t].seed(std::random_device{}() + t);

    auto initMat = [&](Matrix& mat, float scale) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < mat.rows(); ++i) {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            int tid = omp_get_thread_num();
            for (int64_t j = 0; j < mat.cols(); ++j)
                mat.at(i, j) = dist(rngs[tid]) * scale;
        }
    };

    // Xavier-style initialisation for each space.
    float scale_w   = 1.0f / std::sqrt(static_cast<float>(d_w_));
    float scale_p   = 1.0f / std::sqrt(static_cast<float>(d_p_));
    float scale_pr  = 1.0f / std::sqrt(static_cast<float>(d_pr_));
    float scale_out = 1.0f / std::sqrt(static_cast<float>(d_out_));

    initMat(output_matrix_,  scale_out);
    initMat(input_matrix_,   scale_w);
    initMat(ngram_matrix_,   scale_w);
    initMat(W_proj_,         scale_out);  // Xavier for projection
    initMat(patient_matrix_, scale_p);
    initMat(provider_matrix_,scale_pr);

    std::cout << "Matrices initialised:\n"
              << "  input:    " << V   << " x " << d_w_   << " (word embeddings)\n"
              << "  ngram:    " << ngram_buckets_ << " x " << d_w_ << "\n"
              << "  output:   " << HS  << " x " << d_out_ << " (HS nodes)\n"
              << "  W_proj:   " << d_out_ << " x " << CD << "\n"
              << "  patient:  " << NP  << " x " << d_p_  << "\n"
              << "  provider: " << PR  << " x " << d_pr_ << "\n"
              << "  threads:  " << T   << "\n" << std::endl;
}

void FastTextContext::precomputeWordVectors() {
    const int V = vocab_.wordSize();
    if (V == 0) { std::cout << "No words to precompute.\n"; return; }
    std::cout << "Precomputing " << V << " projected word vectors (d_out=" << d_out_ << ")..." << std::endl;

    // Cache is vocab_size x d_out (word-only projection, metadata regions zeroed).
    cached_word_vectors_.resize(V, d_out_);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < V; ++i) {
        std::vector<float> projected = inference_->getProjectedWordVector(vocab_.getWord(i));
        bool valid = true;
        for (float v : projected) if (!std::isfinite(v)) { valid = false; break; }

        float* row = cached_word_vectors_.row(i);
        for (int j = 0; j < d_out_; ++j)
            row[j] = valid ? projected[j] : 0.0f;
    }
    std::cout << "Word vector cache ready (" << V << " x " << d_out_ << ").\n";
}

void FastTextContext::trainStreaming(const std::string& filename) {
    std::cout << "Building vocabulary from " << filename << "...\n";

    std::unordered_map<std::string, int> word_freq, patient_freq, provider_freq;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);

    std::string line;
    int lc = 0;
    const std::string delim = " ||| ";

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        ++lc;

        // Split on " ||| " — expect 3 groups.
        std::vector<std::string> groups;
        size_t start = 0;
        while (true) {
            size_t pos = line.find(delim, start);
            if (pos == std::string::npos) { groups.push_back(line.substr(start)); break; }
            groups.push_back(line.substr(start, pos - start));
            start = pos + delim.size();
        }

        if (groups.size() != 3) continue;

        // Patient fields.
        std::istringstream ps(groups[0]);
        std::string tok;
        while (ps >> tok) patient_freq[tok]++;

        // Provider fields.
        std::istringstream prs(groups[1]);
        while (prs >> tok) provider_freq[tok]++;

        // Words.
        std::istringstream ws(groups[2]);
        while (ws >> tok) word_freq[tok]++;

        if (lc % 100000 == 0)
            std::cout << "\rCounting: " << lc << " lines" << std::flush;
    }
    file.close();
    std::cout << "\nProcessed " << lc << " lines.\n";

    vocab_ = Vocabulary(threshold_);
    vocab_.buildFromCounts(word_freq, patient_freq, provider_freq);
    vocab_.buildHuffmanTree();
    vocab_.computeDiscardProbs(subsample_t_);

    if (vocab_.wordSize() == 0)
        throw std::runtime_error("No words after filtering. Lower -threshold.");

    initializeMatrices();
    makeInference();

    Trainer trainer(d_w_, d_p_, d_pr_, d_out_,
                    epoch_, lr_, min_n_, max_n_,
                    chunk_size_, ngram_buckets_, window_size_, grad_clip_, weight_decay_);
    trainer.train(filename, vocab_,
                  input_matrix_, output_matrix_, ngram_matrix_,
                  W_proj_, patient_matrix_, provider_matrix_);

    // Rebuild inference with final matrices, then precompute cache.
    makeInference();
    precomputeWordVectors();
    std::cout << "\nTraining complete." << std::endl;
}

// Helpers for serialising/deserialising vector<vector<int>>.
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

// Binary format:
//   [header: d_w, d_p, d_pr, d_out, min_n, max_n, threshold, window_size]
//   [sizes: vocab_size, patient_size, provider_size, ngram_size, output_size]
//   [word vocabulary entries]
//   [patient vocabulary entries]
//   [provider vocabulary entries]
//   [output_matrix data]      // hs_nodes x d_out
//   [ngram_matrix data]       // ngram_buckets x d_w
//   [input_matrix data]       // vocab_size x d_w
//   [W_proj data]             // d_out x concat_dim
//   [patient_matrix data]     // patient_size x d_p
//   [provider_matrix data]    // provider_size x d_pr
//   [word_codes]
//   [word_paths]
void FastTextContext::saveModel(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open for writing: " + filename);

    out.write(reinterpret_cast<const char*>(&d_w_),         sizeof(d_w_));
    out.write(reinterpret_cast<const char*>(&d_p_),         sizeof(d_p_));
    out.write(reinterpret_cast<const char*>(&d_pr_),        sizeof(d_pr_));
    out.write(reinterpret_cast<const char*>(&d_out_),       sizeof(d_out_));
    out.write(reinterpret_cast<const char*>(&min_n_),       sizeof(min_n_));
    out.write(reinterpret_cast<const char*>(&max_n_),       sizeof(max_n_));
    out.write(reinterpret_cast<const char*>(&threshold_),   sizeof(threshold_));
    out.write(reinterpret_cast<const char*>(&window_size_), sizeof(window_size_));

    int vs  = vocab_.wordSize();
    int np  = vocab_.patientSize();
    int npr = vocab_.providerSize();
    int ng  = static_cast<int>(ngram_matrix_.rows());
    int os  = static_cast<int>(output_matrix_.rows());
    out.write(reinterpret_cast<const char*>(&vs),  sizeof(vs));
    out.write(reinterpret_cast<const char*>(&np),  sizeof(np));
    out.write(reinterpret_cast<const char*>(&npr), sizeof(npr));
    out.write(reinterpret_cast<const char*>(&ng),  sizeof(ng));
    out.write(reinterpret_cast<const char*>(&os),  sizeof(os));

    for (int i = 0; i < vs; ++i) {
        const std::string& w = vocab_.getWord(i);
        uint32_t len = static_cast<uint32_t>(w.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(w.c_str(), len);
        out.write(reinterpret_cast<const char*>(&i), sizeof(i));
    }
    for (int i = 0; i < np; ++i) {
        const std::string& m = vocab_.getPatient(i);
        uint32_t len = static_cast<uint32_t>(m.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(m.c_str(), len);
        out.write(reinterpret_cast<const char*>(&i), sizeof(i));
    }
    for (int i = 0; i < npr; ++i) {
        const std::string& m = vocab_.getProvider(i);
        uint32_t len = static_cast<uint32_t>(m.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(m.c_str(), len);
        out.write(reinterpret_cast<const char*>(&i), sizeof(i));
    }

    // Matrix order: output, ngram, input, W_proj, patient, provider.
    output_matrix_.save(out);
    ngram_matrix_.save(out);
    input_matrix_.save(out);
    W_proj_.save(out);
    patient_matrix_.save(out);
    provider_matrix_.save(out);

    writeVecVec(out, vocab_.word_codes_);
    writeVecVec(out, vocab_.word_paths_);

    out.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void FastTextContext::loadModel(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open for reading: " + filename);

    in.read(reinterpret_cast<char*>(&d_w_),         sizeof(d_w_));
    in.read(reinterpret_cast<char*>(&d_p_),         sizeof(d_p_));
    in.read(reinterpret_cast<char*>(&d_pr_),        sizeof(d_pr_));
    in.read(reinterpret_cast<char*>(&d_out_),       sizeof(d_out_));
    in.read(reinterpret_cast<char*>(&min_n_),       sizeof(min_n_));
    in.read(reinterpret_cast<char*>(&max_n_),       sizeof(max_n_));
    in.read(reinterpret_cast<char*>(&threshold_),   sizeof(threshold_));
    in.read(reinterpret_cast<char*>(&window_size_), sizeof(window_size_));

    int vs, np, npr, ng, os;
    in.read(reinterpret_cast<char*>(&vs),  sizeof(vs));
    in.read(reinterpret_cast<char*>(&np),  sizeof(np));
    in.read(reinterpret_cast<char*>(&npr), sizeof(npr));
    in.read(reinterpret_cast<char*>(&ng),  sizeof(ng));
    in.read(reinterpret_cast<char*>(&os),  sizeof(os));

    int concat_dim = d_w_ + d_p_ + d_pr_;
    output_matrix_.resize(os,            d_out_);
    ngram_matrix_.resize(ng,             d_w_);
    input_matrix_.resize(vs,             d_w_);
    W_proj_.resize(d_out_,              concat_dim);
    patient_matrix_.resize(np,           d_p_);
    provider_matrix_.resize(npr,         d_pr_);

    for (int i = 0; i < vs; ++i) {
        uint32_t len; in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string word(len, '\0'); in.read(&word[0], len);
        int idx;      in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        vocab_.addWord(idx, word);
    }
    for (int i = 0; i < np; ++i) {
        uint32_t len; in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string field(len, '\0'); in.read(&field[0], len);
        int idx;      in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        vocab_.addPatient(idx, field);
    }
    for (int i = 0; i < npr; ++i) {
        uint32_t len; in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string field(len, '\0'); in.read(&field[0], len);
        int idx;      in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        vocab_.addProvider(idx, field);
    }

    output_matrix_.load(in);
    ngram_matrix_.load(in);
    input_matrix_.load(in);
    W_proj_.load(in);
    patient_matrix_.load(in);
    provider_matrix_.load(in);

    readVecVec(in, vocab_.word_codes_, vs);
    readVecVec(in, vocab_.word_paths_, vs);

    in.close();
    makeInference();
    precomputeWordVectors();
    std::cout << "Model loaded from " << filename
              << " (d_w=" << d_w_ << " d_p=" << d_p_ << " d_pr=" << d_pr_
              << " d_out=" << d_out_ << " vocab=" << vs
              << " patient=" << np << " provider=" << npr << ")" << std::endl;
}

// Returns d_w-dimensional raw word vector (word_emb + ngrams).
// The cached_word_vectors_ holds projected d_out vectors for NN search — not usable here.
// Always delegates to Inference::getWordVector for the correct unprojected representation.
std::vector<float> FastTextContext::getWordVector(const std::string& word) {
    if (!inference_) throw std::runtime_error("Model not initialised.");
    return inference_->getWordVector(word);
}

std::vector<float> FastTextContext::getCombinedVector(
    const std::vector<std::string>& words,
    const std::vector<std::string>& patient_meta,
    const std::vector<std::string>& provider_meta) {
    if (!inference_) throw std::runtime_error("Model not initialised.");
    return inference_->getCombinedVector(words, patient_meta, provider_meta);
}

std::vector<std::pair<std::string, float>> FastTextContext::getNearestNeighbors(
    const std::vector<std::string>& words,
    const std::vector<std::string>& patient_meta,
    const std::vector<std::string>& provider_meta,
    int k) {
    if (!inference_) throw std::runtime_error("Model not initialised.");
    return inference_->getNearestNeighbors(words, patient_meta, provider_meta, k,
                                           cached_word_vectors_);
}

} // namespace fasttext

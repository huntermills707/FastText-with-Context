#include "trainer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <limits>

namespace fasttext {

Trainer::Trainer(int dim, int epoch, float lr, int min_n, int max_n,
                 int chunk_size, int ngram_buckets, int window_size, float grad_clip)
    : dim_(dim), epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n),
      chunk_size_(chunk_size), ngram_buckets_(ngram_buckets), window_size_(window_size),
      grad_clip_(grad_clip) {
    int T = omp_get_max_threads();
    rngs_.resize(T);
    for (int i = 0; i < T; ++i) rngs_[i].seed(std::random_device{}() + i);

    // Pre-allocate per-thread scratch buffers so the hot loop never hits the
    // allocator.  Each thread exclusively uses its own slot.
    thread_bufs_.resize(T);
    for (auto& buf : thread_bufs_) buf.resize(dim);
}

// Scales v down so ||v||_2 <= max_norm. Direction is preserved.
// A zero vector or max_norm <= 0 passes through unchanged.
void Trainer::clipNorm(std::vector<float>& v, float max_norm) {
    if (max_norm <= 0.0f) return;
    float norm_sq = 0.0f;
    for (float x : v) norm_sq += x * x;
    if (norm_sq <= max_norm * max_norm) return;
    float scale = max_norm / std::sqrt(norm_sq);
    for (float& x : v) x *= scale;
}

uint64_t Trainer::hash(const std::string& str) const {
    uint64_t h = 14695981039346656037ULL;
    for (char c : str) { h ^= static_cast<uint64_t>(c); h *= 1099511628211ULL; }
    return h;
}

void Trainer::getNgramIndices(const std::string& word, std::vector<int>& out) const {
    out.clear();
    std::string bordered = "<" + word + ">";
    for (int n = min_n_; n <= max_n_; ++n) {
        for (size_t i = 0; i + n <= bordered.size(); ++i) {
            out.push_back(static_cast<int>(hash(bordered.substr(i, n)) % ngram_buckets_));
        }
    }
}

bool Trainer::parseLine(const std::string& line, StreamingSample& sample) const {
    sample.metadata_fields.clear();
    sample.words.clear();
    if (line.empty()) return false;

    std::stringstream ss(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(ss, field, '|')) fields.push_back(field);
    if (fields.empty()) return false;

    for (size_t i = 0; i + 1 < fields.size(); ++i)
        sample.metadata_fields.push_back(fields[i]);

    std::istringstream sent(fields.back());
    std::string word;
    while (sent >> word) sample.words.push_back(word);
    return !sample.words.empty();
}

int Trainer::countLines(const std::string& filename) const {
    std::ifstream f(filename);
    int n = 0;
    std::string line;
    while (std::getline(f, line)) if (!line.empty()) ++n;
    return n;
}

bool Trainer::checkMatrixHealth(const Matrix& m, const std::string& name, int ep) const {
    int nan_c = 0, inf_c = 0;
    int64_t total = m.rows() * m.cols();
    if (total == 0) return true;

    #pragma omp parallel for reduction(+:nan_c, inf_c)
    for (int64_t i = 0; i < m.rows(); ++i) {
        const float* row = m.row(i);
        for (int64_t j = 0; j < m.cols(); ++j) {
            if (std::isnan(row[j]))      ++nan_c;
            else if (std::isinf(row[j])) ++inf_c;
        }
    }

    if (nan_c > 0 || inf_c > 0) {
        std::cerr << "MATRIX HEALTH [epoch " << ep << "] " << name
                  << ": NaN=" << nan_c << " Inf=" << inf_c
                  << " / " << total << std::endl;
        return false;
    }
    return true;
}

std::pair<std::vector<float>, std::vector<int>>
Trainer::gatherMetadataVec(const StreamingSample& sample, const Vocabulary& vocab,
                           const Matrix& metadata_matrix) const {
    std::vector<float> vec(dim_, 0.0f);
    std::vector<int>   active;

    for (const auto& meta : sample.metadata_fields) {
        int idx = vocab.getMetadataIdx(meta);
        if (idx < 0) continue;
        const float* row = metadata_matrix.row(idx);
        for (int j = 0; j < dim_; ++j) vec[j] += row[j];
        active.push_back(idx);
    }
    return {vec, active};
}

// Fills pre-allocated out_vec and out_ngram instead of returning newly
// allocated containers.  This is called once per center word inside the
// innermost training loop, so avoiding heap traffic matters.
void Trainer::buildCenterVec(int word_idx, const std::string& word,
                             const Matrix& input_matrix, const Matrix& ngram_matrix,
                             const std::vector<float>& meta_vec,
                             std::vector<float>& out_vec,
                             std::vector<int>& out_ngram) const {
    // Zero-fill the output vector (size is guaranteed by caller).
    std::fill(out_vec.begin(), out_vec.begin() + dim_, 0.0f);

    // Word-level embedding.
    const float* wr = input_matrix.row(word_idx);
    for (int j = 0; j < dim_; ++j) out_vec[j] += wr[j];

    // N-gram embeddings.
    getNgramIndices(word, out_ngram);
    for (int idx : out_ngram) {
        const float* nr = ngram_matrix.row(idx);
        for (int j = 0; j < dim_; ++j) out_vec[j] += nr[j];
    }

    // Metadata (pre-computed, shared across all center words in the sample).
    for (int j = 0; j < dim_; ++j) out_vec[j] += meta_vec[j];
}

// Follows the word2vec update order:
//   1. Dot product and sigmoid using pre-update output row.
//   2. Compute gradient g = lr * (target - sigmoid).
//   3. Accumulate input gradient from pre-update output row.
//   4. Update output row (Hogwild: direct, intentionally racy write).
//
// scratch is a pre-allocated dim_-sized buffer used for the output update
// vector, replacing the per-call allocation in the previous version.
float Trainer::hsStep(int node_idx, int direction,
                      const std::vector<float>& center_vec,
                      Matrix& output_matrix,
                      std::vector<float>& center_grad,
                      std::vector<float>& scratch,
                      float lr) {
    const float* out_row = output_matrix.row(node_idx);

    float dot = 0.0f;
    for (int j = 0; j < dim_; ++j) dot += out_row[j] * center_vec[j];
    dot = std::max(-20.0f, std::min(20.0f, dot));

    const float sig    = 1.0f / (1.0f + std::exp(-dot));
    const float target = (direction == 0) ? 1.0f : 0.0f;
    const float g      = lr * (target - sig);

    // Input gradient uses pre-update output row (word2vec ordering).
    for (int j = 0; j < dim_; ++j) center_grad[j] += g * out_row[j];

    // Build the output update vector in the scratch buffer, clip its norm,
    // then apply (Hogwild).
    for (int j = 0; j < dim_; ++j) scratch[j] = g * center_vec[j];
    clipNorm(scratch, grad_clip_);

    float* mut_row = output_matrix.row(node_idx);
    for (int j = 0; j < dim_; ++j) mut_row[j] += scratch[j];

    const float p = std::max(1e-7f, std::min(1.0f - 1e-7f, sig));
    return -(target * std::log(p) + (1.0f - target) * std::log(1.0f - p));
}

// Hogwild: direct writes to shared matrices without locks.
void Trainer::distributeGrad(const std::vector<float>& cg,
                             int word_idx, const std::vector<int>& ngram_indices,
                             const std::vector<int>& active_meta,
                             Matrix& input_matrix, Matrix& ngram_matrix,
                             Matrix& metadata_matrix) {
    float* wr = input_matrix.row(word_idx);
    for (int j = 0; j < dim_; ++j) wr[j] += cg[j];

    for (int idx : ngram_indices) {
        float* nr = ngram_matrix.row(idx);
        for (int j = 0; j < dim_; ++j) nr[j] += cg[j];
    }

    for (int idx : active_meta) {
        float* mr = metadata_matrix.row(idx);
        for (int j = 0; j < dim_; ++j) mr[j] += cg[j];
    }
}

void Trainer::processSample(const StreamingSample& sample, const Vocabulary& vocab,
                            Matrix& input_matrix, Matrix& output_matrix,
                            Matrix& ngram_matrix, Matrix& metadata_matrix,
                            float lr, int tid, double& loss_acc, int& pred_count) {
    auto [meta_vec, active_meta] = gatherMetadataVec(sample, vocab, metadata_matrix);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    // Grab this thread's pre-allocated scratch buffers.
    auto& buf = thread_bufs_[tid];

    for (int cp = 0; cp < static_cast<int>(sample.words.size()); ++cp) {
        const int cw_idx = vocab.getWordIdx(sample.words[cp]);
        if (cw_idx < 0) continue;

        // Subsample center word.
        if (!vocab.discard_probs_.empty() && vocab.discard_probs_[cw_idx] > 0.0f)
            if (ud(rngs_[tid]) < vocab.discard_probs_[cw_idx]) continue;

        // Build center vector into pre-allocated buffers.
        buildCenterVec(cw_idx, sample.words[cp], input_matrix, ngram_matrix,
                       meta_vec, buf.center_vec, buf.ngram_indices);

        // Zero the gradient accumulator (reuse pre-allocated buffer).
        std::fill(buf.center_grad.begin(), buf.center_grad.begin() + dim_, 0.0f);

        // Sample window size uniformly from [1, window_size_] per center word.
        const int win   = 1 + static_cast<int>(rngs_[tid]() % window_size_);
        const int wstart = std::max(0, cp - win);
        const int wend   = std::min(static_cast<int>(sample.words.size()), cp + win + 1);

        for (int ctx = wstart; ctx < wend; ++ctx) {
            if (ctx == cp) continue;
            const int ctx_idx = vocab.getWordIdx(sample.words[ctx]);
            if (ctx_idx < 0) continue;

            // Subsample context word.
            if (!vocab.discard_probs_.empty() && vocab.discard_probs_[ctx_idx] > 0.0f)
                if (ud(rngs_[tid]) < vocab.discard_probs_[ctx_idx]) continue;

            const auto& path = vocab.word_paths_[ctx_idx];
            const auto& code = vocab.word_codes_[ctx_idx];

            for (size_t i = 0; i < path.size(); ++i) {
                float loss = hsStep(path[i], code[i], buf.center_vec,
                                    output_matrix, buf.center_grad,
                                    buf.out_update, lr);
                #pragma omp atomic
                loss_acc += loss;
                #pragma omp atomic
                pred_count++;
            }
        }

        // Clip the accumulated input gradient before distributing.
        clipNorm(buf.center_grad, grad_clip_);

        distributeGrad(buf.center_grad, cw_idx, buf.ngram_indices, active_meta,
                       input_matrix, ngram_matrix, metadata_matrix);
    }
}

void Trainer::processChunk(const std::vector<StreamingSample>& chunk,
                           const Vocabulary& vocab,
                           Matrix& input_matrix, Matrix& output_matrix,
                           Matrix& ngram_matrix, Matrix& metadata_matrix,
                           float lr, double& loss_acc, int& pred_count) {
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < static_cast<int>(chunk.size()); ++s) {
        processSample(chunk[s], vocab, input_matrix, output_matrix,
                      ngram_matrix, metadata_matrix, lr,
                      omp_get_thread_num(), loss_acc, pred_count);
    }
}

void Trainer::runEpoch(int ep, int total_samples, long long& global_step,
                       long long total_steps, const std::string& filename,
                       Vocabulary& vocab,
                       Matrix& input_matrix, Matrix& output_matrix,
                       Matrix& ngram_matrix, Matrix& metadata_matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);

    const auto epoch_start = std::chrono::steady_clock::now();

    double epoch_loss  = 0.0;
    int    epoch_preds = 0, processed = 0, last_reported = 0;
    const float LR_FLOOR = lr_ * 0.0001f;
    const int   BAR_WIDTH = 40;

    std::vector<StreamingSample> chunk;
    chunk.reserve(chunk_size_);
    StreamingSample sample;
    std::string line;

    auto flushChunk = [&]() {
        float lr = std::max(LR_FLOOR, lr_ * (1.0f - static_cast<float>(global_step) / total_steps));
        processChunk(chunk, vocab, input_matrix, output_matrix, ngram_matrix, metadata_matrix,
                     lr, epoch_loss, epoch_preds);
        global_step += static_cast<long long>(chunk.size());
        processed   += static_cast<int>(chunk.size());
        chunk.clear();
    };

    while (std::getline(file, line)) {
        if (parseLine(line, sample)) chunk.push_back(std::move(sample));
        if (static_cast<int>(chunk.size()) < chunk_size_) continue;
        flushChunk();

        if (processed - last_reported >= 1000) {
            float  prog   = static_cast<float>(processed) / total_samples;
            int    filled = static_cast<int>(BAR_WIDTH * prog);
            double avg    = epoch_preds > 0 ? epoch_loss / epoch_preds : 0.0;
            float  lr     = std::max(LR_FLOOR, lr_ * (1.0f - static_cast<float>(global_step) / total_steps));
            std::cout << "\rEpoch " << (ep + 1) << "/" << epoch_ << " | ["
                      << std::string(filled, '#') << std::string(BAR_WIDTH - filled, ' ') << "] "
                      << std::fixed << std::setprecision(2) << (prog * 100) << "%"
                      << " | LR: " << std::scientific << std::setprecision(2) << lr
                      << " | Loss: " << std::fixed << std::setprecision(4) << avg << std::flush;
            last_reported = processed;
        }
    }

    if (!chunk.empty()) flushChunk();

    const auto epoch_end = std::chrono::steady_clock::now();
    const double elapsed  = std::chrono::duration<double>(epoch_end - epoch_start).count();
    const double avg_loss = epoch_preds > 0 ? epoch_loss / epoch_preds : 0.0;
    const float  final_lr = std::max(LR_FLOOR, lr_ * (1.0f - static_cast<float>(global_step) / total_steps));

    // Complete the progress bar on its own line, then print a dedicated summary.
    std::cout << "\rEpoch " << (ep + 1) << "/" << epoch_ << " | ["
              << std::string(BAR_WIDTH, '#') << "] 100.00%"
              << " | LR: " << std::scientific << std::setprecision(2) << final_lr
              << std::endl;

    std::cout << "  -> Avg loss: " << std::fixed << std::setprecision(6) << avg_loss
              << "  |  Predictions: " << epoch_preds
              << "  |  Time: " << std::fixed << std::setprecision(1) << elapsed << "s"
              << std::endl;
}

void Trainer::train(const std::string& filename, Vocabulary& vocab,
                    Matrix& input_matrix, Matrix& output_matrix,
                    Matrix& ngram_matrix, Matrix& metadata_matrix) {
    const int       total_samples = countLines(filename);
    const long long total_steps   = static_cast<long long>(total_samples) * epoch_;
    long long       global_step   = 0;

    std::cout << "Samples: " << total_samples << " | Steps: " << total_steps
              << " | Threads: " << omp_get_max_threads()
              << " | Window: [1," << window_size_ << "] (sampled)\n"
              << "Subsampling: " << (!vocab.discard_probs_.empty() ? "enabled" : "disabled")
              << " | SGD: Hogwild (lock-free)"
              << " | Grad clip: " << (grad_clip_ > 0.0f ? std::to_string(grad_clip_) : "off")
              << "\n";

    for (int ep = 0; ep < epoch_; ++ep) {
        runEpoch(ep, total_samples, global_step, total_steps, filename, vocab,
                 input_matrix, output_matrix, ngram_matrix, metadata_matrix);

        bool healthy = checkMatrixHealth(output_matrix,  "output",   ep + 1)
                    && checkMatrixHealth(ngram_matrix,    "ngram",    ep + 1)
                    && checkMatrixHealth(metadata_matrix, "metadata", ep + 1)
                    && checkMatrixHealth(input_matrix,    "input",    ep + 1);
        if (!healthy) throw std::runtime_error("Training aborted: matrix corruption.");
    }

    std::cout << "\nTraining complete." << std::endl;
}

} // namespace fasttext

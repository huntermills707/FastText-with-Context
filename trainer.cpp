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
#include <cstring>

namespace fasttext {

Trainer::Trainer(int d_word, int d_patient, int d_encounter, int d_out,
                 int epoch, float lr, int min_n, int max_n,
                 int chunk_size, int ngram_buckets, int window_size,
                 float grad_clip, float weight_decay)
    : d_word_(d_word), d_patient_(d_patient), d_encounter_(d_encounter), d_out_(d_out),
      concat_dim_(d_word + d_patient + d_encounter),
      epoch_(epoch), lr_(lr), min_n_(min_n), max_n_(max_n),
      chunk_size_(chunk_size), ngram_buckets_(ngram_buckets),
      window_size_(window_size), grad_clip_(grad_clip), weight_decay_(weight_decay) {
    int T = omp_get_max_threads();
    rngs_.resize(T);
    for (int i = 0; i < T; ++i) rngs_[i].seed(std::random_device{}() + i);

    thread_bufs_.resize(T);
    for (auto& buf : thread_bufs_) buf.resize(d_word, d_patient, d_encounter, d_out);
}

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

bool Trainer::parseGroupedLine(const std::string& line, GroupedSample& sample) const {
    sample.patient_group.clear();
    sample.encounter_group.clear();
    sample.words.clear();

    if (line.empty()) return false;

    // Split on " ||| " (space-pipe-pipe-pipe-space).
    // Expect exactly 3 groups: patient_group, encounter_group, words.
    std::vector<std::string> groups;
    size_t start = 0;
    const std::string delim = " ||| ";
    while (true) {
        size_t pos = line.find(delim, start);
        if (pos == std::string::npos) {
            groups.push_back(line.substr(start));
            break;
        }
        groups.push_back(line.substr(start, pos - start));
        start = pos + delim.size();
    }

    if (groups.size() != 3) return false;

    std::string token;

    // Patient group (index 0).
    std::istringstream ps(groups[0]);
    while (ps >> token) sample.patient_group.push_back(token);

    // Encounter group (index 1).
    std::istringstream prs(groups[1]);
    while (prs >> token) sample.encounter_group.push_back(token);

    // Words group (index 2, always last).
    std::istringstream ws(groups[2]);
    while (ws >> token) sample.words.push_back(token);

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

void Trainer::broadcastDenseParams(const Matrix& W_proj,
                                   const Matrix& patient_matrix,
                                   const Matrix& encounter_matrix) {
    int T = omp_get_max_threads();
    for (int t = 0; t < T; ++t) {
        std::memcpy(W_proj_local_[t].data(), W_proj.data(),
                    static_cast<size_t>(W_proj.size()) * sizeof(float));
        std::memcpy(patient_local_[t].data(), patient_matrix.data(),
                    static_cast<size_t>(patient_matrix.size()) * sizeof(float));
        std::memcpy(encounter_local_[t].data(), encounter_matrix.data(),
                    static_cast<size_t>(encounter_matrix.size()) * sizeof(float));
    }
}

void Trainer::reduceDenseParams(Matrix& W_proj,
                                Matrix& patient_matrix,
                                Matrix& encounter_matrix) {
    int T = omp_get_max_threads();
    float inv_T = 1.0f / static_cast<float>(T);

    auto average = [&](Matrix& shared, std::vector<Matrix>& locals) {
        for (int64_t i = 0; i < shared.size(); ++i) {
            float sum = 0.0f;
            for (int t = 0; t < T; ++t)
                sum += locals[t].data()[i];
            shared.data()[i] = sum * inv_T;
        }
    };

    average(W_proj, W_proj_local_);
    average(patient_matrix, patient_local_);
    average(encounter_matrix, encounter_local_);

    // L2 weight decay on W_proj only (not on embedding matrices).
    // Applied after each chunk reduce: W_proj *= (1 - weight_decay).
    // Prevents the projection from memorizing low-cardinality metadata patterns.
    if (weight_decay_ > 0.0f) {
        const float scale = 1.0f - weight_decay_;
        for (int64_t i = 0; i < W_proj.size(); ++i)
            W_proj.data()[i] *= scale;
    }
}

void Trainer::buildConcatVec(int word_idx, const std::string& word,
                              const GroupedSample& sample,
                              const Vocabulary& vocab,
                              const Matrix& input_matrix,
                              const Matrix& ngram_matrix,
                              const Matrix& patient_matrix,
                              const Matrix& encounter_matrix,
                              std::vector<float>& concat_out,
                              std::vector<int>& ngram_out) const {
    std::fill(concat_out.begin(), concat_out.end(), 0.0f);

    // Word part: concat_out[0..d_word_).
    float* wp = concat_out.data();
    const float* wr = input_matrix.row(word_idx);
    for (int j = 0; j < d_word_; ++j) wp[j] += wr[j];

    getNgramIndices(word, ngram_out);
    for (int idx : ngram_out) {
        const float* nr = ngram_matrix.row(idx);
        for (int j = 0; j < d_word_; ++j) wp[j] += nr[j];
    }

    // Patient part: concat_out[d_word_..d_word_+d_patient_).
    float* pp = concat_out.data() + d_word_;
    int n_patient = 0;
    for (const auto& field : sample.patient_group) {
        int idx = vocab.getPatientIdx(field);
        if (idx < 0) continue;
        const float* pr = patient_matrix.row(idx);
        for (int j = 0; j < d_patient_; ++j) pp[j] += pr[j];
        ++n_patient;
    }
    if (n_patient > 1) {
        float inv = 1.0f / static_cast<float>(n_patient);
        for (int j = 0; j < d_patient_; ++j) pp[j] *= inv;
    }

    // Encounter part: concat_out[d_word_+d_patient_..concat_dim).
    float* encp = concat_out.data() + d_word_ + d_patient_;
    int n_encounter = 0;
    for (const auto& field : sample.encounter_group) {
        int idx = vocab.getEncounterIdx(field);
        if (idx < 0) continue;
        const float* pr = encounter_matrix.row(idx);
        for (int j = 0; j < d_encounter_; ++j) encp[j] += pr[j];
        ++n_encounter;
    }
    if (n_encounter > 1) {
        float inv = 1.0f / static_cast<float>(n_encounter);
        for (int j = 0; j < d_encounter_; ++j) encp[j] *= inv;
    }
}

float Trainer::hsStep(int node_idx, int direction,
                      const std::vector<float>& center_vec,
                      Matrix& output_matrix,
                      std::vector<float>& center_grad,
                      std::vector<float>& scratch,
                      float lr) {
    const float* out_row = output_matrix.row(node_idx);

    float dot = 0.0f;
    for (int j = 0; j < d_out_; ++j) dot += out_row[j] * center_vec[j];
    dot = std::max(-20.0f, std::min(20.0f, dot));

    const float sig    = 1.0f / (1.0f + std::exp(-dot));
    const float target = (direction == 0) ? 1.0f : 0.0f;
    const float g      = lr * (target - sig);

    for (int j = 0; j < d_out_; ++j) center_grad[j] += g * out_row[j];

    // Clip output node update before applying (Hogwild).
    for (int j = 0; j < d_out_; ++j) scratch[j] = g * center_vec[j];
    clipNorm(scratch, grad_clip_);

    float* mut_row = output_matrix.row(node_idx);
    for (int j = 0; j < d_out_; ++j) mut_row[j] += scratch[j];

    const float p = std::max(1e-7f, std::min(1.0f - 1e-7f, sig));
    return -(target * std::log(p) + (1.0f - target) * std::log(1.0f - p));
}

// Backward pass through the linear projection.
//
// Given:
//   center_vec = W_proj * concat_vec   (forward pass)
//   center_grad ∈ R^d_out              (accumulated from HS, already clipped)
//
// Computes:
//   concat_grad = W_proj^T * center_grad        [step 6a]
//   W_proj     += lr * center_grad * concat^T   [step 6b, thread-local]
//
// Then distributes concat_grad slices:
//   word_grad     = concat_grad[0..d_word_)         → input_matrix, ngram_matrix (Hogwild)
//   patient_grad  = concat_grad[d_word_..d_word_+d_patient_)    → patient_matrix (thread-local)
//   encounter_grad = concat_grad[d_word_+d_patient_..end)     → encounter_matrix (thread-local)
void Trainer::distributeGrad(const std::vector<float>& concat_grad,
                              const std::vector<float>& center_grad,
                              const std::vector<float>& concat_vec,
                              int word_idx,
                              const std::vector<int>& ngram_indices,
                              const GroupedSample& sample,
                              const Vocabulary& vocab,
                              Matrix& input_matrix,
                              Matrix& ngram_matrix,
                              Matrix& W_proj,
                              Matrix& patient_matrix,
                              Matrix& encounter_matrix,
                              float lr) {
    // Update W_proj (thread-local): W_proj += lr * center_grad * concat^T.
    W_proj.addOuterProduct(center_grad.data(), concat_vec.data(), lr);

    // Word part [0..d_word_): Hogwild writes to shared input/ngram matrices.
    const float* word_grad = concat_grad.data();

    float* wr = input_matrix.row(word_idx);
    for (int j = 0; j < d_word_; ++j)
        wr[j] += lr * word_grad[j];

    for (int idx : ngram_indices) {
        float* nr = ngram_matrix.row(idx);
        for (int j = 0; j < d_word_; ++j)
            nr[j] += lr * word_grad[j];
    }

    // Patient part [d_word_..d_word_+d_patient_): write to thread-local patient_matrix.
    const float* patient_grad = concat_grad.data() + d_word_;
    int n_patient = 0;
    for (const auto& field : sample.patient_group)
        if (vocab.getPatientIdx(field) >= 0) ++n_patient;

    if (n_patient > 0) {
        float patient_scale = lr / static_cast<float>(n_patient);
        for (const auto& field : sample.patient_group) {
            int idx = vocab.getPatientIdx(field);
            if (idx < 0) continue;
            float* pr = patient_matrix.row(idx);
            for (int j = 0; j < d_patient_; ++j)
                pr[j] += patient_scale * patient_grad[j];
        }
    }

    // Encounter part [d_word_+d_patient_..concat_dim): write to thread-local encounter_matrix.
    const float* encounter_grad = concat_grad.data() + d_word_ + d_patient_;
    int n_encounter = 0;
    for (const auto& field : sample.encounter_group)
        if (vocab.getEncounterIdx(field) >= 0) ++n_encounter;

    if (n_encounter > 0) {
        float encounter_scale = lr / static_cast<float>(n_encounter);
        for (const auto& field : sample.encounter_group) {
            int idx = vocab.getEncounterIdx(field);
            if (idx < 0) continue;
            float* pr = encounter_matrix.row(idx);
            for (int j = 0; j < d_encounter_; ++j)
                pr[j] += encounter_scale * encounter_grad[j];
        }
    }
}

void Trainer::processSample(const GroupedSample& sample,
                            const Vocabulary& vocab,
                            Matrix& input_matrix, Matrix& output_matrix,
                            Matrix& ngram_matrix,
                            Matrix& W_proj,
                            Matrix& patient_matrix,
                            Matrix& encounter_matrix,
                            float lr, int tid,
                            double& loss_acc, long long& pred_count) {
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    auto& buf = thread_bufs_[tid];

    for (int cp = 0; cp < static_cast<int>(sample.words.size()); ++cp) {
        const int cw_idx = vocab.getWordIdx(sample.words[cp]);
        if (cw_idx < 0) continue;

        // Subsampling.
        if (!vocab.discard_probs_.empty() && vocab.discard_probs_[cw_idx] > 0.0f)
            if (ud(rngs_[tid]) < vocab.discard_probs_[cw_idx]) continue;

        // 1. Build concat_vec = [word_part ; patient_avg ; encounter_avg].
        buildConcatVec(cw_idx, sample.words[cp], sample, vocab,
                       input_matrix, ngram_matrix,
                       patient_matrix, encounter_matrix,
                       buf.concat_vec, buf.ngram_indices);

        // 2. Project to d_out: center_vec = W_proj * concat_vec.
        W_proj.mulVec(buf.concat_vec.data(), buf.center_vec.data());

        // 3. Zero center_grad accumulator.
        std::fill(buf.center_grad.begin(), buf.center_grad.end(), 0.0f);

        // 4. Skip-gram hierarchical softmax loop.
        const int win    = 1 + static_cast<int>(rngs_[tid]() % window_size_);
        const int wstart = std::max(0, cp - win);
        const int wend   = std::min(static_cast<int>(sample.words.size()), cp + win + 1);

        for (int ctx = wstart; ctx < wend; ++ctx) {
            if (ctx == cp) continue;
            const int ctx_idx = vocab.getWordIdx(sample.words[ctx]);
            if (ctx_idx < 0) continue;

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

        // 5. Clip accumulated center_grad before backpropagation through projection.
        clipNorm(buf.center_grad, grad_clip_);

        // 6a. concat_grad = W_proj^T * center_grad.
        W_proj.mulVecTranspose(buf.center_grad.data(), buf.concat_grad.data());

        // 6b + 7. Update W_proj (thread-local) and distribute concat_grad slices.
        distributeGrad(buf.concat_grad, buf.center_grad, buf.concat_vec,
                       cw_idx, buf.ngram_indices, sample, vocab,
                       input_matrix, ngram_matrix,
                       W_proj, patient_matrix, encounter_matrix,
                       lr);
    }
}

void Trainer::processChunk(const std::vector<GroupedSample>& chunk,
                           const Vocabulary& vocab,
                           Matrix& input_matrix, Matrix& output_matrix,
                           Matrix& ngram_matrix,
                           Matrix& W_proj,
                           Matrix& patient_matrix,
                           Matrix& encounter_matrix,
                           float lr, double& loss_acc, long long& pred_count) {
    // BROADCAST: shared → thread-local.
    broadcastDenseParams(W_proj, patient_matrix, encounter_matrix);

    // PROCESS: parallel over samples, each thread uses its own local copies.
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < static_cast<int>(chunk.size()); ++s) {
        int tid = omp_get_thread_num();
        processSample(chunk[s], vocab,
                      input_matrix, output_matrix, ngram_matrix,
                      W_proj_local_[tid],
                      patient_local_[tid],
                      encounter_local_[tid],
                      lr, tid, loss_acc, pred_count);
    }
    // Implicit barrier at end of #pragma omp parallel for.

    // REDUCE: average thread-local → shared dense params.
    reduceDenseParams(W_proj, patient_matrix, encounter_matrix);
}

void Trainer::runEpoch(int ep, int total_samples, long long& global_step,
                       long long total_steps, const std::string& filename,
                       Vocabulary& vocab,
                       Matrix& input_matrix, Matrix& output_matrix,
                       Matrix& ngram_matrix,
                       Matrix& W_proj, Matrix& patient_matrix, Matrix& encounter_matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);

    const auto epoch_start = std::chrono::steady_clock::now();

    double    epoch_loss  = 0.0;
    long long epoch_preds = 0;
    int       processed = 0, last_reported = 0;
    const float LR_FLOOR = lr_ * 0.0001f;
    const int   BAR_WIDTH = 40;

    std::vector<GroupedSample> chunk;
    chunk.reserve(chunk_size_);
    GroupedSample sample;
    std::string line;

    auto flushChunk = [&]() {
        float lr = std::max(LR_FLOOR, lr_ * (1.0f - static_cast<float>(global_step) / total_steps));
        processChunk(chunk, vocab, input_matrix, output_matrix, ngram_matrix,
                     W_proj, patient_matrix, encounter_matrix,
                     lr, epoch_loss, epoch_preds);
        global_step += static_cast<long long>(chunk.size());
        processed   += static_cast<int>(chunk.size());
        chunk.clear();
    };

    while (std::getline(file, line)) {
        if (parseGroupedLine(line, sample)) chunk.push_back(std::move(sample));
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
                    Matrix& ngram_matrix,
                    Matrix& W_proj, Matrix& patient_matrix, Matrix& encounter_matrix) {
    const int       total_samples = countLines(filename);
    const long long total_steps   = static_cast<long long>(total_samples) * epoch_;
    long long       global_step   = 0;

    int T = omp_get_max_threads();

    // Initialize thread-local dense parameter copies.
    W_proj_local_.resize(T);
    patient_local_.resize(T);
    encounter_local_.resize(T);
    for (int t = 0; t < T; ++t) {
        W_proj_local_[t].resize(W_proj.rows(), W_proj.cols());
        patient_local_[t].resize(patient_matrix.rows(), patient_matrix.cols());
        encounter_local_[t].resize(encounter_matrix.rows(), encounter_matrix.cols());
    }

    std::cout << "Samples: " << total_samples << " | Steps: " << total_steps
              << " | Threads: " << T
              << " | Window: [1," << window_size_ << "] (sampled)\n"
              << "Subsampling: " << (!vocab.discard_probs_.empty() ? "enabled" : "disabled")
              << " | Sparse updates: Hogwild (input, ngram, output)"
              << " | Dense sync: chunk-averaged (W_proj, patient, encounter)"
              << " | Chunk size: " << chunk_size_
              << " | Grad clip: " << (grad_clip_ > 0.0f ? std::to_string(grad_clip_) : "off") << " | W_proj decay: " << (weight_decay_ > 0.0f ? std::to_string(weight_decay_) : "off")
              << "\nDimensions: d_word=" << d_word_ << " d_patient=" << d_patient_ << " d_encounter=" << d_encounter_
              << " d_out=" << d_out_ << " concat=" << concat_dim_
              << " W_proj=" << W_proj.rows() << "x" << W_proj.cols()
              << "\n";

    for (int ep = 0; ep < epoch_; ++ep) {
        runEpoch(ep, total_samples, global_step, total_steps, filename, vocab,
                 input_matrix, output_matrix, ngram_matrix,
                 W_proj, patient_matrix, encounter_matrix);

        bool healthy = checkMatrixHealth(output_matrix,   "output",   ep + 1)
                    && checkMatrixHealth(ngram_matrix,     "ngram",    ep + 1)
                    && checkMatrixHealth(patient_matrix,   "patient",  ep + 1)
                    && checkMatrixHealth(encounter_matrix, "encounter", ep + 1)
                    && checkMatrixHealth(input_matrix,     "input",    ep + 1)
                    && checkMatrixHealth(W_proj,           "W_proj",   ep + 1);
        if (!healthy) throw std::runtime_error("Training aborted: matrix corruption.");
    }

    std::cout << "\nTraining complete." << std::endl;
}

} // namespace fasttext

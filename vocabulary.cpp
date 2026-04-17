#include "vocabulary.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace fasttext {

void Vocabulary::buildFromCounts(const std::unordered_map<std::string, int>& word_freq,
                                 const std::unordered_map<std::string, int>& patient_freq,
                                 const std::unordered_map<std::string, int>& encounter_freq) {
    // Words: sort by frequency descending, apply threshold.
    int word_idx = 0, filtered = 0;
    std::vector<std::pair<std::string, int>> sorted_words(word_freq.begin(), word_freq.end());
    std::sort(sorted_words.begin(), sorted_words.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& [word, count] : sorted_words) {
        if (count >= threshold_) {
            word2idx_[word] = word_idx;
            idx2word_.push_back(word);
            word_freqs_.push_back(static_cast<double>(count));
            ++word_idx;
        } else {
            ++filtered;
        }
    }

    // Patient group fields: all kept, sorted by frequency.
    int pat_idx = 0;
    std::vector<std::pair<std::string, int>> sorted_patient(patient_freq.begin(), patient_freq.end());
    std::sort(sorted_patient.begin(), sorted_patient.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& [field, count] : sorted_patient) {
        patient2idx_[field] = pat_idx;
        idx2patient_.push_back(field);
        ++pat_idx;
    }

    // Encounter group fields: all kept, sorted by frequency.
    int enc_idx = 0;
    std::vector<std::pair<std::string, int>> sorted_encounter(encounter_freq.begin(), encounter_freq.end());
    std::sort(sorted_encounter.begin(), sorted_encounter.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& [field, count] : sorted_encounter) {
        encounter2idx_[field] = enc_idx;
        idx2encounter_.push_back(field);
        ++enc_idx;
    }

    std::cout << "\n=== VOCABULARY BUILD SUMMARY ===\n"
              << "Unique words in file:        " << word_freq.size()   << "\n"
              << "Words meeting threshold(" << threshold_ << "): " << word_idx  << "\n"
              << "Words filtered out:          " << filtered            << "\n"
              << "Patient group fields:        " << pat_idx             << "\n"
              << "Encounter group fields:      " << enc_idx            << "\n"
              << "================================\n" << std::endl;
}

void Vocabulary::buildHuffmanTree() {
    const int V = static_cast<int>(word2idx_.size());
    std::cout << "\n=== HUFFMAN TREE CONSTRUCTION ===\n"
              << "Vocabulary size: " << V << std::endl;

    if (V == 0) { std::cerr << "ERROR: empty vocabulary.\n"; return; }

    if (V == 1) {
        word_codes_ = {{}};
        word_paths_ = {{}};
        std::cout << "Single-word vocabulary; codes/paths empty.\n"
                  << "=================================\n" << std::endl;
        return;
    }

    std::vector<int64_t> count(2 * V);
    std::vector<int>     parent(2 * V, -1);
    std::vector<uint8_t> binary(2 * V, 0);

    for (int i = 0; i < V; ++i) count[i] = static_cast<int64_t>(word_freqs_[i]);
    for (int i = V; i < 2 * V; ++i) count[i] = std::numeric_limits<int64_t>::max() / 2;

    int pos1 = V - 1, pos2 = V;

    auto pickMin = [&](int& pl, int& pi) -> int {
        if (pl >= 0 && count[pl] <= count[pi]) return pl--;
        return pi++;
    };

    for (int i = V; i < 2 * V - 1; ++i) {
        int min1 = pickMin(pos1, pos2);
        int min2 = pickMin(pos1, pos2);
        count[i]     = count[min1] + count[min2];
        parent[min1] = i;
        parent[min2] = i;
        binary[min2] = 1;
    }

    word_codes_.resize(V);
    word_paths_.resize(V);

    for (int w = 0; w < V; ++w) {
        std::vector<int> code_rev, path_rev;
        for (int n = w; parent[n] != -1; n = parent[n]) {
            code_rev.push_back(binary[n]);
            path_rev.push_back(parent[n] - V);
        }
        std::reverse(code_rev.begin(), code_rev.end());
        std::reverse(path_rev.begin(), path_rev.end());
        word_codes_[w] = std::move(code_rev);
        word_paths_[w] = std::move(path_rev);
    }

    std::cout << "Internal nodes: " << (V - 1) << std::endl;
    diagnose();
    std::cout << "=================================\n" << std::endl;
}

void Vocabulary::computeDiscardProbs(float t) {
    const int V = static_cast<int>(word_freqs_.size());
    discard_probs_.resize(V, 0.0f);

    double total = std::accumulate(word_freqs_.begin(), word_freqs_.end(), 0.0);
    if (total <= 0.0) return;

    for (int i = 0; i < V; ++i) {
        double f = word_freqs_[i] / total;
        double p = 1.0 - std::sqrt(static_cast<double>(t) / f);
        discard_probs_[i] = static_cast<float>(std::max(0.0, p));
    }
}

void Vocabulary::diagnose() const {
    if (word_paths_.empty()) return;

    size_t min_d = word_paths_[0].size();
    size_t max_d = 0;
    size_t tot_d = 0;

    for (const auto& p : word_paths_) {
        min_d = std::min(min_d, p.size());
        max_d = std::max(max_d, p.size());
        tot_d += p.size();
    }

    std::cout << "Path depth stats: min=" << min_d
              << " max=" << max_d
              << " avg=" << (tot_d / word_paths_.size()) << std::endl;

    if (min_d == 0) {
        int zero_count = 0;
        for (const auto& p : word_paths_) if (p.empty()) ++zero_count;
        std::cerr << "WARNING: " << zero_count << " words with depth 0.\n";
    }
}

} // namespace fasttext

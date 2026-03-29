#ifndef FASTTEXT_VOCABULARY_H
#define FASTTEXT_VOCABULARY_H

#include "types.h"
#include <unordered_map>
#include <string>
#include <vector>

namespace fasttext {

class Vocabulary {
public:
    explicit Vocabulary(int threshold = 5) : threshold_(threshold) {}

    void buildFromCounts(const std::unordered_map<std::string, int>& word_freq,
                         const std::unordered_map<std::string, int>& patient_freq,
                         const std::unordered_map<std::string, int>& provider_freq);

    void buildHuffmanTree();
    void computeDiscardProbs(float t);

    // Word vocabulary (threshold-filtered).
    inline int getWordIdx(const std::string& word) const {
        auto it = word2idx_.find(word);
        return it != word2idx_.end() ? it->second : -1;
    }
    inline const std::string& getWord(int idx) const {
        static const std::string empty;
        if (idx < 0 || idx >= static_cast<int>(idx2word_.size())) return empty;
        return idx2word_[idx];
    }
    inline int wordSize() const { return static_cast<int>(word2idx_.size()); }
    inline int huffmanNodes() const { return std::max(1, wordSize() - 1); }

    // Patient metadata vocabulary (no threshold).
    inline int getPatientIdx(const std::string& field) const {
        auto it = patient2idx_.find(field);
        return it != patient2idx_.end() ? it->second : -1;
    }
    inline const std::string& getPatient(int idx) const {
        static const std::string empty;
        if (idx < 0 || idx >= static_cast<int>(idx2patient_.size())) return empty;
        return idx2patient_[idx];
    }
    inline int patientSize() const { return static_cast<int>(patient2idx_.size()); }

    // Provider metadata vocabulary (no threshold).
    inline int getProviderIdx(const std::string& field) const {
        auto it = provider2idx_.find(field);
        return it != provider2idx_.end() ? it->second : -1;
    }
    inline const std::string& getProvider(int idx) const {
        static const std::string empty;
        if (idx < 0 || idx >= static_cast<int>(idx2provider_.size())) return empty;
        return idx2provider_[idx];
    }
    inline int providerSize() const { return static_cast<int>(provider2idx_.size()); }

    // Serialization helpers.
    void addWord(int idx, const std::string& word) {
        word2idx_[word] = idx;
        if (idx >= static_cast<int>(idx2word_.size())) idx2word_.resize(idx + 1);
        idx2word_[idx] = word;
    }
    void addPatient(int idx, const std::string& field) {
        patient2idx_[field] = idx;
        if (idx >= static_cast<int>(idx2patient_.size())) idx2patient_.resize(idx + 1);
        idx2patient_[idx] = field;
    }
    void addProvider(int idx, const std::string& field) {
        provider2idx_[field] = idx;
        if (idx >= static_cast<int>(idx2provider_.size())) idx2provider_.resize(idx + 1);
        idx2provider_[idx] = field;
    }

    std::vector<std::vector<int>> word_codes_;
    std::vector<std::vector<int>> word_paths_;
    std::vector<double>           word_freqs_;
    std::vector<float>            discard_probs_;

    void diagnose() const;

private:
    int threshold_;

    std::unordered_map<std::string, int> word2idx_;
    std::unordered_map<std::string, int> patient2idx_;
    std::unordered_map<std::string, int> provider2idx_;

    std::vector<std::string> idx2word_;
    std::vector<std::string> idx2patient_;
    std::vector<std::string> idx2provider_;
};

} // namespace fasttext

#endif

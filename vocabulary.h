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
                         const std::unordered_map<std::string, int>& metadata_freq);

    void buildHuffmanTree();
    void computeDiscardProbs(float t);

    inline int getWordIdx(const std::string& word) const {
        auto it = word2idx_.find(word);
        return it != word2idx_.end() ? it->second : -1;
    }

    inline int getMetadataIdx(const std::string& meta) const {
        auto it = metadata2idx_.find(meta);
        return it != metadata2idx_.end() ? it->second : -1;
    }

    inline const std::string& getWord(int idx) const {
        static const std::string empty;
        if (idx < 0 || idx >= static_cast<int>(idx2word_.size())) return empty;
        return idx2word_[idx];
    }

    inline const std::string& getMetadata(int idx) const {
        static const std::string empty;
        if (idx < 0 || idx >= static_cast<int>(idx2metadata_.size())) return empty;
        return idx2metadata_[idx];
    }

    inline int wordSize()     const { return static_cast<int>(word2idx_.size()); }
    inline int metadataSize() const { return static_cast<int>(metadata2idx_.size()); }
    inline int huffmanNodes() const { return std::max(1, wordSize() - 1); }

    void addWord(int idx, const std::string& word) {
        word2idx_[word] = idx;
        if (idx >= static_cast<int>(idx2word_.size())) idx2word_.resize(idx + 1);
        idx2word_[idx] = word;
    }

    void addMetadata(int idx, const std::string& meta) {
        metadata2idx_[meta] = idx;
        if (idx >= static_cast<int>(idx2metadata_.size())) idx2metadata_.resize(idx + 1);
        idx2metadata_[idx] = meta;
    }

    std::vector<std::vector<int>> word_codes_;
    std::vector<std::vector<int>> word_paths_;
    std::vector<double>           word_freqs_;
    std::vector<float>            discard_probs_;

    void diagnose() const;

private:
    int threshold_;

    std::unordered_map<std::string, int> word2idx_;
    std::unordered_map<std::string, int> metadata2idx_;
    std::vector<std::string>             idx2word_;
    std::vector<std::string>             idx2metadata_;
};

} // namespace fasttext

#endif

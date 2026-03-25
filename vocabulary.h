#ifndef FASTTEXT_VOCABULARY_H
#define FASTTEXT_VOCABULARY_H

#include "types.h"
#include <unordered_map>
#include <string>
#include <vector>

namespace fasttext {

class Vocabulary {
public:
    Vocabulary(int threshold = 5) : threshold_(threshold) {}
    
    // Build vocabulary from frequency maps
    void buildFromCounts(const std::unordered_map<std::string, int>& word_freq,
                         const std::unordered_map<std::string, int>& metadata_freq);
    
    // Build Huffman tree for hierarchical softmax
    void buildHuffmanTree();
    
    // Accessors
    inline int getWordIdx(const std::string& word) const {
        auto it = word2idx_.find(word);
        return (it != word2idx_.end()) ? it->second : -1;
    }
    
    inline int getMetadataIdx(const std::string& meta) const {
        auto it = metadata2idx_.find(meta);
        return (it != metadata2idx_.end()) ? it->second : -1;
    }
    
    // Reverse lookup methods
    inline const std::string& getWord(int idx) const { 
        if (idx < 0 || idx >= static_cast<int>(idx2word_.size())) {
            static const std::string empty = "";
            return empty;
        }
        return idx2word_[idx]; 
    }
    
    inline const std::string& getMetadata(int idx) const { 
        if (idx < 0 || idx >= static_cast<int>(idx2metadata_.size())) {
            static const std::string empty = "";
            return empty;
        }
        return idx2metadata_[idx]; 
    }
    
    inline int wordSize() const { return word2idx_.size(); }
    inline int metadataSize() const { return metadata2idx_.size(); }
    inline int huffmanNodes() const { return std::max(1, wordSize() - 1); }
    
    // Methods for model loading
    void addWord(int idx, const std::string& word) {
        word2idx_[word] = idx;
        if (idx >= static_cast<int>(idx2word_.size())) {
            idx2word_.resize(idx + 1);
        }
        idx2word_[idx] = word;
    }
    
    void addMetadata(int idx, const std::string& meta) {
        metadata2idx_[meta] = idx;
        if (idx >= static_cast<int>(idx2metadata_.size())) {
            idx2metadata_.resize(idx + 1);
        }
        idx2metadata_[idx] = meta;
    }
    
    // Huffman tree data (public for trainer access)
    std::vector<std::vector<int>> word_codes_;
    std::vector<std::vector<int>> word_paths_;
    std::vector<double> word_freqs_;
    
    // Diagnostics
    void diagnose() const;

private:
    int threshold_;
    
    std::unordered_map<std::string, int> word2idx_;
    std::unordered_map<std::string, int> metadata2idx_;
    
    std::vector<std::string> idx2word_;
    std::vector<std::string> idx2metadata_;
    
    HuffmanNode* huffman_root_ = nullptr;
    
    void cleanupHuffmanTree(HuffmanNode* node);
    void generateCodes(HuffmanNode* node, std::vector<int>& code, std::vector<int>& path);
};

} // namespace fasttext

#endif

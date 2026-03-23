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
                         const std::unordered_map<std::string, int>& context_freq);
    
    // Build Huffman tree for hierarchical softmax
    void buildHuffmanTree();
    
    // Accessors
    inline int getWordIdx(const std::string& word) const {
        auto it = word2idx_.find(word);
        return (it != word2idx_.end()) ? it->second : -1;
    }
    
    inline int getContextIdx(const std::string& ctx) const {
        auto it = context2idx_.find(ctx);
        return (it != context2idx_.end()) ? it->second : -1;
    }
    
    // NEW: Reverse lookup methods
    inline const std::string& getWord(int idx) const { 
        if (idx < 0 || idx >= static_cast<int>(idx2word_.size())) {
            static const std::string empty = "";
            return empty;
        }
        return idx2word_[idx]; 
    }
    
    inline const std::string& getContext(int idx) const { 
        if (idx < 0 || idx >= static_cast<int>(idx2context_.size())) {
            static const std::string empty = "";
            return empty;
        }
        return idx2context_[idx]; 
    }
    
    inline int wordSize() const { return word2idx_.size(); }
    inline int contextSize() const { return context2idx_.size(); }
    inline int huffmanNodes() const { return std::max(1, wordSize() - 1); }
    
    // NEW: Methods for model loading (populate vocabulary from serialized data)
    void addWord(int idx, const std::string& word) {
        word2idx_[word] = idx;
        if (idx >= static_cast<int>(idx2word_.size())) {
            idx2word_.resize(idx + 1);
        }
        idx2word_[idx] = word;
    }
    
    void addContext(int idx, const std::string& ctx) {
        context2idx_[ctx] = idx;
        if (idx >= static_cast<int>(idx2context_.size())) {
            idx2context_.resize(idx + 1);
        }
        idx2context_[idx] = ctx;
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
    std::unordered_map<std::string, int> context2idx_;
    
    std::vector<std::string> idx2word_;  // Reverse lookup
    std::vector<std::string> idx2context_;
    
    HuffmanNode* huffman_root_ = nullptr;
    
    void cleanupHuffmanTree(HuffmanNode* node);
    void generateCodes(HuffmanNode* node, std::vector<int>& code, std::vector<int>& path);
};

} // namespace fasttext

#endif

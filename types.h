#ifndef FASTTEXT_TYPES_H
#define FASTTEXT_TYPES_H

#include <vector>
#include <string>
#include <cstdint>

namespace fasttext {

// Training sample parsed from file
struct StreamingSample {
    std::vector<std::string> metadata_fields;
    std::vector<std::string> words;
};

// Node for Huffman Tree construction
struct HuffmanNode {
    int word_idx;
    double frequency;
    HuffmanNode* left;
    HuffmanNode* right;
    
    HuffmanNode(int idx, double freq) 
        : word_idx(idx), frequency(freq), left(nullptr), right(nullptr) {}
};

// Constants
constexpr float MIN_NORM = 1e-8f;

} // namespace fasttext

#endif

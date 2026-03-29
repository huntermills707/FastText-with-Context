#ifndef FASTTEXT_TYPES_H
#define FASTTEXT_TYPES_H

#include <vector>
#include <string>
#include <cstdint>

namespace fasttext {

// Training sample parsed from triple-pipe-delimited file.
// Format: <PatientGroup> ||| <ProviderGroup> ||| <WordsGroup>
struct GroupedSample {
    std::vector<std::string> patient_fields;   // e.g., {"elderly", "male", "white"}
    std::vector<std::string> provider_fields;  // e.g., {"attending", "emergency"}
    // Future: std::vector<std::string> outcome_fields;
    std::vector<std::string> words;            // sentence tokens
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

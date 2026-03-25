#include "vocabulary.h"
#include <iostream>
#include <queue>
#include <algorithm>
#include <functional>
#include <cmath>

namespace fasttext {

void Vocabulary::buildFromCounts(const std::unordered_map<std::string, int>& word_freq,
                                 const std::unordered_map<std::string, int>& metadata_freq) {
    int word_idx = 0;
    int filtered_count = 0;

    // Sort words by frequency for deterministic ordering
    std::vector<std::pair<std::string, int>> sorted_words(word_freq.begin(), word_freq.end());
    std::sort(sorted_words.begin(), sorted_words.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& [word, count] : sorted_words) {
        if (count >= threshold_) {
            word2idx_[word] = word_idx;
            idx2word_.push_back(word);
            word_freqs_.push_back(static_cast<double>(count));
            word_idx++;
        } else {
            filtered_count++;
        }
    }

    int meta_idx = 0;
    // Sort metadata fields similarly
    std::vector<std::pair<std::string, int>> sorted_meta(metadata_freq.begin(), metadata_freq.end());
    std::sort(sorted_meta.begin(), sorted_meta.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& [meta, count] : sorted_meta) {
        metadata2idx_[meta] = meta_idx;
        idx2metadata_.push_back(meta);
        meta_idx++;
    }

    std::cout << "\n=== VOCABULARY BUILD SUMMARY ===" << std::endl;
    std::cout << "Total unique words in file: " << word_freq.size() << std::endl;
    std::cout << "Words meeting threshold (" << threshold_ << "): " << word_idx << std::endl;
    std::cout << "Words filtered out: " << filtered_count << std::endl;
    std::cout << "Metadata fields: " << metadata2idx_.size() << std::endl;
    std::cout << "=================================\n" << std::endl;
}

void Vocabulary::buildHuffmanTree() {
    int vocab_size = word2idx_.size();
    std::cout << "\n=== HUFFMAN TREE CONSTRUCTION ===" << std::endl;
    std::cout << "Vocabulary size: " << vocab_size << std::endl;

    if (vocab_size == 0) {
        std::cerr << "ERROR: No words in vocabulary! Cannot build Huffman tree." << std::endl;
        return;
    }

    if (vocab_size == 1) {
        std::cerr << "WARNING: Only 1 word in vocabulary. Huffman tree will have depth 0." << std::endl;
        word_codes_.resize(1);
        word_paths_.resize(1);
        word_codes_[0] = {};
        word_paths_[0] = {};
        return;
    }

    auto cmp = [](HuffmanNode* a, HuffmanNode* b) {
        return a->frequency > b->frequency;
    };
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, decltype(cmp)> pq(cmp);

    std::vector<HuffmanNode*> nodes(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        nodes[i] = new HuffmanNode(i, word_freqs_[i]);
        pq.push(nodes[i]);
    }

    std::cout << "Priority queue initialized with " << vocab_size << " leaf nodes" << std::endl;

    int internal_nodes = 0;
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();

        HuffmanNode* parent = new HuffmanNode(-1, left->frequency + right->frequency);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
        internal_nodes++;
    }

    huffman_root_ = pq.top();

    std::cout << "Tree constructed with " << internal_nodes << " internal nodes" << std::endl;
    std::cout << "Root node frequency: " << huffman_root_->frequency << std::endl;

    word_codes_.resize(vocab_size);
    word_paths_.resize(vocab_size);

    std::vector<int> code;
    std::vector<int> path;
    generateCodes(huffman_root_, code, path);

    std::cout << "Codes and paths generated for " << vocab_size << " words" << std::endl;
    diagnose();
    std::cout << "================================\n" << std::endl;
}

void Vocabulary::generateCodes(HuffmanNode* node, std::vector<int>& code, std::vector<int>& path) {
    if (node == nullptr) return;

    if (node->word_idx >= 0) {
        // Leaf node
        word_codes_[node->word_idx] = code;
        word_paths_[node->word_idx] = path;
        return;
    }

    int path_idx = path.size();
    path.push_back(path_idx);

    // Left child gets code 0
    code.push_back(0);
    generateCodes(node->left, code, path);
    code.pop_back();

    // Right child gets code 1
    code.push_back(1);
    generateCodes(node->right, code, path);
    code.pop_back();

    path.pop_back();
}

void Vocabulary::cleanupHuffmanTree(HuffmanNode* node) {
    if (node == nullptr) return;
    cleanupHuffmanTree(node->left);
    cleanupHuffmanTree(node->right);
    delete node;
}

void Vocabulary::diagnose() const {
    std::cout << "\n=== HUFFMAN TREE DIAGNOSTICS ===" << std::endl;
    std::cout << "huffman_root_: " << (huffman_root_ ? "exists" : "NULL") << std::endl;
    std::cout << "word_codes_ size: " << word_codes_.size() << std::endl;
    std::cout << "word_paths_ size: " << word_paths_.size() << std::endl;

    if (!word_paths_.empty()) {
        std::cout << "Path depth statistics:" << std::endl;
        size_t min_depth = word_paths_[0].size();
        size_t max_depth = word_paths_[0].size();
        size_t total_depth = 0;

        for (const auto& path : word_paths_) {
            min_depth = std::min(min_depth, path.size());
            max_depth = std::max(max_depth, path.size());
            total_depth += path.size();
        }

        std::cout << "  Min depth: " << min_depth << std::endl;
        std::cout << "  Max depth: " << max_depth << std::endl;
        std::cout << "  Avg depth: " << (total_depth / word_paths_.size()) << std::endl;

        if (min_depth == 0) {
            std::cerr << "  ERROR: Some words have depth 0!" << std::endl;
            int zero_depth_count = 0;
            for (const auto& path : word_paths_) {
                if (path.size() == 0) zero_depth_count++;
            }
            std::cerr << "  Words with depth 0: " << zero_depth_count << std::endl;
        }
    }

    if (huffman_root_) {
        std::cout << "Root node: word_idx=" << huffman_root_->word_idx 
                  << ", frequency=" << huffman_root_->frequency << std::endl;
    }

    std::cout << "================================\n" << std::endl;
}

} // namespace fasttext

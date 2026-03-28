#include "fasttext_context.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.bin> <word1> [word2 ...] [--ctx <meta1> [meta2 ...]] [--k <number>]\n\n"
              << "Arguments:\n"
              << "  model.bin    Path to the saved model file (required)\n"
              << "  word1, word2...  One or more words to combine (required)\n"
              << "  --ctx        Flag indicating METADATA fields follow (optional)\n"
              << "  meta1, meta2...    Metadata field values\n"
              << "  --k          Flag indicating number of neighbors (optional, default: 10)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) { printUsage(argv[0]); return 1; }

    std::string modelFile = argv[1];
    std::vector<std::string> words, metadata;
    int k = 10;

    bool metaFlag = false, kFlag = false;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--ctx") { metaFlag = true; kFlag = false; continue; }
        else if (arg == "--k") { kFlag = true; metaFlag = false; continue; }
        if (kFlag) { try { k = std::stoi(arg); kFlag = false; } catch (...) { std::cerr << "Error: Invalid --k\n"; return 1; } }
        else if (metaFlag) metadata.push_back(arg);
        else words.push_back(arg);
    }

    if (words.empty()) { std::cerr << "Error: At least one word required.\n"; return 1; }
    if (k < 1) k = 1;

    try {
        fasttext::FastTextContext ft;
        std::cout << "Loading model from " << modelFile << "..." << std::endl;
        ft.loadModel(modelFile);

        std::cout << "Searching for neighbors of: [";
        for (size_t i = 0; i < words.size(); ++i) { std::cout << words[i]; if (i < words.size()-1) std::cout << ", "; }
        std::cout << "]";
        if (!metadata.empty()) {
            std::cout << " with metadata: [";
            for (size_t i = 0; i < metadata.size(); ++i) { std::cout << metadata[i]; if (i < metadata.size()-1) std::cout << ", "; }
            std::cout << "]";
        }
        std::cout << "\nFinding " << k << " nearest neighbors...\n";

        auto neighbors = ft.getNearestNeighbors(words, metadata, k);
        if (neighbors.empty()) { std::cout << "No neighbors found.\n"; return 0; }

        std::cout << "\nResults:\nRank\tWord\t\tScore\n----\t----\t\t-----\n";
        int rank = 1;
        for (const auto& [word, score] : neighbors)
            std::cout << rank++ << "\t" << word << "\t\t" << score << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

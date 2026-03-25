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
              << "               (e.g., author, domain, year - NOT surrounding words)\n"
              << "  meta1, meta2...    Metadata field values (e.g., alice, tech, 2024)\n"
              << "  --k          Flag indicating number of neighbors (optional, default: 10)\n"
              << "  number       Number of neighbors to return (default: 10)\n"
              << std::endl;
    std::cerr << "Examples:\n"
              << "  # Find neighbors of 'machine' with no metadata\n"
              << "  " << prog << " model.bin machine --k 10\n"
              << "  # Find neighbors of 'bitcoin' conditioned on metadata 'finance' and 'bob'\n"
              << "  " << prog << " model.bin bitcoin --ctx finance bob --k 10\n"
              << "  # Combine multiple words and metadata\n"
              << "  " << prog << " model.bin machine learning --ctx alice tech --k 20\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string modelFile = argv[1];
    std::vector<std::string> words;
    std::vector<std::string> metadata;
    int k = 10;

    // Parse arguments
    bool metaFlag = false;
    bool kFlag = false;
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--ctx") {
            metaFlag = true;
            kFlag = false;
            continue;
        }
        else if (arg == "--k") {
            kFlag = true;
            metaFlag = false;
            continue;
        }
        
        if (kFlag) {
            try {
                k = std::stoi(arg);
                kFlag = false;
            } catch (...) {
                std::cerr << "Error: Invalid value for --k: " << arg << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (metaFlag) {
            // Add to metadata
            metadata.push_back(arg);
        }
        else {
            // Add to words (default behavior)
            words.push_back(arg);
        }
    }

    if (words.empty()) {
        std::cerr << "Error: At least one word is required." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    if (k < 1) {
        std::cerr << "Error: k must be at least 1. Setting to 1." << std::endl;
        k = 1;
    }

    try {
        fasttext::FastTextContext ft;
        
        std::cout << "Loading model from " << modelFile << "..." << std::endl;
        ft.loadModel(modelFile);
        
        std::cout << "Searching for neighbors of: [";
        for (size_t i = 0; i < words.size(); ++i) {
            std::cout << words[i];
            if (i < words.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        
        if (!metadata.empty()) {
            std::cout << " with metadata: [";
            for (size_t i = 0; i < metadata.size(); ++i) {
                std::cout << metadata[i];
                if (i < metadata.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        std::cout << std::endl;
        
        std::cout << "Finding " << k << " nearest neighbors..." << std::endl;
        
        auto neighbors = ft.getNearestNeighbors(words, metadata, k);
        
        if (neighbors.empty()) {
            std::cout << "No neighbors found. Check your input words." << std::endl;
            return 0;
        }
        
        std::cout << "\nResults:\n";
        std::cout << "Rank\tWord\t\tScore\n";
        std::cout << "----\t----\t\t-----\n";
        
        int rank = 1;
        for (const auto& [word, score] : neighbors) {
            std::cout << rank++ << "\t" << word << "\t\t" << score << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

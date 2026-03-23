#include "fasttext_context.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.bin> <word1> [word2 ...] [--ctx <ctx1> [ctx2 ...]] [--k <number>]\n\n"
              << "Arguments:\n"
              << "  model.bin    Path to the saved model file (required)\n"
              << "  word1, word2...  One or more words to combine (required)\n"
              << "  --ctx        Flag indicating context fields follow (optional)\n"
              << "  ctx1, ctx2...    Context fields (e.g., alice, tech)\n"
              << "  --k          Flag indicating number of neighbors (optional, default: 10)\n"
              << "  number       Number of neighbors to return (default: 10)\n"
              << std::endl;
    std::cerr << "Examples:\n"
              << "  " << prog << " model.bin machine 10\n"
              << "  " << prog << " model.bin machine --k 20\n"
              << "  " << prog << " model.bin machine learning --ctx alice tech --k 10\n"
              << "  " << prog << " model.bin machine --ctx alice --k 15\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string modelFile = argv[1];
    std::vector<std::string> words;
    std::vector<std::string> contexts;
    int k = 10;  // Default value

    // Parse arguments
    bool ctxFlag = false;
    bool kFlag = false;
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--ctx") {
            ctxFlag = true;
            kFlag = false;
            continue;
        }
        else if (arg == "--k") {
            kFlag = true;
            ctxFlag = false;
            continue;
        }
        
        if (kFlag) {
            // Next argument after --k should be the number
            try {
                k = std::stoi(arg);
                kFlag = false;  // Reset flag after consuming the value
            } catch (...) {
                std::cerr << "Error: Invalid value for --k: " << arg << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (ctxFlag) {
            // Add to contexts
            contexts.push_back(arg);
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
        
        if (!contexts.empty()) {
            std::cout << " with contexts: [";
            for (size_t i = 0; i < contexts.size(); ++i) {
                std::cout << contexts[i];
                if (i < contexts.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        std::cout << std::endl;
        
        std::cout << "Finding " << k << " nearest neighbors..." << std::endl;
        
        auto neighbors = ft.getNearestNeighbors(words, contexts, k);
        
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

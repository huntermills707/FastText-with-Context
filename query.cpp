#include "fasttext_context.h"
#include <iostream>
#include <vector>
#include <string>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.bin> <word> [k]" << std::endl;
    std::cerr << "  model.bin: Path to the saved model file" << std::endl;
    std::cerr << "  word: The word to find neighbors for" << std::endl;
    std::cerr << "  k: Number of neighbors (default: 10)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string modelFile = argv[1];
    std::string queryWord = argv[2];
    int k = 10;
    
    if (argc >= 4) {
        try {
            k = std::stoi(argv[3]);
        } catch (...) {
            std::cerr << "Invalid k value. Defaulting to 10." << std::endl;
        }
    }

    try {
        fasttext::FastTextContext ft;
        
        std::cout << "Loading model from " << modelFile << "..." << std::endl;
        ft.loadModel(modelFile);
        
        std::cout << "Finding " << k << " nearest neighbors for '" << queryWord << "'..." << std::endl;
        
        auto neighbors = ft.getNearestNeighbors(queryWord, k);
        
        if (neighbors.empty()) {
            std::cout << "No neighbors found. The word might be out of vocabulary." << std::endl;
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

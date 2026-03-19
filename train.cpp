#include "fasttext_context.h"
#include <iostream>
#include <chrono>
#include <cstdlib>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <training_file> <output_model.bin>" << std::endl;
    std::cerr << "Example: " << prog << " data.txt model.bin" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];

    std::cout << "OpenMP available threads: " << omp_get_max_threads() << std::endl;
    
    fasttext::FastTextContext ft(
        100,      // dim
        5,        // epoch
        0.05f,    // lr
        3,        // min_n
        6,        // max_n
        5         // threshold
    );
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::cout << "=== Training FastText ===" << std::endl;
        std::cout << "Input: " << inputFile << std::endl;
        std::cout << "Output: " << outputFile << std::endl;
        
        ft.train(inputFile);
        
        std::cout << "Saving model..." << std::endl;
        ft.saveModel(outputFile);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Total time: " << duration.count() << " seconds" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

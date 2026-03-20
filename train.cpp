#include "fasttext_context.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <omp.h>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <training_file> <output_model.bin>\n\n"
              << "Options:\n"
              << "  -dim <int>        Embedding dimension (default: 100)\n"
              << "  -epoch <int>      Number of training epochs (default: 5)\n"
              << "  -lr <float>       Learning rate (default: 0.05)\n"
              << "  -minn <int>       Minimum n-gram length (default: 3)\n"
              << "  -maxn <int>       Maximum n-gram length (default: 6)\n"
              << "  -threshold <int>  Word frequency threshold (default: 5)\n"
              << "  -threads <int>    Number of OpenMP threads (default: system max)\n"
              << "  -help             Show this help message\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Default parameters
    int dim = 100;
    int epoch = 5;
    float lr = 0.05f;
    int min_n = 3;
    int max_n = 6;
    int threshold = 5;
    int threads = omp_get_max_threads();

    std::string inputFile;
    std::string outputFile;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-help" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-dim" && i + 1 < argc) {
            dim = std::stoi(argv[++i]);
        }
        else if (arg == "-epoch" && i + 1 < argc) {
            epoch = std::stoi(argv[++i]);
        }
        else if (arg == "-lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        }
        else if (arg == "-minn" && i + 1 < argc) {
            min_n = std::stoi(argv[++i]);
        }
        else if (arg == "-maxn" && i + 1 < argc) {
            max_n = std::stoi(argv[++i]);
        }
        else if (arg == "-threshold" && i + 1 < argc) {
            threshold = std::stoi(argv[++i]);
        }
        else if (arg == "-threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        }
        else if (arg[0] != '-') {
            // Positional arguments
            if (inputFile.empty()) {
                inputFile = arg;
            } else if (outputFile.empty()) {
                outputFile = arg;
            }
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (inputFile.empty() || outputFile.empty()) {
        std::cerr << "Error: Input and output files must be specified.\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Set number of threads
    omp_set_num_threads(threads);

    std::cout << "=== FastText Context Training ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Input:       " << inputFile << std::endl;
    std::cout << "  Output:      " << outputFile << std::endl;
    std::cout << "  Dimension:   " << dim << std::endl;
    std::cout << "  Epochs:      " << epoch << std::endl;
    std::cout << "  LR:          " << lr << std::endl;
    std::cout << "  N-grams:     " << min_n << "-" << max_n << std::endl;
    std::cout << "  Threshold:   " << threshold << std::endl;
    std::cout << "  Threads:     " << threads << std::endl;
    std::cout << std::endl;

    try {
        fasttext::FastTextContext ft(dim, epoch, lr, min_n, max_n, threshold);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        ft.train(inputFile);
        
        std::cout << "Saving model to " << outputFile << "..." << std::endl;
        ft.saveModel(outputFile);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        
        std::cout << "\nDone! Total time: " << duration.count() << " seconds" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

#include "fasttext_context.h"
#include <iostream>
#include <chrono>

int main() {
    // Set number of OpenMP threads via environment variable or default
    std::cout << "OpenMP available threads: " << omp_get_max_threads() << std::endl;
    
    fasttext::FastTextContext ft(
        100,      // embedding dimension
        5,        // epochs
        0.05f,    // learning rate
        3,        // min n-gram length
        6,        // max n-gram length
        5         // word frequency threshold
    );
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::cout << "=== Training FastText with OpenMP Parallelization ===" << std::endl;
        std::cout << "Features enabled:" << std::endl;
        std::cout << "  - Thread-local gradient accumulation (training)" << std::endl;
        std::cout << "  - Parallel nearest neighbor search" << std::endl;
        std::cout << "  - Parallel word vector computation (n-grams)" << std::endl;
        std::cout << std::endl;
        
        ft.train("training_data_with_context.txt");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "\nTraining took " << duration.count() << " seconds" << std::endl;
        
        // Test word vector
        std::string test_word = "machine";
        auto word_vec = ft.getWordVector(test_word);
        std::cout << "\n=== Word Vector ===" << std::endl;
        std::cout << "'" << test_word << "' first 5 dims: " 
                  << word_vec[0] << ", " << word_vec[1] << ", " 
                  << word_vec[2] << ", " << word_vec[3] << ", " 
                  << word_vec[4] << std::endl;
        
        // Test context vector
        std::string test_ctx = "alice";
        auto ctx_vec = ft.getContextVector(test_ctx);
        std::cout << "\n=== Context Vector ===" << std::endl;
        std::cout << "'" << test_ctx << "' first 5 dims: " 
                  << ctx_vec[0] << ", " << ctx_vec[1] << ", " 
                  << ctx_vec[2] << ", " << ctx_vec[3] << ", " 
                  << ctx_vec[4] << std::endl;
        
        // Test combined vector
        std::vector<std::string> contexts = {"alice"};
        auto combined = ft.getCombinedVector(test_word, contexts);
        std::cout << "\n=== Combined Vector ===" << std::endl;
        std::cout << "First 5 dims: " << combined[0] << ", " << combined[1] << ", " 
                  << combined[2] << ", " << combined[3] << ", " 
                  << combined[4] << std::endl;
        
        // Verify additive property
        std::cout << "\n=== Verification ===" << std::endl;
        bool correct = true;
        for (int i = 0; i < 5; ++i) {
            float expected = word_vec[i] + ctx_vec[i];
            if (std::abs(combined[i] - expected) > 1e-6f) {
                correct = false;
                std::cout << "Mismatch at dim " << i << std::endl;
            }
        }
        std::cout << "Additive property: " << (correct ? "PASS" : "FAIL") << std::endl;
        
        // Nearest neighbors (parallelized)
        std::cout << "\n=== Nearest Neighbors (Parallelized Search) ===" << std::endl;
        auto neighbors = ft.getNearestNeighbors(test_word, 10);
        for (const auto& [word, score] : neighbors) {
            std::cout << "  " << word << ": " << score << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

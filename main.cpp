#include "fasttext_context.h"
#include <iostream>
#include <chrono>

int main() {
    fasttext::FastTextContext ft(
        100,      // word embedding dimension
        5,        // epochs
        0.05f,    // learning rate
        3,        // min n-gram
        6,        // max n-gram
        5,        // word threshold
        50        // context embedding dimension
    );
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // File format: context1|context2|...|sentence
        // Example: "author:john|domain:tech|date:2024|Machine learning is advancing rapidly"
        ft.train("training_data_with_context.txt");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Training took " << duration.count() << " seconds" << std::endl;
        
        // Test word vector
        std::string test_word = "machine";
        auto vec = ft.getWordVector(test_word);
        std::cout << "\nWord vector for '" << test_word << "': " 
                  << vec[0] << ", " << vec[1] << "..." << std::endl;
        
        // Test context vector
        std::string test_ctx = "john";
        auto ctx_vec = ft.getContextVector(test_ctx);
        std::cout << "Context vector for '" << test_ctx << "': " 
                  << ctx_vec[0] << ", " << ctx_vec[1] << "..." << std::endl;
        
        // Test combined vector
        std::vector<std::string> contexts = {"author:john", "domain:tech"};
        auto combined = ft.getCombinedVector(test_word, contexts);
        std::cout << "Combined vector size: " << combined.size() << std::endl;
        
        // Nearest neighbors
        std::cout << "\nNearest neighbors to '" << test_word << "':" << std::endl;
        auto neighbors = ft.getNearestNeighbors(test_word, 10);
        for (const auto& [word, sim] : neighbors) {
            std::cout << "  " << word << ": " << sim << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

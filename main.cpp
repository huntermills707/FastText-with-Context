#include "fasttext_context.h"
#include <iostream>
#include <chrono>

int main() {
    // Create FastTextContext instance
    fasttext::FastTextContext ft(
        100,      // embedding dimension (used for BOTH word and context)
        5,        // epochs
        0.05f,    // learning rate
        3,        // min n-gram length
        6,        // max n-gram length
        5         // word frequency threshold
    );
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Train on context-piped data
        // Format: context1|context2|...|sentence
        // Example: "author:alice|domain:tech|year:2024|Machine learning is advancing"
        std::cout << "Starting training with additive context combination..." << std::endl;
        ft.train("training_data_with_context.txt");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "\nTraining took " << duration.count() << " seconds" << std::endl;
        
        // Test word vector retrieval
        std::string test_word = "machine";
        auto word_vec = ft.getWordVector(test_word);
        std::cout << "\n=== Word Vector ===" << std::endl;
        std::cout << "Word '" << test_word << "' vector (first 5 dims): " 
                  << word_vec[0] << ", " << word_vec[1] << ", " 
                  << word_vec[2] << ", " << word_vec[3] << ", " 
                  << word_vec[4] << "..." << std::endl;
        std::cout << "Total dimensions: " << word_vec.size() << std::endl;
        
        // Test context vector retrieval
        std::string test_ctx = "alice";
        auto ctx_vec = ft.getContextVector(test_ctx);
        std::cout << "\n=== Context Vector ===" << std::endl;
        std::cout << "Context '" << test_ctx << "' vector (first 5 dims): " 
                  << ctx_vec[0] << ", " << ctx_vec[1] << ", " 
                  << ctx_vec[2] << ", " << ctx_vec[3] << ", " 
                  << ctx_vec[4] << "..." << std::endl;
        std::cout << "Total dimensions: " << ctx_vec.size() << std::endl;
        
        // Test combined (additive) vector
        std::vector<std::string> contexts = {"alice"};
        auto combined = ft.getCombinedVector(test_word, contexts);
        std::cout << "\n=== Combined Vector (Additive) ===" << std::endl;
        std::cout << "Combined vector for '" << test_word << "' with contexts [";
        for (size_t i = 0; i < contexts.size(); ++i) {
            std::cout << contexts[i];
            if (i < contexts.size() - 1) std::cout << ", ";
        }
        std::cout << "]:" << std::endl;
        std::cout << "First 5 dims: " << combined[0] << ", " << combined[1] << ", " 
                  << combined[2] << ", " << combined[3] << ", " 
                  << combined[4] << "..." << std::endl;
        std::cout << "Total dimensions: " << combined.size() << std::endl;
        
        // Verify additive property: combined ≈ word_vec + context_vec
        std::cout << "\n=== Verification ===" << std::endl;
        std::cout << "Checking additive property (combined[i] ≈ word[i] + context[i]):" << std::endl;
        float diff_sum = 0.0f;
        for (int i = 0; i < 5; ++i) {
            float expected = word_vec[i] + ctx_vec[i];
            float actual = combined[i];
            float diff = std::abs(expected - actual);
            diff_sum += diff;
            std::cout << "  Dim " << i << ": word=" << word_vec[i] 
                      << " + ctx=" << ctx_vec[i] 
                      << " = " << expected 
                      << " | combined=" << actual 
                      << " | diff=" << diff << std::endl;
        }
        std::cout << "Average difference (should be ~0): " << (diff_sum / 5.0f) << std::endl;
        
        // Find nearest neighbors
        std::cout << "\n=== Nearest Neighbors ===" << std::endl;
        std::cout << "Nearest neighbors to '" << test_word << "':" << std::endl;
        auto neighbors = ft.getNearestNeighbors(test_word, 10);
        for (const auto& [word, sim] : neighbors) {
            std::cout << "  " << word << ": " << sim << std::endl;
        }
        
        // Compare context vectors (optional)
        std::cout << "\n=== Context Similarity ===" << std::endl;
        std::string ctx1 = "alice";
        std::string ctx2 = "bob";
        auto ctx1_vec = ft.getContextVector(ctx1);
        auto ctx2_vec = ft.getContextVector(ctx2);
        
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (int i = 0; i < 100; ++i) {
            dot += ctx1_vec[i] * ctx2_vec[i];
            norm1 += ctx1_vec[i] * ctx1_vec[i];
            norm2 += ctx2_vec[i] * ctx2_vec[i];
        }
        float similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2));
        std::cout << "Cosine similarity between '" << ctx1 << "' and '" << ctx2 << "': " 
                  << similarity << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

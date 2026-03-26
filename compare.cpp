#include "fasttext_context.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.bin> \\\n"
              << "       --words1 <word1> [word2 ...] [--meta1 <meta1> [meta2 ...]] \\\n"
              << "       --words2 <word1> [word2 ...] [--meta2 <meta1> [meta2 ...]]\n\n"
              << "Arguments:\n"
              << "  model.bin    Path to the saved model file (required)\n"
              << "  --words1     First set of words (required)\n"
              << "               (Words are represented by n-grams only, no word embeddings)\n"
              << "  --meta1      Metadata fields for first vector (optional)\n"
              << "  --words2     Second set of words (required)\n"
              << "  --meta2      Metadata fields for second vector (optional)\n"
              << std::endl;
    std::cerr << "Examples:\n"
              << "  # Compare two single words\n"
              << "  " << prog << " model.bin --words1 machine --words2 computer\n\n"
              << "  # Compare word combinations\n"
              << "  " << prog << " model.bin --words1 machine learning --words2 neural networks\n\n"
              << "  # Compare with metadata\n"
              << "  " << prog << " model.bin --words1 bitcoin --meta1 finance bob \\\n"
              << "                         --words2 crypto --meta2 tech alice\n\n"
              << "  # Same words, different metadata\n"
              << "  " << prog << " model.bin --words1 market --meta1 finance \\\n"
              << "                         --words2 market --meta2 tech\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }

    std::string modelFile = argv[1];
    std::vector<std::string> words1, words2;
    std::vector<std::string> meta1, meta2;

    // Parse arguments
    enum class ParseState {
        DEFAULT,
        WORDS1,
        META1,
        WORDS2,
        META2
    };
    
    ParseState state = ParseState::DEFAULT;
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--words1") {
            state = ParseState::WORDS1;
            continue;
        }
        else if (arg == "--meta1") {
            state = ParseState::META1;
            continue;
        }
        else if (arg == "--words2") {
            state = ParseState::WORDS2;
            continue;
        }
        else if (arg == "--meta2") {
            state = ParseState::META2;
            continue;
        }
        
        switch (state) {
            case ParseState::WORDS1:
                words1.push_back(arg);
                break;
            case ParseState::META1:
                meta1.push_back(arg);
                break;
            case ParseState::WORDS2:
                words2.push_back(arg);
                break;
            case ParseState::META2:
                meta2.push_back(arg);
                break;
            case ParseState::DEFAULT:
                std::cerr << "Error: Unexpected argument '" << arg << "' (no flag specified)\n";
                printUsage(argv[0]);
                return 1;
        }
    }

    // Validation
    if (words1.empty()) {
        std::cerr << "Error: --words1 requires at least one word.\n";
        printUsage(argv[0]);
        return 1;
    }
    if (words2.empty()) {
        std::cerr << "Error: --words2 requires at least one word.\n";
        printUsage(argv[0]);
        return 1;
    }

    try {
        fasttext::FastTextContext ft;
        
        std::cout << "Loading model from " << modelFile << "..." << std::endl;
        ft.loadModel(modelFile);

        std::vector<float> test_vec = ft.getWordVector("bitcoin");
        std::cout << "Debug: bitcoin vector norm = " << std::sqrt(std::inner_product(test_vec.begin(), test_vec.end(), test_vec.begin(), 0.0f)) << std::endl;
                
        // Compute combined vectors
        std::vector<float> vec1 = ft.getCombinedVector(words1, meta1);
        std::vector<float> vec2 = ft.getCombinedVector(words2, meta2);
        
        // Check for zero vectors
        float norm1 = 0.0f, norm2 = 0.0f;
        for (float v : vec1) norm1 += v * v;
        for (float v : vec2) norm2 += v * v;
        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);
        
        if (norm1 < 1e-8f) {
            std::cerr << "Error: Vector 1 has near-zero magnitude. Check input words/metadata.\n";
            return 1;
        }
        if (norm2 < 1e-8f) {
            std::cerr << "Error: Vector 2 has near-zero magnitude. Check input words/metadata.\n";
            return 1;
        }
        
        // Compute cosine similarity (vectors are already normalized from getCombinedVector)
        float dot = 0.0f;
        int dim = ft.getDim();
        for (int i = 0; i < dim; ++i) {
            dot += vec1[i] * vec2[i];
        }
        
        // Cosine similarity (should be in [-1, 1] for normalized vectors)
        float cosine_sim = dot;
        
        // Compute Euclidean distance
        float euclidean_dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = vec1[i] - vec2[i];
            euclidean_dist += diff * diff;
        }
        euclidean_dist = std::sqrt(euclidean_dist);
        
        // Output results
        std::cout << "\n=== Vector Comparison ===\n\n";
        
        std::cout << "Vector 1:\n";
        std::cout << "  Words:    [";
        for (size_t i = 0; i < words1.size(); ++i) {
            std::cout << words1[i];
            if (i < words1.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        if (!meta1.empty()) {
            std::cout << "  Metadata: [";
            for (size_t i = 0; i < meta1.size(); ++i) {
                std::cout << meta1[i];
                if (i < meta1.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << "  Norm:     " << std::fixed << std::setprecision(6) << norm1 << "\n";
        
        std::cout << "\nVector 2:\n";
        std::cout << "  Words:    [";
        for (size_t i = 0; i < words2.size(); ++i) {
            std::cout << words2[i];
            if (i < words2.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        if (!meta2.empty()) {
            std::cout << "  Metadata: [";
            for (size_t i = 0; i < meta2.size(); ++i) {
                std::cout << meta2[i];
                if (i < meta2.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << "  Norm:     " << std::fixed << std::setprecision(6) << norm2 << "\n";
        
        std::cout << "\n--- Similarity Metrics ---\n";
        std::cout << "  Cosine Similarity:  " << std::fixed << std::setprecision(6) << cosine_sim << "\n";
        std::cout << "  Cosine Distance:    " << std::fixed << std::setprecision(6) << (1.0f - cosine_sim) << "\n";
        std::cout << "  Euclidean Distance: " << std::fixed << std::setprecision(6) << euclidean_dist << "\n";
        
        // Interpretation hint
        std::cout << "\n--- Interpretation ---\n";
        if (cosine_sim > 0.8f) {
            std::cout << "  Very similar (same semantic neighborhood)\n";
        } else if (cosine_sim > 0.5f) {
            std::cout << "  Moderately similar (related concepts)\n";
        } else if (cosine_sim > 0.2f) {
            std::cout << "  Weakly similar (loosely related)\n";
        } else if (cosine_sim > -0.2f) {
            std::cout << "  Unrelated (orthogonal in embedding space)\n";
        } else {
            std::cout << "  Opposite (antonymous or contrasting)\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

#include "fasttext_context.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.bin> \\\n"
              << "       --words1 <word1> [word2 ...] [--meta1 <meta1> [meta2 ...]] \\\n"
              << "       --words2 <word1> [word2 ...] [--meta2 <meta1> [meta2 ...]]\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 5) { printUsage(argv[0]); return 1; }

    std::string modelFile = argv[1];
    std::vector<std::string> words1, words2, meta1, meta2;

    enum class State { DEFAULT, WORDS1, META1, WORDS2, META2 };
    State state = State::DEFAULT;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--words1") { state = State::WORDS1; continue; }
        else if (arg == "--meta1")  { state = State::META1;  continue; }
        else if (arg == "--words2") { state = State::WORDS2; continue; }
        else if (arg == "--meta2")  { state = State::META2;  continue; }
        switch (state) {
            case State::WORDS1: words1.push_back(arg); break;
            case State::META1:  meta1.push_back(arg);  break;
            case State::WORDS2: words2.push_back(arg); break;
            case State::META2:  meta2.push_back(arg);  break;
            default: std::cerr << "Unexpected arg: " << arg << "\n"; return 1;
        }
    }

    if (words1.empty() || words2.empty()) { std::cerr << "Both --words1 and --words2 required.\n"; return 1; }

    try {
        fasttext::FastTextContext ft;
        std::cout << "Loading model from " << modelFile << "..." << std::endl;
        ft.loadModel(modelFile);

        std::vector<float> vec1 = ft.getCombinedVector(words1, meta1);
        std::vector<float> vec2 = ft.getCombinedVector(words2, meta2);

        float norm1 = 0.0f, norm2 = 0.0f;
        for (float v : vec1) norm1 += v * v;
        for (float v : vec2) norm2 += v * v;
        norm1 = std::sqrt(norm1); norm2 = std::sqrt(norm2);

        if (norm1 < 1e-8f) { std::cerr << "Error: Vector 1 near-zero.\n"; return 1; }
        if (norm2 < 1e-8f) { std::cerr << "Error: Vector 2 near-zero.\n"; return 1; }

        int dim = ft.getDim();
        float cosine_sim = 0.0f;
        for (int i = 0; i < dim; ++i) cosine_sim += vec1[i] * vec2[i];

        float euclidean_dist = 0.0f;
        for (int i = 0; i < dim; ++i) { float d = vec1[i] - vec2[i]; euclidean_dist += d*d; }
        euclidean_dist = std::sqrt(euclidean_dist);

        std::cout << "\n=== Vector Comparison ===\n\n";
        std::cout << "Vector 1: [";
        for (size_t i = 0; i < words1.size(); ++i) { std::cout << words1[i]; if (i < words1.size()-1) std::cout << ", "; }
        std::cout << "]";
        if (!meta1.empty()) { std::cout << " meta: ["; for (size_t i = 0; i < meta1.size(); ++i) { std::cout << meta1[i]; if (i < meta1.size()-1) std::cout << ", "; } std::cout << "]"; }
        std::cout << "\nVector 2: [";
        for (size_t i = 0; i < words2.size(); ++i) { std::cout << words2[i]; if (i < words2.size()-1) std::cout << ", "; }
        std::cout << "]";
        if (!meta2.empty()) { std::cout << " meta: ["; for (size_t i = 0; i < meta2.size(); ++i) { std::cout << meta2[i]; if (i < meta2.size()-1) std::cout << ", "; } std::cout << "]"; }
        std::cout << "\n\n--- Similarity Metrics ---\n";
        std::cout << "  Cosine Similarity:  " << std::fixed << std::setprecision(6) << cosine_sim << "\n";
        std::cout << "  Cosine Distance:    " << std::fixed << std::setprecision(6) << (1.0f - cosine_sim) << "\n";
        std::cout << "  Euclidean Distance: " << std::fixed << std::setprecision(6) << euclidean_dist << "\n";

        std::cout << "\n--- Interpretation ---\n";
        if      (cosine_sim > 0.8f)  std::cout << "  Very similar\n";
        else if (cosine_sim > 0.5f)  std::cout << "  Moderately similar\n";
        else if (cosine_sim > 0.2f)  std::cout << "  Weakly similar\n";
        else if (cosine_sim > -0.2f) std::cout << "  Unrelated\n";
        else                         std::cout << "  Opposite\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

#include "fasttext_context.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.bin> \\\n"
              << "       --words1 <w1> [w2 ...] [--patient1 <p1> [p2 ...]] [--encounter1 <e1> [e2 ...]] \\\n"
              << "       --words2 <w1> [w2 ...] [--patient2 <p1> [p2 ...]] [--encounter2 <e1> [e2 ...]]\n\n"
              << "Examples:\n"
              << "  " << prog << " model.bin --words1 chest pain --patient1 elderly male --encounter1 attending \\\n"
              << "                            --words2 chest pain --patient2 young_adult female --encounter2 resident\n"
              << "  " << prog << " model.bin --words1 market --patient1 elderly --words2 market --patient2 young_adult\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) { printUsage(argv[0]); return 1; }

    std::string modelFile = argv[1];
    std::vector<std::string> words1, words2, patient1, patient2, encounter1, encounter2;

    enum class State { DEFAULT, WORDS1, PATIENT1, ENCOUNTER1, WORDS2, PATIENT2, ENCOUNTER2 };
    State state = State::DEFAULT;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--words1")      { state = State::WORDS1;     continue; }
        else if (arg == "--patient1")    { state = State::PATIENT1;   continue; }
        else if (arg == "--encounter1")  { state = State::ENCOUNTER1; continue; }
        else if (arg == "--words2")      { state = State::WORDS2;     continue; }
        else if (arg == "--patient2")    { state = State::PATIENT2;   continue; }
        else if (arg == "--encounter2")  { state = State::ENCOUNTER2; continue; }

        switch (state) {
            case State::WORDS1:      words1.push_back(arg);      break;
            case State::PATIENT1:    patient1.push_back(arg);    break;
            case State::ENCOUNTER1:  encounter1.push_back(arg);  break;
            case State::WORDS2:      words2.push_back(arg);      break;
            case State::PATIENT2:    patient2.push_back(arg);    break;
            case State::ENCOUNTER2:  encounter2.push_back(arg);  break;
            default: std::cerr << "Unexpected arg: " << arg << "\n"; return 1;
        }
    }

    if (words1.empty() || words2.empty()) {
        std::cerr << "Both --words1 and --words2 are required.\n";
        return 1;
    }

    try {
        fasttext::FastTextContext ft;
        std::cout << "Loading model from " << modelFile << "...\n";
        ft.loadModel(modelFile);

        std::vector<float> vec1 = ft.getCombinedVector(words1, patient1, encounter1);
        std::vector<float> vec2 = ft.getCombinedVector(words2, patient2, encounter2);

        // Both are L2-normalised by getCombinedVector, so norms should be ~1.
        float norm1 = 0.0f, norm2 = 0.0f;
        for (float v : vec1) norm1 += v * v;
        for (float v : vec2) norm2 += v * v;
        norm1 = std::sqrt(norm1); norm2 = std::sqrt(norm2);

        if (norm1 < 1e-8f) { std::cerr << "Error: Vector 1 near-zero.\n"; return 1; }
        if (norm2 < 1e-8f) { std::cerr << "Error: Vector 2 near-zero.\n"; return 1; }

        int dim = ft.getDOut();
        float cosine_sim = 0.0f;
        for (int i = 0; i < dim; ++i) cosine_sim += vec1[i] * vec2[i];

        float euclidean_dist = 0.0f;
        for (int i = 0; i < dim; ++i) { float d = vec1[i] - vec2[i]; euclidean_dist += d * d; }
        euclidean_dist = std::sqrt(euclidean_dist);

        std::cout << "\n=== Vector Comparison ===\n\n";

        auto printGroup = [](const std::string& label,
                              const std::vector<std::string>& words,
                              const std::vector<std::string>& patient,
                              const std::vector<std::string>& encounter) {
            std::cout << label << " words: [";
            for (size_t i = 0; i < words.size(); ++i) {
                std::cout << words[i]; if (i < words.size()-1) std::cout << ", ";
            }
            std::cout << "]";
            if (!patient.empty()) {
                std::cout << "  patient: [";
                for (size_t i = 0; i < patient.size(); ++i) {
                    std::cout << patient[i]; if (i < patient.size()-1) std::cout << ", ";
                }
                std::cout << "]";
            }
            if (!encounter.empty()) {
                std::cout << "  encounter: [";
                for (size_t i = 0; i < encounter.size(); ++i) {
                    std::cout << encounter[i]; if (i < encounter.size()-1) std::cout << ", ";
                }
                std::cout << "]";
            }
            std::cout << "\n";
        };

        printGroup("A:", words1, patient1, encounter1);
        printGroup("B:", words2, patient2, encounter2);

        std::cout << "\n--- Similarity Metrics ---\n"
                  << "  Cosine Similarity:  " << std::fixed << std::setprecision(6) << cosine_sim  << "\n"
                  << "  Cosine Distance:    " << std::fixed << std::setprecision(6) << (1.0f - cosine_sim) << "\n"
                  << "  Euclidean Distance: " << std::fixed << std::setprecision(6) << euclidean_dist << "\n"
                  << "\n--- Interpretation ---\n";

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

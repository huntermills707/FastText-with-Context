#include "fasttext_context.h"
#include <iostream>
#include <vector>
#include <string>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <model.bin> <word1> [word2 ...] [--patient <p1> [p2 ...]] [--provider <pr1> [pr2 ...]] [--k <num>]\n\n"
              << "Arguments:\n"
              << "  model.bin            Path to the saved model file (required)\n"
              << "  word1, word2...      One or more words to combine (required)\n"
              << "  --patient            Flag: patient metadata fields follow (optional)\n"
              << "  --provider           Flag: provider metadata fields follow (optional)\n"
              << "  --k                  Number of nearest neighbors (optional, default: 10)\n\n"
              << "Examples:\n"
              << "  " << prog << " model.bin chest pain --patient elderly male white medicare --provider attending emergency --k 10\n"
              << "  " << prog << " model.bin chest pain --k 10\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) { printUsage(argv[0]); return 1; }

    std::string modelFile = argv[1];
    std::vector<std::string> words, patient_meta, provider_meta;
    int k = 10;

    enum class State { WORDS, PATIENT, PROVIDER, K };
    State state = State::WORDS;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--patient")  { state = State::PATIENT;  continue; }
        else if (arg == "--provider") { state = State::PROVIDER; continue; }
        else if (arg == "--k")        { state = State::K;        continue; }

        switch (state) {
            case State::WORDS:    words.push_back(arg);         break;
            case State::PATIENT:  patient_meta.push_back(arg);  break;
            case State::PROVIDER: provider_meta.push_back(arg); break;
            case State::K:
                try { k = std::stoi(arg); } catch (...) { std::cerr << "Error: Invalid --k\n"; return 1; }
                state = State::WORDS;
                break;
        }
    }

    if (words.empty()) { std::cerr << "Error: At least one word required.\n"; return 1; }
    if (k < 1) k = 1;

    try {
        fasttext::FastTextContext ft;
        std::cout << "Loading model from " << modelFile << "...\n";
        ft.loadModel(modelFile);

        std::cout << "Query: [";
        for (size_t i = 0; i < words.size(); ++i) {
            std::cout << words[i];
            if (i < words.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (!patient_meta.empty()) {
            std::cout << "  patient: [";
            for (size_t i = 0; i < patient_meta.size(); ++i) {
                std::cout << patient_meta[i];
                if (i < patient_meta.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        if (!provider_meta.empty()) {
            std::cout << "  provider: [";
            for (size_t i = 0; i < provider_meta.size(); ++i) {
                std::cout << provider_meta[i];
                if (i < provider_meta.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        std::cout << "\nFinding " << k << " nearest neighbors...\n";

        auto neighbors = ft.getNearestNeighbors(words, patient_meta, provider_meta, k);
        if (neighbors.empty()) { std::cout << "No neighbors found.\n"; return 0; }

        std::cout << "\nResults:\nRank\tWord\t\tScore\n----\t----\t\t-----\n";
        int rank = 1;
        for (const auto& [word, score] : neighbors)
            std::cout << rank++ << "\t" << word << "\t\t" << score << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

#include "fasttext_context.h"
#include <iostream>
#include <chrono>
#include <omp.h>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <training_file> <output_model.bin>\n\n"
              << "Dimension options:\n"
              << "  -d-word <int>         Word + n-gram embedding dimension (default: 150)\n"
              << "  -d-patient <int>      Patient group embedding dimension (default: 30)\n"
              << "  -d-provider <int>     Provider group embedding dimension (default: 15)\n"
              << "  -d-out <int>          Output/projected dimension for HS (default: 150)\n\n"
              << "Training options:\n"
              << "  -epoch <int>          Number of training epochs (default: 5)\n"
              << "  -lr <float>           Learning rate (default: 0.05)\n"
              << "  -minn <int>           Minimum n-gram length (default: 3)\n"
              << "  -maxn <int>           Maximum n-gram length (default: 8)\n"
              << "  -threshold <int>      Word frequency threshold (default: 5)\n"
              << "  -subsample <float>    Subsampling threshold t (default: 1e-4)\n"
              << "  -grad-clip <float>    Gradient norm clip threshold (default: 1.0, 0=off)\n"
              << "  -weight-decay <float> L2 decay on W_proj per chunk (default: 0, off)\n"
              << "  -threads <int>        Number of OpenMP threads (default: system max)\n"
              << "  -chunk-size <int>     Samples per sync chunk (default: 1000)\n"
              << "  -ngram-buckets <int>  N-gram hash buckets (default: 2000000)\n"
              << "  -window-size <int>    Max skip-gram window size (default: 5)\n"
              << "  -help                 Show this help message\n\n"
              << "Data format: <PatientGroup> ||| <ProviderGroup> ||| <WordsGroup>\n"
              << "  Elements within each group are space-delimited.\n"
              << "  Groups may be empty (delimiters still required).\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    int   d_w = 150, d_p = 30, d_pr = 15, d_out = 150;
    int   epoch = 5, min_n = 3, max_n = 8;
    int   threshold = 5, threads = omp_get_max_threads();
    int   chunk_size = 1000, ngram_buckets = 2000000, window_size = 5;
    float lr = 0.05f, subsample_t = 1e-4f, grad_clip = 1.0f, weight_decay = 0.0f;
    std::string inputFile, outputFile;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-help" || arg == "--help") { printUsage(argv[0]); return 0; }
        else if (arg == "-d-word"       && i+1 < argc) d_w           = std::stoi(argv[++i]);
        else if (arg == "-d-patient"    && i+1 < argc) d_p           = std::stoi(argv[++i]);
        else if (arg == "-d-provider"   && i+1 < argc) d_pr          = std::stoi(argv[++i]);
        else if (arg == "-d-out"        && i+1 < argc) d_out         = std::stoi(argv[++i]);
        else if (arg == "-epoch"        && i+1 < argc) epoch         = std::stoi(argv[++i]);
        else if (arg == "-lr"           && i+1 < argc) lr            = std::stof(argv[++i]);
        else if (arg == "-minn"         && i+1 < argc) min_n         = std::stoi(argv[++i]);
        else if (arg == "-maxn"         && i+1 < argc) max_n         = std::stoi(argv[++i]);
        else if (arg == "-threshold"    && i+1 < argc) threshold     = std::stoi(argv[++i]);
        else if (arg == "-subsample"    && i+1 < argc) subsample_t   = std::stof(argv[++i]);
        else if (arg == "-grad-clip"    && i+1 < argc) grad_clip     = std::stof(argv[++i]);
        else if (arg == "-threads"      && i+1 < argc) threads       = std::stoi(argv[++i]);
        else if (arg == "-chunk-size"   && i+1 < argc) chunk_size    = std::stoi(argv[++i]);
        else if (arg == "-ngram-buckets"&& i+1 < argc) ngram_buckets = std::stoi(argv[++i]);
        else if (arg == "-window-size"  && i+1 < argc) window_size   = std::stoi(argv[++i]);
        else if (arg == "-weight-decay"  && i+1 < argc) weight_decay  = std::stof(argv[++i]);
        else if (arg[0] != '-') {
            if      (inputFile.empty())  inputFile  = arg;
            else if (outputFile.empty()) outputFile = arg;
        } else { std::cerr << "Unknown argument: " << arg << "\n"; printUsage(argv[0]); return 1; }
    }

    if (inputFile.empty() || outputFile.empty()) {
        std::cerr << "Error: input and output files required.\n";
        printUsage(argv[0]);
        return 1;
    }

    omp_set_num_threads(threads);

    std::cout << "=== FastTextContext Training (Concat+Projection) ===\n"
              << "  Input:             " << inputFile       << "\n"
              << "  Output:            " << outputFile      << "\n"
              << "  d_word:            " << d_w             << "\n"
              << "  d_patient:         " << d_p             << "\n"
              << "  d_provider:        " << d_pr            << "\n"
              << "  d_out:             " << d_out           << "\n"
              << "  concat_dim:        " << (d_w + d_p + d_pr) << "\n"
              << "  W_proj shape:      " << d_out << " x " << (d_w + d_p + d_pr) << "\n"
              << "  Epochs:            " << epoch           << "\n"
              << "  LR:                " << lr              << "\n"
              << "  N-grams:           " << min_n << "-" << max_n << "\n"
              << "  Threshold:         " << threshold       << "\n"
              << "  Subsample t:       " << subsample_t     << "\n"
              << "  Grad clip:         " << (grad_clip > 0.0f ? std::to_string(grad_clip) : "off") << "\n"
              << "  Window (max):      " << window_size     << "\n"
              << "  Threads:           " << threads         << "\n"
              << "  Chunk size:        " << chunk_size      << "\n"
              << "  N-gram buckets:    " << ngram_buckets   << "\n"
              << "  Sparse updates:    Hogwild (input, ngram, output)\n"
              << "  Dense sync:        chunk-averaged (W_proj, patient, provider)\n"
              << "  W_proj decay:      " << (weight_decay > 0.0f ? std::to_string(weight_decay) : "off") << "\n"
              << "  Data format:       <PatientGroup> ||| <ProviderGroup> ||| <Words>\n\n";

    try {
        fasttext::FastTextContext ft(d_w, d_p, d_pr, d_out,
                                     epoch, lr, min_n, max_n, threshold,
                                     chunk_size, ngram_buckets, window_size,
                                     subsample_t, grad_clip, weight_decay);
        auto t0 = std::chrono::high_resolution_clock::now();
        ft.trainStreaming(inputFile);
        ft.saveModel(outputFile);
        auto t1   = std::chrono::high_resolution_clock::now();
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();
        std::cout << "Done. Total time: " << secs << "s\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

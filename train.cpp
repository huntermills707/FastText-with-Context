#include "fasttext_context.h"
#include <iostream>
#include <chrono>
#include <omp.h>

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <training_file> <output_model.bin>\n\n"
              << "Options:\n"
              << "  -dim <int>            Embedding dimension (default: 100)\n"
              << "  -epoch <int>          Number of training epochs (default: 5)\n"
              << "  -lr <float>           Learning rate (default: 0.05)\n"
              << "  -minn <int>           Minimum n-gram length (default: 3)\n"
              << "  -maxn <int>           Maximum n-gram length (default: 8)\n"
              << "  -threshold <int>      Word frequency threshold (default: 5)\n"
              << "  -subsample <float>    Subsampling threshold t (default: 1e-4)\n"
              << "  -grad-clip <float>    Gradient norm clip threshold (default: 1.0, 0=off)\n"
              << "  -threads <int>        Number of OpenMP threads (default: system max)\n"
              << "  -chunk-size <int>     Samples per chunk (default: 100000)\n"
              << "  -ngram-buckets <int>  N-gram hash buckets (default: 2000000)\n"
              << "  -window-size <int>    Max skip-gram window size (default: 5)\n"
              << "  -help                 Show this help message\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    int   dim = 100, epoch = 5, min_n = 3, max_n = 8;
    int   threshold = 5, threads = omp_get_max_threads();
    int   chunk_size = 100000, ngram_buckets = 2000000, window_size = 5;
    float lr = 0.05f, subsample_t = 1e-4f, grad_clip = 1.0f;
    std::string inputFile, outputFile;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-help" || arg == "--help")  { printUsage(argv[0]); return 0; }
        else if (arg == "-dim"          && i+1 < argc) dim           = std::stoi(argv[++i]);
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

    std::cout << "=== FastTextContext Training ===\n"
              << "  Input:           " << inputFile       << "\n"
              << "  Output:          " << outputFile      << "\n"
              << "  Dimension:       " << dim             << "\n"
              << "  Epochs:          " << epoch           << "\n"
              << "  LR:              " << lr              << "\n"
              << "  N-grams:         " << min_n << "-" << max_n << "\n"
              << "  Threshold:       " << threshold       << "\n"
              << "  Subsample t:     " << subsample_t     << "\n"
              << "  Grad clip:       " << (grad_clip > 0.0f ? std::to_string(grad_clip) : "off") << "\n"
              << "  Window (max):    " << window_size     << "\n"
              << "  Threads:         " << threads         << "\n"
              << "  Chunk size:      " << chunk_size      << "\n"
              << "  N-gram buckets:  " << ngram_buckets   << "\n"
              << "  Input repr:      word embedding + n-grams + gated metadata\n"
              << "  Metadata gate:   word-dependent sigmoid gate + per-field alpha\n"
              << "  SGD:             Hogwild lock-free\n\n";

    try {
        fasttext::FastTextContext ft(dim, epoch, lr, min_n, max_n, threshold,
                                     chunk_size, ngram_buckets, window_size,
                                     subsample_t, grad_clip);
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

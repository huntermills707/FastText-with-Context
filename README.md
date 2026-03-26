# FastTextContext

A high-performance, custom C++ implementation of FastText extended with additive metadata embeddings. Built from scratch with OpenMP parallelization, this library supports streaming training on large datasets and allows word vectors to be dynamically conditioned on metadata (e.g., author, domain, timestamp).

## Key Features

- **Additive Metadata Modeling**: Unlike standard FastText, this implementation learns separate embeddings for metadata fields and adds them element-wise to word vectors during training and inference.
  - Formula: `combined_vector = word_embedding + sum(ngram_embeddings) + sum(metadata_embeddings)`
- **Streaming Training**: Two-pass architecture (vocabulary counting → training) designed to handle datasets larger than RAM.
- **Hierarchical Softmax**: Optimized training using Huffman trees for logarithmic complexity relative to vocabulary size.
- **Subword Awareness**: Character n-grams (configurable range) handle out-of-vocabulary words and morphological variations.
- **Multi-threaded**: Full OpenMP integration for parallel matrix initialization, gradient updates (Hogwild), and nearest-neighbor search.
- **Three Binaries**:
  - `train`: Streamlined training pipeline.
  - `query`: Interactive nearest-neighbor search with metadata support.
  - `compare`: Cosine/Euclidean similarity analysis between complex word + metadata combinations.
- **Python Compatibility**: Includes a pure Python loader (`fasttext_context.py`) for easy integration into data science workflows.

## Prerequisites

- **Compiler**: GCC 7+ or Clang 5+ (C++17 support required)
- **Libraries**: OpenMP (`libomp` or `libgomp`)
- **Python** (optional, for the loader script): `numpy`

## Building

The project uses a simple Makefile.

```bash
# Clone or copy source files
cd FastTextContext

# Build all binaries (train, query, compare)
make all

# Build individually
make train
make query
make compare

# Clean build artifacts
make clean
```

### Compiler Flags

The default build uses aggressive optimization:

```makefile
CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -fopenmp
```

## Usage

### 1. Training

The training process reads a pipe-delimited text file where the last field is the sentence and all preceding fields are metadata.

**Data Format Example (`data.txt`):**

```text
alice|tech|Machine learning is advancing rapidly
bob|finance|Bitcoin prices fluctuate wildly
alice|tech|2024|Deep learning models improve accuracy
```

**Basic Command:**

```bash
./train data.txt model.bin
```

**Advanced Options:**

```bash
./train -dim 200 -epoch 10 -lr 0.05 -minn 3 -maxn 6 -threshold 5 \
        -threads 8 -chunk-size 100000 -ngram-buckets 2000000 \
        -window-size 20 -subsample 1e-4 -grad-clip 1.0 \
        data.txt model.bin
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-dim` | `100` | Embedding dimension |
| `-epoch` | `5` | Number of training epochs |
| `-lr` | `0.05` | Initial learning rate (linear decay to 0.01% of initial) |
| `-minn` | `3` | Minimum n-gram length |
| `-maxn` | `6` | Maximum n-gram length |
| `-threshold` | `5` | Minimum word frequency to include in vocabulary |
| `-subsample` | `1e-4` | Subsampling threshold `t`; high-frequency words are discarded with probability `1 - sqrt(t / freq)` |
| `-grad-clip` | `1.0` | Maximum L2 norm for gradient vectors before they are applied; set to `0` to disable |
| `-threads` | System max | OpenMP thread count |
| `-chunk-size` | `100000` | Samples per processing chunk |
| `-ngram-buckets` | `2000000` | Hash buckets for subword n-gram hashing |
| `-window-size` | `20` | Maximum skip-gram window size (sampled uniformly from `[1, window-size]` per center word) |

### 2. Querying (Nearest Neighbors)

Find words semantically similar to a query, optionally conditioned on metadata.

**Syntax:**

```bash
./query <model.bin> <word1> [word2 ...] [--ctx <meta1> [meta2 ...]] [--k <num>]
```

**Examples:**

```bash
# Find neighbors of "machine" with no metadata
./query model.bin machine --k 10

# Find neighbors of "bitcoin" conditioned on metadata "finance" and "bob"
./query model.bin bitcoin --ctx finance bob --k 10

# Combine multiple words and metadata
./query model.bin machine learning --ctx alice tech --k 20
```

### 3. Comparing Vectors

Calculate similarity metrics between two complex queries (words + metadata).

**Syntax:**

```bash
./compare <model.bin> \
    --words1 <w1> [w2 ...] [--meta1 <m1> [m2 ...]] \
    --words2 <w1> [w2 ...] [--meta2 <m1> [m2 ...]]
```

**Examples:**

```bash
# Compare two single words
./compare model.bin --words1 machine --words2 computer

# Compare word combinations
./compare model.bin --words1 machine learning --words2 neural networks

# Same word, different metadata context
./compare model.bin --words1 market --meta1 finance --words2 market --meta2 tech

# Combine multiple words and metadata
./compare model.bin --words1 bitcoin --meta1 finance bob \
                    --words2 crypto  --meta2 tech alice
```

**Output includes:**
- Cosine Similarity
- Cosine Distance
- Euclidean Distance
- Semantic interpretation hint

## Python Integration

A Python loader is provided to load the binary model and perform inference without compiling C++.

**Installation:**

```bash
pip install numpy
```

**Usage:**

```python
from fasttext_context import FastTextContext

ft = FastTextContext()
ft.load_model("model.bin")

# Get word vector (word embedding + n-gram embeddings)
vec = ft.get_word_vector("machine")

# Get combined vector (words + metadata), L2-normalised
combined = ft.get_combined_vector(["bitcoin"], ["finance"])

# Find nearest neighbors
neighbors = ft.get_nearest_neighbors(["bitcoin"], ["finance"], k=5)
for word, score in neighbors:
    print(f"{word}: {score:.4f}")

# Compare two concepts
similarity = ft.compare_vectors(["market"], ["finance"], ["market"], ["tech"])
print(f"Cosine Similarity: {similarity['cosine_similarity']}")
```

## Architecture Overview

### Data Flow

1. **Pass 1 (Vocabulary)**: Scans the input file to count word and metadata frequencies. Words below `-threshold` are filtered out.
2. **Huffman Tree**: Constructs a binary tree based on word frequencies using the original word2vec two-pointer algorithm. Internal nodes are indexed into the output (hierarchical softmax) matrix.
3. **Initialization**: All matrices (input, output, n-gram, metadata) are initialized with values sampled from `N(0, 1/sqrt(dim))` using thread-local RNGs.
4. **Pass 2 (Training)**:
   - Parses lines into `StreamingSample` objects, buffered in chunks.
   - For each center word, builds `center_vec = word_embedding + sum(ngram_embeddings) + sum(metadata_embeddings)`.
   - Performs a forward/backward pass along the Huffman path for each context word (hierarchical softmax).
   - Updates all matrices in-place using **Hogwild** (lock-free, intentionally racy writes) — there is no gradient accumulation buffer or merge step.
   - Gradient norms are clipped at two points: per output-node update and on the accumulated center gradient before it is distributed to the input, n-gram, and metadata matrices.
5. **Learning Rate**: Decays linearly from `-lr` to `0.01%` of its initial value over the total number of training steps.

### Memory Layout

| Matrix | Shape | Description |
| :--- | :--- | :--- |
| Input | `(vocab_size, dim)` | Word-level embeddings |
| N-gram | `(ngram_buckets, dim)` | Subword character n-gram embeddings |
| Metadata | `(metadata_size, dim)` | Metadata field embeddings |
| Output | `(vocab_size - 1, dim)` | Hierarchical softmax internal node weights |

At inference time a `(vocab_size, dim)` cache of precomputed word vectors (word embedding + n-gram sum) is held in memory to accelerate nearest-neighbor search.

## Performance Tuning

- **Memory**: Decrease `-ngram-buckets` (e.g., to `1000000`) or `-dim` for smaller models.
- **Speed**: Increase `-chunk-size` to amortize file I/O overhead; tune `-threads` to match your CPU core count.
- **Quality**: Increase `-epoch` and lower `-threshold` for better coverage of rare words, at the cost of training time. Lower `-subsample` (e.g., `1e-5`) to retain more high-frequency words.
- **Stability**: If training diverges, reduce `-lr` or tighten `-grad-clip`.

## File Structure

```
.
├── fasttext_context.h      # Core class definition
├── fasttext_context.cpp    # Training orchestration & inference dispatch
├── types.h                 # Shared structs (StreamingSample, HuffmanNode, constants)
├── matrix.h                # Matrix class (header-only)
├── vocabulary.h/cpp        # Vocabulary building, Huffman tree, subsampling
├── trainer.h/cpp           # Training loop, Hogwild SGD, gradient logic
├── inference.h/cpp         # Vector computation & nearest-neighbor search
├── train.cpp               # CLI: training
├── query.cpp               # CLI: nearest-neighbor queries
├── compare.cpp             # CLI: vector comparison
├── fasttext_context.py     # Python loader (numpy only)
├── generate_sentences.py   # Synthetic training data generator
├── Makefile                # Build configuration
└── README.md               # This file
```

## License

MIT License. See LICENSE for details.

# FastTextContext

A high-performance, custom C++ implementation of FastText extended with additive context embeddings. Built from scratch with OpenMP parallelization, this library supports streaming training on large datasets and allows word vectors to be dynamically adjusted by metadata (e.g., author, domain, timestamp).

## Key Features

* Additive Context Modeling: Unlike standard FastText, this implementation learns separate embeddings for context fields (metadata) and adds them element-wise to word vectors during training and inference.
    * Formula: combined_vector = word_vector + context_vector
* Streaming Training: Two-pass architecture (vocabulary counting -> training) designed to handle datasets larger than RAM.
* Hierarchical Softmax: Optimized training using Huffman trees for logarithmic complexity relative to vocabulary size.
* Subword Awareness: Character n-grams (configurable range) handle out-of-vocabulary words and morphological variations.
* Multi-threaded: Full OpenMP integration for parallel matrix initialization, gradient accumulation, and nearest-neighbor search.
* Three Binaries:
    * train: Streamlined training pipeline.
    * query: Interactive nearest-neighbor search with context support.
    * compare: Cosine/Euclidean similarity analysis between complex word+context combinations.
* Python Compatibility: Includes a pure Python loader (fasttext_context.py) for easy integration into data science workflows.

## Prerequisites

* Compiler: GCC 7+ or Clang 5+ (C++17 support required)
* Libraries: OpenMP (libomp or libgomp)
* Python (Optional, for the loader script): numpy, struct

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

The training process reads a pipe-delimited text file where the last field is the sentence and preceding fields are context metadata.

**Data Format Example (data.txt):**
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
./train -dim 200 -epoch 10 -lr 0.05 -minn 3 -maxn 6 -threshold 5 -threads 8 -chunk-size 10000 -ngram-buckets 2000000 data.txt model.bin
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-dim` | `100` | Embedding dimension |
| `-epoch` | `5` | Number of training epochs |
| `-lr` | `0.05` | Initial learning rate (linear decay) |
| `-minn` | `3` | Minimum n-gram length |
| `-maxn` | `6` | Maximum n-gram length |
| `-threshold` | `5` | Minimum word frequency to include |
| `-threads` | `System` | OpenMP thread count |
| `-chunk-size` | `10000` | Samples processed before gradient merge |
| `-ngram-buckets` | `2000000` | Hash buckets for subword hashing |

### 2. Querying (Nearest Neighbors)

Find words semantically similar to a query, optionally conditioned on context.

**Syntax:**
```bash
./query <model.bin> <word1> [word2...] [--ctx <ctx1> [ctx2...]] [--k <num>]
```

**Examples:**
```bash
# Find neighbors of "machine" with no context
./query model.bin machine --k 10

# Find neighbors of "bitcoin" conditioned on "finance" and "bob"
./query model.bin bitcoin --ctx finance bob --k 10

# Combine multiple words and contexts
./query model.bin machine learning --ctx alice tech --k 20
```

### 3. Comparing Vectors

Calculate similarity metrics between two complex queries (words + contexts).

**Syntax:**
```bash
./compare <model.bin> --words1 <w1> [w2...] [--ctx1 <c1>...] --words2 <w1> [w2...] [--ctx2 <c1>...]
```

**Example:**
```bash
# Compare "market" in finance vs "market" in tech
./compare model.bin --words1 market --ctx1 finance --words2 market --ctx2 tech
```

**Output includes:**
* Cosine Similarity
* Cosine Distance
* Euclidean Distance
* Semantic interpretation hints

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

# Get word vector (includes n-grams)
vec = ft.get_word_vector("machine")

# Get combined vector (words + contexts)
combined = ft.get_combined_vector(["bitcoin"], ["finance"])

# Find neighbors
neighbors = ft.get_nearest_neighbors(["bitcoin"], ["finance"], k=5)
for word, score in neighbors:
    print(f"{word}: {score:.4f}")

# Compare two concepts
similarity = ft.compare_vectors(["market"], ["finance"], ["market"], ["tech"])
print(f"Cosine Similarity: {similarity['cosine_similarity']}")
```

## Architecture Overview

### Data Flow
1. Pass 1 (Vocabulary): Scans input file to count word and context frequencies. Filters by threshold.
2. Huffman Tree: Constructs a binary tree based on word frequencies for Hierarchical Softmax.
3. Initialization: Matrices (Input, Output, Context, N-gram) are initialized with small random values using thread-local RNGs.
4. Pass 2 (Training):
    * Parses lines into StreamingSample objects.
    * Computes combined_vector = word_vec + avg(context_vec).
    * Performs forward/backward pass along the Huffman path.
    * Accumulates gradients in thread-local buffers.
    * Merges gradients periodically (chunk-based) to update the Output matrix.

### Memory Layout
* Input Matrix: (Vocab_Size, Dim) - Word embeddings.
* Context Matrix: (Context_Size, Dim) - Metadata embeddings.
* N-gram Matrix: (Hash_Buckets, Dim) - Subword embeddings.
* Output Matrix: (Huffman_Nodes, Dim) - Hierarchical softmax weights.

## Performance Tuning

* Memory Reduction: Decrease -ngram-buckets (e.g., to 1000000) or -dim for smaller models.
* Speed: Increase -chunk-size to reduce synchronization overhead, or tune -threads to match your CPU core count.
* Quality: Increase -epoch and lower -threshold for better coverage of rare words, at the cost of training time.

## File Structure

```
.
├── fasttext_context.h      # Core class definition
├── fasttext_context.cpp    # Training & Inference logic
├── types.h                 # Shared structs (HuffmanNode, Sample)
├── matrix.h                # Matrix class (inline)
├── vocabulary.h/cpp        # Vocab building & Huffman tree
├── trainer.h/cpp           # Training loop & gradient logic
├── inference.h/cpp         # Vector computation & NN search
├── train.cpp               # CLI for training
├── query.cpp               # CLI for nearest neighbors
├── compare.cpp             # CLI for vector comparison
├── fasttext_context.py     # Python loader
├── Makefile                # Build configuration
└── README.md               # Documentation
```

## License

MIT License. See LICENSE for details.
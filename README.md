# FastTextContext

A custom implementation of FastText with context-aware embeddings, built from scratch in C++ with OpenMP parallelization.

## Overview

FastTextContext extends the original FastText algorithm by incorporating metadata/context embeddings alongside traditional word and subword (n-gram) representations. This allows the model to capture not just semantic meaning, but also contextual signals like author, domain, or other metadata that influence word usage.

Key features:
- **Additive context combination**: Context vectors are added to word vectors in the same embedding space
- **Hierarchical softmax**: Efficient training with logarithmic complexity
- **Subword information**: Character n-grams (3-6 chars) for handling out-of-vocabulary words
- **OpenMP parallelization**: Multi-threaded training and inference
- **Model persistence**: Binary save/load for fast deployment

## Building

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+)
- OpenMP support
- CMake (optional, for build automation)

### Compilation

```bash
# Clone or download the source
git clone <repository-url>
cd FastTextContext

# Build both binaries
make all

# Or build individually
make train    # Training binary
make query    # Inference binary
```

### Compiler Flags
```makefile
CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -fopenmp
LDFLAGS = -fopenmp
```

## Usage

### Training

```bash
# Basic usage
./train training_data.txt model.bin

# With custom parameters
./train -dim 200 -epoch 10 -lr 0.05 -threads 8 training_data.txt model.bin

# Full parameter list
./train -help
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-dim` | 100 | Embedding dimension (word & context) |
| `-epoch` | 5 | Number of training epochs |
| `-lr` | 0.05 | Learning rate |
| `-minn` | 3 | Minimum n-gram length |
| `-maxn` | 6 | Maximum n-gram length |
| `-threshold` | 5 | Word frequency threshold |
| `-threads` | System max | OpenMP thread count |

### Querying

```bash
# Single word, single context
./query model.bin machine --ctx alice --k 10

# Multiple words, multiple contexts
./query model.bin machine learning --ctx alice tech --k 10

# Multiple words, no context
./query model.bin machine learning 10

# Help
./query -help
```

## Data Format

### Training File Format

Each line represents one training sample with pipe-delimited fields:

```
context1|context2|...|contextN|sentence
```

**Example:**
```
alice|tech|Machine learning is advancing rapidly
bob|2024|Bitcoin prices fluctuate wildly
alice|tech|2024|Deep learning models improve accuracy
```

- **Context fields**: All fields except the last one are treated as context metadata
- **Sentence**: The last field is tokenized into words for training
- **Delimiter**: Pipe character (`|`) separates fields

### Model File Format

Binary format containing:
- Header (dimensions, thresholds, vocabulary sizes)
- Word vocabulary map
- Context vocabulary map
- Input matrix (word embeddings)
- Output matrix (hierarchical softmax nodes)
- N-gram matrix (subword embeddings)
- Context matrix
- Huffman tree codes/paths

**File size estimate:**
- N-gram matrix: ~800 MB (2M buckets × 100 dims × 4 bytes)
- Word matrix: ~20 MB (50K words × 100 dims × 4 bytes)
- Context matrix: ~0.4 MB (1K contexts × 100 dims × 4 bytes)

## Architecture

### Vector Combination

```
combined_vector = word_vector + context_vector
```

Both word and context vectors exist in the same embedding space (`dim` dimensions). Context vectors are averaged across multiple context fields, then added element-wise to the word vector.

### Training Algorithm

1. **Parse training data** → Extract words and context fields
2. **Build vocabulary** → Filter words by frequency threshold
3. **Build Huffman tree** → For hierarchical softmax
4. **Initialize matrices** → Random initialization with small values
5. **Train with hierarchical softmax** → Update weights along Huffman tree paths
6. **Thread-local gradients** → Accumulate locally, merge once per epoch

### Parallelization Strategy

| Component | Method | Expected Speedup |
|-----------|--------|------------------|
| Training loop | Thread-local gradients | 4-8× (8 cores) |
| Nearest neighbor search | Parallel iteration | 6-10× (8 cores) |
| Word vector computation | N-gram reduction | 1.5-2× |
| Matrix initialization | Parallel for | 2-4× |

## Performance Tuning

### Memory Optimization

To reduce model size:
```bash
# Reduce n-gram buckets (biggest impact)
# Edit fasttext_context.cpp: int ngram_buckets = 1000000;  # Instead of 2000000

# Reduce dimension
./train -dim 50 data.txt model.bin

# Increase threshold (fewer words in vocabulary)
./train -threshold 10 data.txt model.bin
```

### Thread Configuration

```bash
# Set OpenMP threads
export OMP_NUM_THREADS=8
./train data.txt model.bin

# Or use CLI flag
./train -threads 8 data.txt model.bin
```

## Limitations

- **No quantization**: Model saved as float32 (consider int16 for 50% size reduction)
- **Single-file training**: Does not support streaming or incremental updates
- **No supervised mode**: Only unsupervised word embedding training
- **Context dimension**: Must match word dimension (additive combination requirement)

## File Structure

```
FastTextContext/
├── fasttext_context.h     # Class definition
├── fasttext_context.cpp   # Implementation
├── train.cpp              # Training binary
├── query.cpp              # Inference binary
├── Makefile               # Build configuration
└── README.md              # This file
```

## License

MIT License - See LICENSE file for details.

# FastTextContext

A high-performance C++ implementation of FastText extended with a **concatenation-with-projection** architecture for stratified metadata groups. Built from scratch with OpenMP parallelization, it supports streaming training on large datasets and learns separate low-dimensional embeddings for each metadata group (patients, providers), which are concatenated and projected into a shared output space.

## Key Features

- **Stratified Metadata Groups**: Each metadata group (patient demographics, provider role) gets its own embedding space and dimension. Groups are concatenated and projected through a learned matrix rather than additively combined.
  - Forward pass: `concat = [word_part ; patient_avg ; provider_avg]` → `center_vec = W_proj × concat`
  - Extensible: adding a fourth group (e.g. outcomes) requires only a new embedding matrix and widening `W_proj` by `d_outcome` columns.
- **Hybrid Optimization**: Sparse word and n-gram parameters use Hogwild lock-free SGD. Dense parameters (`W_proj`, patient embeddings, provider embeddings) use thread-local copies with periodic synchronized averaging — one sync point per chunk, not per sample.
- **Streaming Training**: Two-pass architecture (vocabulary counting → training) handles datasets larger than RAM.
- **Hierarchical Softmax**: Huffman tree construction for O(log V) training complexity.
- **Subword Awareness**: Character n-grams (configurable range) handle out-of-vocabulary words.
- **Three Binaries**:
  - `train`: Streamlined training pipeline.
  - `query`: Interactive nearest-neighbor search with grouped metadata.
  - `compare`: Cosine/Euclidean similarity between complex word + metadata combinations.
- **Python Loader**: Pure-numpy `fasttext_context.py` for inference and diagnostics without recompiling.

## Prerequisites

- **Compiler**: GCC 7+ or Clang 5+ (C++17 required)
- **Libraries**: OpenMP (`libgomp` or `libomp`)
- **Python** (optional): `numpy`

## Building

```bash
make all      # build train, query, compare
make clean
```

Default compiler flags: `-O3 -std=c++17 -march=native -fopenmp`

## Data Format

Input lines use triple-pipe delimiters between groups:

```
<PatientGroup> ||| <ProviderGroup> ||| <WordsGroup>
```

- Each group's tokens are space-delimited.
- Individual tokens contain no internal spaces (use underscores).
- Groups may be empty; the delimiters are always present.
- The words group is always last.

**Examples:**

```
elderly male white english medicare married ||| attending emergency ||| the patient was admitted with chest pain
aged female hispanic ||| resident_physician elective ||| scheduled for elective knee replacement
 ||| nurse urgent ||| vital signs were stable on admission
elderly male ||| ||| no acute distress noted on examination
```

A synthetic data generator is included:

```bash
python3 generate_sentences.py   # writes training_data_with_context.txt (1M rows)
```

## Usage

### 1. Training

```bash
./train [options] <training_file> <output_model.bin>
```

**Basic command:**

```bash
./train data.txt model.bin
```

**Full example (MIMIC-III scale):**

```bash
./train -d-word 150 -d-patient 30 -d-provider 15 -d-out 150 \
        -epoch 10 -lr 0.05 -chunk-size 1000 -threads 8 \
        -weight-decay 1e-5 -grad-clip 1.0 \
        mimic-iii-sents.txt model.bin
```

**All flags:**

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-d-word` | `150` | Word + n-gram embedding dimension |
| `-d-patient` | `30` | Patient group embedding dimension |
| `-d-provider` | `15` | Provider group embedding dimension |
| `-d-out` | `150` | Output/projected dimension (HS and NN search operate here) |
| `-epoch` | `5` | Training epochs |
| `-lr` | `0.05` | Initial learning rate (linear decay to 0.01% of initial) |
| `-minn` | `3` | Minimum character n-gram length |
| `-maxn` | `8` | Maximum character n-gram length |
| `-threshold` | `5` | Minimum word frequency; patient/provider fields are not filtered |
| `-subsample` | `1e-4` | Subsampling threshold `t`; high-frequency words skipped with prob `1 - sqrt(t/freq)` |
| `-grad-clip` | `1.0` | Max L2 norm for gradient vectors (0 = off) |
| `-weight-decay` | `0` | L2 decay applied to `W_proj` after each chunk reduce (e.g. `1e-5`); 0 = off |
| `-threads` | system max | OpenMP thread count |
| `-chunk-size` | `1000` | Samples per broadcast→process→reduce sync cycle |
| `-ngram-buckets` | `2000000` | Hash buckets for character n-gram embeddings |
| `-window-size` | `5` | Max skip-gram window size (sampled uniformly from `[1, window-size]`) |

### 2. Querying (Nearest Neighbors)

Find words semantically similar to a query, optionally conditioned on grouped metadata.

```bash
./query <model.bin> <word1> [word2 ...] [--patient <p1> [p2 ...]] [--provider <pr1> [pr2 ...]] [--k <num>]
```

**Examples:**

```bash
# Words only
./query model.bin chest pain --k 10

# Words + full context
./query model.bin chest pain \
    --patient elderly male white medicare \
    --provider attending emergency \
    --k 10

# Words + patient context only
./query model.bin chest pain --patient young_adult female --k 5
```

### 3. Comparing Vectors

Calculate similarity metrics between two queries, each with independent metadata context.

```bash
./compare <model.bin> \
    --words1 <w1> [w2 ...] [--patient1 <p1> ...] [--provider1 <pr1> ...] \
    --words2 <w1> [w2 ...] [--patient2 <p1> ...] [--provider2 <pr1> ...]
```

**Examples:**

```bash
# Same words, different patient demographics
./compare model.bin \
    --words1 chest pain --patient1 elderly male \
    --words2 chest pain --patient2 young_adult female

# Same words, different provider context
./compare model.bin \
    --words1 shortness of breath --provider1 attending emergency \
    --words2 shortness of breath --provider2 resident elective

# Full context comparison
./compare model.bin \
    --words1 pain --patient1 elderly male white medicare --provider1 attending \
    --words2 pain --patient2 adult female hispanic medicaid --provider2 nurse
```

**Output:**

```
--- Similarity Metrics ---
  Cosine Similarity:  0.832541
  Cosine Distance:    0.167459
  Euclidean Distance: 0.578012

--- Interpretation ---
  Very similar
```

## Python Integration

```bash
pip install numpy
```

```python
from fasttext_context import FastTextContext

ft = FastTextContext()
ft.load_model("model.bin")

# Raw word vector (d_w-dimensional, unprojected)
vec = ft.get_word_vector("pain")

# Combined vector (all groups, projected, L2-normalised, d_out-dimensional)
combined = ft.get_combined_vector(
    words=["chest", "pain"],
    patient_meta=["elderly", "male"],
    provider_meta=["attending"]
)

# Group ablation: any subset of groups, absent ones are zeroed
words_only    = ft.get_group_vector(words=["chest", "pain"])
with_patient  = ft.get_group_vector(words=["chest", "pain"], patient_meta=["elderly"])
with_provider = ft.get_group_vector(words=["chest", "pain"], provider_meta=["attending"])

# Nearest neighbors
neighbors = ft.get_nearest_neighbors(
    ["chest", "pain"],
    patient_meta=["elderly", "male"],
    k=10
)
for word, score in neighbors:
    print(f"{word}: {score:.4f}")

# Compare two contexts
result = ft.compare_vectors(
    ["pain"], ["elderly"], ["attending"],
    ["pain"], ["young_adult"], ["resident"]
)
print(result["cosine_similarity"])

# Diagnostics
ft.print_projection_block_norms()  # check word/patient/provider blocks in W_proj
ft.print_patient_vocab()
ft.print_provider_vocab()
```

## Architecture

### Dimensions

| Symbol | Default | Description |
| :--- | :--- | :--- |
| `d_w` | 150 | Word + n-gram embedding space |
| `d_p` | 30 | Patient group embedding space |
| `d_pr` | 15 | Provider group embedding space |
| `d_out` | 150 | Projected output space (HS and NN search) |
| `concat_dim` | 195 | `d_w + d_p + d_pr` |
| `W_proj` | 150 × 195 | 29,250 parameters |

### Forward Pass (Single Center Word)

```
word_part    = word_emb[w] + sum(ngram_embs[w])           ∈ R^d_w
patient_part = avg(patient_embs[active patient fields])   ∈ R^d_p   (zero if none)
provider_part= avg(provider_embs[active provider fields]) ∈ R^d_pr  (zero if none)

concat       = [word_part ; patient_part ; provider_part] ∈ R^concat_dim
center_vec   = W_proj × concat                            ∈ R^d_out
```

The `center_vec` enters the standard skip-gram hierarchical softmax loop unchanged.

### Backward Pass Through Projection

After accumulating `center_grad ∈ R^d_out` from the HS loop (and clipping by L2 norm):

```
concat_grad  = W_proj^T × center_grad        (matrix-vector, concat_dim FLOPs)
W_proj      += lr × center_grad × concat^T   (outer product update, thread-local)
```

`concat_grad` is sliced into `word_grad`, `patient_grad`, `provider_grad` and distributed:

- `word_grad` → `input_matrix[word]` and each `ngram_matrix[idx]` (Hogwild)
- `patient_grad / N_patient` → each active `patient_matrix[idx]` (thread-local)
- `provider_grad / N_provider` → each active `provider_matrix[idx]` (thread-local)

### Synchronization Protocol

Dense parameters (`W_proj`, `patient_matrix`, `provider_matrix`) use thread-local copies with a per-chunk broadcast → process → reduce cycle:

1. **Broadcast**: copy shared dense params to all thread-local copies (sequential, before parallel region)
2. **Process**: threads update their local copies in parallel (no contention on dense params)
3. **Reduce**: average all thread-local copies back to shared (sequential, after barrier)

Sparse parameters (`input_matrix`, `ngram_matrix`, `output_matrix`) use Hogwild lock-free writes throughout.

This approach has one synchronization point per 1000 samples rather than one per center word — roughly 10,000× less frequent than a mutex-per-update approach, with no atomic float overhead.

### Inference Composition

At query time:

```
concat  = [avg(word_vecs) ; avg(patient_vecs) ; avg(provider_vecs)]
result  = L2_normalise(W_proj × concat)
```

For nearest-neighbor search, a cache of precomputed word-only projected vectors (`vocab_size × d_out`) is held in memory. Candidate scoring uses this cache; the query vector includes metadata. Both live in the same `d_out` space so cosine similarity is well-defined.

### Memory Layout

| Matrix | Shape | Update strategy |
| :--- | :--- | :--- |
| `input_matrix` | `vocab_size × d_w` | Hogwild |
| `ngram_matrix` | `ngram_buckets × d_w` | Hogwild |
| `output_matrix` | `(vocab_size-1) × d_out` | Hogwild |
| `W_proj` | `d_out × concat_dim` | Thread-local + chunk average |
| `patient_matrix` | `patient_vocab × d_p` | Thread-local + chunk average |
| `provider_matrix` | `provider_vocab × d_pr` | Thread-local + chunk average |

### Extending to a Fourth Group

To add an outcome group (dimension `d_o`):

1. Add `outcome_fields` to `GroupedSample` in `types.h`.
2. Add `outcome2idx_` / `outcomeSize()` to `Vocabulary`.
3. Add `outcome_matrix_` (`outcome_vocab × d_o`) to `FastTextContext` and `Trainer`.
4. Widen `W_proj` from `d_out × (d_w+d_p+d_pr)` to `d_out × (d_w+d_p+d_pr+d_o)`.
5. Extend `buildConcatVec`, `distributeGrad`, `getCombinedVector`, and the Python loader to handle the new slice.

No structural changes to the HS loop, synchronization protocol, or save/load format beyond the new sizes.

## MIMIC-III Preprocessing

The preprocessing script at `mimic-iii/mimic-iii-preprocess.py` reads ADMISSIONS, PATIENTS, CAREGIVERS, and NOTEEVENTS, cleans and sentence-segments clinical notes, and writes the triple-pipe format:

```
elderly male white english medicare ||| attending emergency ||| the patient presented with dyspnea
```

Patient columns: MeSH age category, gender, ethnicity, language, religion, marital status, insurance.
Provider columns: caregiver title, admission type.

```bash
cd mimic-iii
python3 mimic-iii-preprocess.py   # writes mimic-iii-sents.txt
```

## Performance Tuning

- **Memory**: reduce `-ngram-buckets` (e.g. `1000000`) or `-d-word` for smaller models.
- **Speed**: reduce `-chunk-size` if threads sit idle between syncs; increase it if sync overhead is visible in profiling. Tune `-threads` to physical core count.
- **Quality**: increase `-epoch` and lower `-threshold`. Use `-subsample 1e-5` to retain more high-frequency words.
- **Stability**: reduce `-lr` or tighten `-grad-clip` if loss diverges. Add `-weight-decay 1e-5` if `W_proj` Frobenius norm grows unchecked across epochs.
- **Metadata signal**: check `ft.print_projection_block_norms()` after training. If patient or provider blocks have near-zero norms, increase their dimension or lower `-chunk-size` (more frequent sync improves dense gradient accumulation).

## Testing

**Gradient verification:** Perturb each parameter by ε = 1e-4, measure loss change, compare to analytical gradient. Verify all groups: `input_matrix`, `ngram_matrix`, `output_matrix`, `W_proj`, `patient_matrix`, `provider_matrix`. The projection gradients are the most likely source of bugs.

**Sync correctness:** Train with 1 thread and 8 threads; compare loss curves. Similar final loss confirms the reduce step is correct. If multi-threaded loss is substantially worse, decrease `-chunk-size`.

**Group ablation (Python):**

```python
words_only    = ft.get_group_vector(words=["chest", "pain"])
with_patient  = ft.get_group_vector(words=["chest", "pain"], patient_meta=["elderly", "male"])
all_groups    = ft.get_group_vector(words=["chest", "pain"],
                                    patient_meta=["elderly", "male"],
                                    provider_meta=["attending"])
```

Verify that adding metadata shifts NN results in expected directions.

**Projection decomposition:**

```python
ft.print_projection_block_norms()
# W_word block should dominate; near-zero patient/provider blocks signal those groups aren't learning.
```

**Python/C++ parity:** Load the same model in both runtimes, compute combined vectors for identical inputs, verify they match to floating-point tolerance.

## File Structure

```
.
├── fasttext_context.h      # Core class definition
├── fasttext_context.cpp    # Training orchestration & inference dispatch
├── types.h                 # GroupedSample, HuffmanNode, constants
├── matrix.h                # Matrix class with mulVec/mulVecTranspose/addOuterProduct
├── vocabulary.h/cpp        # Three-group vocabulary (word/patient/provider)
├── trainer.h/cpp           # Training loop, hybrid SGD, projection backward pass
├── inference.h/cpp         # Concat+projection composition & NN search
├── train.cpp               # CLI: training
├── query.cpp               # CLI: nearest-neighbor queries
├── compare.cpp             # CLI: vector comparison
├── fasttext_context.py     # Python loader (numpy only)
├── generate_sentences.py   # Synthetic training data generator (triple-pipe format)
├── Makefile
├── mimic-iii/
│   └── mimic-iii-preprocess.py   # MIMIC-III preprocessing → triple-pipe format
└── README.md
```

## License

MIT License. See LICENSE for details.

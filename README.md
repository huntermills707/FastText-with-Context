# FastTextContext

A high-performance C++ implementation of FastText extended with a **concatenation-with-projection** architecture for stratified metadata groups. Built from scratch with OpenMP parallelization, it supports streaming training on large datasets and learns separate low-dimensional embeddings for each metadata group (patients, providers), which are concatenated and projected into a shared output space.

The project ships with a MIMIC-III preprocessing pipeline, a two-stage hierarchical bootstrap workflow for confidence intervals, and a pure-numpy Python loader for inference and diagnostics.

## Key Features

- **Stratified Metadata Groups**: Each metadata group (patient demographics, provider role) gets its own embedding space and dimension. Groups are concatenated and projected through a learned matrix rather than additively combined.
  - Forward pass: `concat = [word_part ; patient_avg ; provider_avg]` → `center_vec = W_proj × concat`
  - Extensible: adding a fourth group (e.g. outcomes) requires only a new embedding matrix and widening `W_proj` by `d_outcome` columns.
- **Hybrid Optimization**: Sparse word and n-gram parameters use Hogwild lock-free SGD. Dense parameters (`W_proj`, patient embeddings, provider embeddings) use thread-local copies with periodic synchronized averaging — one sync point per chunk, not per sample.
- **Streaming Training**: Two-pass architecture (vocabulary counting → training) handles datasets larger than RAM.
- **Hierarchical Softmax**: Huffman tree construction for O(log V) training complexity.
- **Subword Awareness**: Character n-grams (configurable range) handle out-of-vocabulary words.
- **Statistical Uncertainty**: Two-stage hierarchical bootstrap (patients → notes) for computing 95% confidence intervals on any similarity query, via `run_bootstrap.sh` + `compare_bootstrap.py`.
- **C++ Binaries**:
  - `train`: Streamlined training pipeline.
  - `query`: Interactive nearest-neighbor search with grouped metadata.
  - `compare`: Cosine/Euclidean similarity between complex word + metadata combinations.
- **Python Tooling**:
  - `fasttext_context.py`: Pure-numpy model loader for inference and diagnostics.
  - `compare_bootstrap.py`: Bootstrap-based similarity with 95% CIs.
  - `dump_metadata.py`: Export patient/provider vocabularies for inspection.

## Prerequisites

- **Compiler**: GCC 7+ or Clang 5+ (C++17 required)
- **Libraries**: OpenMP (`libgomp` or `libomp`)
- **Python** (optional): `numpy`; `polars`, `pyarrow`, `nltk`, `tqdm` for MIMIC-III preprocessing.

## Building

```bash
make all      # build train, query, compare
make clean
```

Default compiler flags: `-O3 -std=c++17 -march=native -fopenmp`

## Quick Start

Train a model on synthetic data in under a minute to verify the build:

```bash
make all
python3 generate_demo_sentences.py          # writes training_data_with_context.txt (1M rows)
./train -epoch 3 training_data_with_context.txt model.bin
./query model.bin chest pain --k 10
```

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
| `-d-out` | `150` | Output/projected dimension for HS |
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

### 4. Bootstrap Training Workflow

`run_bootstrap.sh` automates training an original model plus N bootstrap replicates. Each bootstrap uses `03_to_text.py --bootstrap-seed $i` (two-stage hierarchical resampling over patients and notes) so the resulting ensemble captures correct uncertainty for clustered note data.

```bash
./run_bootstrap.sh
```

**Configuration** (edit at the top of the script):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `INPUT_PARQUET` | `mimic-iii/mimic-iii-sents.parquet` | Sentences parquet from step 2 of the MIMIC-III pipeline |
| `ORIGINAL_TEXT` | `mimic-iii/mimic-iii-sents.txt` | Path for the non-bootstrapped training text |
| `ORIGINAL_MODEL` | `model.bin` | Output path for the original model |
| `LR` | `0.02` | Learning rate passed to `./train` |
| `WD` | `1e-5` | Weight decay on `W_proj` |
| `NUM_BOOTSTRAPS` | `25` | Number of bootstrap replicates |

The script writes `model.bin` plus `model_boot_1.bin` ... `model_boot_N.bin`. Bootstrap text files are deleted after each run to save disk space.

### 5. Bootstrap Confidence Intervals

Once the ensemble is trained, `compare_bootstrap.py` computes 95% CIs on the cosine similarity between pairs of queries by scoring each pair across every bootstrap replicate.

**Input files** — two plain-text files in triple-pipe format, one query per line:

```
# bases.txt
chest pain ||| elderly male ||| attending
shortness of breath ||| young_adult female ||| resident_physician

# targets.txt
dyspnea ||| elderly male ||| attending
myocardial infarction ||| aged male ||| attending
```

**Run:**

```bash
python3 compare_bootstrap.py <bases_file> <targets_file> [model_prefix]
```

Auto-discovers `<model_prefix>_boot_1.bin`, `<model_prefix>_boot_2.bin`, ... and compares the original point estimate to the empirical 95% interval (2.5th – 97.5th percentile) across the ensemble:

```bash
python3 compare_bootstrap.py bases.txt targets.txt model
```

**Output** (one row per base × target pair):

```
Base                                     | Target                                   | Original |       95% CI
-------------------------------------------------------------------------------------------------------------
chest pain elderly attending             | dyspnea elderly attending                |   0.8421 | [0.8156, 0.8689]
chest pain elderly attending             | myocardial infarction aged attending     |   0.7123 | [0.6801, 0.7452]
```

### 6. Inspecting Metadata Vocabularies

`dump_metadata.py` writes out the learned patient and provider vocabularies from a trained model — useful for sanity-checking field normalization and for assembling query inputs.

```bash
python3 dump_metadata.py    # reads mimic-iii.bin by default
```

**Outputs:**

- `patient_metadata.txt` — one patient field per line, sorted.
- `provider_metadata.txt` — one provider field per line, sorted.

The model path is hardcoded to `mimic-iii.bin` at the top of the script; edit it there to point at a different model.

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

Given a center word $w$ at position $c$ in a sentence with patient fields $\mathcal{P}$ and provider fields $\mathcal{R}$:

$$\mathbf{v}_{\text{word}} = \mathbf{e}_w + \sum_{g \in \mathcal{G}(w)} \mathbf{n}_g \quad \in \mathbb{R}^{d_w}$$

where $\mathbf{e}_w$ is the word embedding, $\mathcal{G}(w)$ is the set of character n-gram hash indices for $w$, and $\mathbf{n}_g$ are the n-gram embeddings.

$$\mathbf{v}_{\text{patient}} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \mathbf{m}_p \quad \in \mathbb{R}^{d_p} \qquad (\mathbf{0} \text{ if } \mathcal{P} = \emptyset)$$

$$\mathbf{v}_{\text{provider}} = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \mathbf{m}_r \quad \in \mathbb{R}^{d_{pr}} \qquad (\mathbf{0} \text{ if } \mathcal{R} = \emptyset)$$

The three parts are concatenated and projected into the output space:

$$\mathbf{z} = \begin{bmatrix} \mathbf{v}_{\text{word}} \\ \mathbf{v}_{\text{patient}} \\ \mathbf{v}_{\text{provider}} \end{bmatrix} \in \mathbb{R}^{d_{\text{concat}}}$$

$$\mathbf{h} = W_{\text{proj}} \, \mathbf{z} \quad \in \mathbb{R}^{d_{\text{out}}}$$

The projected vector $\mathbf{h}$ enters the skip-gram hierarchical softmax loop.

### Loss Function

For each context word $w_o$ within the sampled skip-gram window around center word $w_c$, the hierarchical softmax loss traverses the Huffman path $\{(n_1, d_1), \ldots, (n_L, d_L)\}$ from root to $w_o$, where $n_i$ is the $i$-th internal node and $d_i \in \{0, 1\}$ is the Huffman code bit (left or right branch):

$$\mathcal{L}(w_c, w_o) = -\sum_{i=1}^{L} \Big[ \, t_i \log \sigma(\mathbf{u}_{n_i}^\top \mathbf{h}) + (1 - t_i) \log \big(1 - \sigma(\mathbf{u}_{n_i}^\top \mathbf{h})\big) \Big]$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function, $\mathbf{u}_{n_i} \in \mathbb{R}^{d_{\text{out}}}$ is the output vector for internal node $n_i$, and $t_i = 1 - d_i$ maps the Huffman code to a binary target (left child $\to 1$, right child $\to 0$).

The total training loss over a sentence with $T$ words is:

$$\mathcal{L}_{\text{total}} = \sum_{c=1}^{T} \sum_{\substack{o = c - \tilde{w} \\ o \neq c}}^{c + \tilde{w}} \mathcal{L}(w_c, w_o)$$

where $\tilde{w} \sim \text{Uniform}\{1, \ldots, W\}$ is the sampled window size for each center word.

### Synchronization Protocol

Dense parameters ($W_{\text{proj}}$, `patient_matrix`, `provider_matrix`) use thread-local copies with a per-chunk broadcast → process → reduce cycle:

1. **Broadcast**: copy shared dense params to all thread-local copies (sequential, before parallel region)
2. **Process**: threads update their local copies in parallel (no contention on dense params)
3. **Reduce**: average all thread-local copies back to shared (sequential, after barrier)

Sparse parameters (`input_matrix`, `ngram_matrix`, `output_matrix`) use Hogwild lock-free writes throughout.

This approach has one synchronization point per 1000 samples rather than one per center word — roughly 10,000× less frequent than a mutex-per-update approach, with no atomic float overhead.

### Inference Composition

At query time:

$$\mathbf{z}_q = \begin{bmatrix} \frac{1}{|\mathcal{W}|} \sum_{w \in \mathcal{W}} \mathbf{v}_{\text{word}}(w) \\ \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \mathbf{m}_p \\ \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \mathbf{m}_r \end{bmatrix}$$

$$\mathbf{q} = \frac{W_{\text{proj}} \, \mathbf{z}_q}{\| W_{\text{proj}} \, \mathbf{z}_q \|_2}$$

For nearest-neighbor search, a cache of precomputed word-only projected vectors (`vocab_size × d_out`) is held in memory. Candidate scoring uses this cache; the query vector includes metadata. Both live in the same $d_{\text{out}}$ space so cosine similarity is well-defined.

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

To add an outcome group (dimension $d_o$):

1. Add `outcome_fields` to `GroupedSample` in `types.h`.
2. Add `outcome2idx_` / `outcomeSize()` to `Vocabulary`.
3. Add `outcome_matrix_` (`outcome_vocab × d_o`) to `FastTextContext` and `Trainer`.
4. Widen $W_{\text{proj}}$ from $d_{\text{out}} \times (d_w + d_p + d_{pr})$ to $d_{\text{out}} \times (d_w + d_p + d_{pr} + d_o)$.
5. Extend `buildConcatVec`, `distributeGrad`, `getCombinedVector`, and the Python loader to handle the new slice.

No structural changes to the HS loop, synchronization protocol, or save/load format beyond the new sizes.

## MIMIC-III Preprocessing

A three-step pipeline in `mimic-iii/` converts raw MIMIC-III CSV tables into the triple-pipe training format. Each step reads the previous step's output, so they must be run in order. The pipeline is designed for low peak RAM usage: step 1 streams via Polars LazyFrames, step 2 processes notes in configurable batches with multiprocessing, and step 3 operates on compact numpy index arrays rather than copying strings.

**Prerequisites:**

```bash
cd mimic-iii
pip install -r requirements.txt   # polars, pyarrow, nltk, numpy, tqdm
```

MIMIC-III source files required in the working directory (or specify `--data-dir`): `ADMISSIONS.csv`, `PATIENTS.csv`, `CAREGIVERS.csv`, and `NOTEEVENTS.parquet`. Convert NOTEEVENTS from CSV first if needed:

```bash
python3 -c "import polars as pl; pl.read_csv('NOTEEVENTS.csv').write_parquet('NOTEEVENTS.parquet')"
```

### Step 1: Merge Source Tables (`01_merge.py`)

Joins ADMISSIONS, PATIENTS, CAREGIVERS, and NOTEEVENTS into a single parquet with context columns and raw note text. Uses Polars `sink_parquet()` for streaming execution — the full joined table is never resident in RAM.

Derived columns: MeSH age category (from admission date and DOB), death flag. All metadata columns are lowercased with spaces replaced by underscores.

```bash
python3 01_merge.py [--data-dir /path/to/csvs] [--out mimic-iii-merged.parquet]
```

**Output columns:** `SUBJECT_ID`, `ROW_ID`, `MeSH`, `GENDER`, `ETHNICITY`, `LANGUAGE`, `RELIGION`, `MARITAL_STATUS`, `INSURANCE`, `CG_TITLE`, `ADMISSION_TYPE`, `TEXT`.

### Step 2: Sentence Segmentation (`02_to_sentences.py`)

Reads the merged parquet and runs NLP processing on each note: NLTK sentence tokenization, de-identification tag stripping (`[** ... **]`), lowercasing, digit removal, non-alpha filtering, and minimum length filtering. Processing is parallelized across worker processes and batched to control memory usage. Output is written incrementally via PyArrow's `ParquetWriter`.

```bash
python3 02_to_sentences.py [--input mimic-iii-merged.parquet] \
                           [--out mimic-iii-sents.parquet]    \
                           [--batch-size 5000]                \
                           [--workers N]                      \
                           [--min-len 20]
```

**Output columns:** `SUBJECT_ID`, `ROW_ID`, `PatientGroup`, `ProviderGroup`, `Sentences` (list of cleaned sentence strings). Patient and provider group strings are pre-computed so step 3 needs no further NLP work.

### Step 3: Generate Training Text (`03_to_text.py`)

Explodes the sentences parquet into the final triple-pipe text file, one sentence per line, shuffled. Operates on compact numpy index arrays rather than copying string data — peak RAM is the string data itself plus lightweight int32/int16 index arrays.

```bash
# Default: all notes, shuffled
python3 03_to_text.py [--input mimic-iii-sents.parquet] [--out mimic-iii-sents.txt]

# Bootstrap sample (two-stage hierarchical, fully reproducible)
python3 03_to_text.py --bootstrap-seed 42 --out mimic-iii-sents-boot42.txt

# Default dataset with reproducible shuffle
python3 03_to_text.py --shuffle-seed 7
```

**Bootstrap mode** (`--bootstrap-seed N`) implements a two-stage hierarchical bootstrap that respects the correlation structure of notes within patients: stage 1 resamples patient IDs with replacement, stage 2 resamples each sampled patient's notes with replacement (preserving original note count). This is the statistically correct bootstrap for clustered data. Bootstrap and shuffle seeds use separate RNG generators and are fully orthogonal. `run_bootstrap.sh` drives this mode in a loop to build a full ensemble.

**Output format** (one line per sentence):

```
elderly male white english medicare ||| attending emergency ||| the patient presented with dyspnea
```

Patient columns: MeSH age category, gender, ethnicity, language, religion, marital status, insurance.
Provider columns: caregiver title, admission type.

### Full Pipeline

```bash
cd mimic-iii
python3 01_merge.py                          # -> mimic-iii-merged.parquet
python3 02_to_sentences.py                   # -> mimic-iii-sents.parquet
python3 03_to_text.py                        # -> mimic-iii-sents.txt
cd ..
./train mimic-iii/mimic-iii-sents.txt model.bin
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
├── fasttext_context.h         # Core class definition
├── fasttext_context.cpp       # Training orchestration & inference dispatch
├── types.h                    # GroupedSample, HuffmanNode, constants
├── matrix.h                   # Matrix class with mulVec/mulVecTranspose/addOuterProduct
├── vocabulary.h/cpp           # Three-group vocabulary (word/patient/provider)
├── trainer.h/cpp              # Training loop, hybrid SGD, projection backward pass
├── inference.h/cpp            # Concat+projection composition & NN search
├── train.cpp                  # CLI: training
├── query.cpp                  # CLI: nearest-neighbor queries
├── compare.cpp                # CLI: vector comparison
├── fasttext_context.py        # Python loader (numpy only)
├── generate_demo_sentences.py # Synthetic training data generator (triple-pipe format)
├── run_bootstrap.sh           # Train original model + N bootstrap replicates
├── compare_bootstrap.py       # 95% CIs on similarity via the bootstrap ensemble
├── dump_metadata.py           # Export patient/provider vocabularies to text files
├── Makefile
├── mimic-iii/
│   ├── 01_merge.py            # Step 1: join source tables -> merged parquet
│   ├── 02_to_sentences.py     # Step 2: NLP sentence segmentation -> sentences parquet
│   ├── 03_to_text.py          # Step 3: explode & shuffle -> triple-pipe training text
│   └── requirements.txt       # Python dependencies for preprocessing
└── README.md
```

## License

MIT License. See LICENSE for details.
"""
Step 3 of 3: Generate FastTextContext training text file from sentences parquet.

Reads:
    mimic-iii-sents.parquet   (written by 02_to_sentences.py)

Writes:
    mimic-iii-sents.txt (default) or a named output file.
    Each line: <PatientGroup> ||| <ProviderGroup> ||| <sentence>

Modes:
    Default  -- all notes, sentences exploded and shuffled.

    Bootstrap (--bootstrap-seed N) -- two-stage hierarchical bootstrap:
        Stage 1: sample len(unique_patients) patient IDs with replacement.
        Stage 2: for each sampled patient instance, sample that patient's
                 notes with replacement (same count as their original note
                 count), preserving intra-patient correlation structure.
        The resulting sentence rows are then shuffled and written.

        This is the statistically correct bootstrap for hierarchical data
        where notes within a patient are not independent.

Shuffling:
    Rows are always shuffled before writing.  Use --shuffle-seed to make the
    shuffle reproducible independently of the bootstrap seed.  If
    --bootstrap-seed is provided and --shuffle-seed is omitted, the bootstrap
    seed also seeds the shuffle (full reproducibility).  If neither is
    provided, the shuffle is non-deterministic.

Memory strategy:
    The parquet is loaded once and immediately converted to plain Python lists,
    freeing the Polars DataFrame.  All sampling and shuffling operates on
    compact numpy int32/int16 index arrays — no string copies, no intermediate
    DataFrames.  Peak RAM is roughly:
        string data  +  2 × (total_sentences × 6 bytes)   [index arrays]
    The string data itself is unavoidable for a global shuffle.

Run:
    # Default dataset
    python3 03_to_text.py [--input mimic-iii-sents.parquet] [--out mimic-iii-sents.txt]

    # Bootstrap sample, fully reproducible
    python3 03_to_text.py --bootstrap-seed 42 --out mimic-iii-sents-boot42.txt

    # Default dataset with reproducible shuffle
    python3 03_to_text.py --shuffle-seed 7
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl


# ------------------------------------------------------------------
# Index-pair builders.
# Each returns two parallel numpy arrays:
#   note_idx  (int32) — index into the note-level lists
#   sent_idx  (int16) — position within that note's sentence list
# No string data is duplicated; writes dereference into the source lists.
# ------------------------------------------------------------------

def _index_pairs_default(sentences_col: list) -> tuple[np.ndarray, np.ndarray]:
    """One index pair per sentence, covering every note."""
    total = sum(len(s) for s in sentences_col)
    note_idx = np.empty(total, dtype=np.int32)
    sent_idx = np.empty(total, dtype=np.int16)
    pos = 0
    for ni, sents in enumerate(sentences_col):
        n = len(sents)
        note_idx[pos:pos + n] = ni
        sent_idx[pos:pos + n] = np.arange(n, dtype=np.int16)
        pos += n
    return note_idx, sent_idx


def _index_pairs_bootstrap(
    sentences_col: list,
    subject_ids: list,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-stage hierarchical bootstrap index pairs.

    Stage 1 — resample patients (unique SUBJECT_IDs) with replacement.
    Stage 2 — for each sampled patient instance, resample that patient's
               notes with replacement (preserving original note count).
    """
    # Build pid → [note_indices] mapping in a single pass.
    pid_to_indices: dict[int, list[int]] = defaultdict(list)
    for ni, pid in enumerate(subject_ids):
        pid_to_indices[int(pid)].append(ni)

    patient_ids = np.array(list(pid_to_indices.keys()), dtype=np.int64)
    n_patients  = len(patient_ids)

    # Stage 1: bootstrap patients.
    sampled_pids = rng.choice(patient_ids, size=n_patients, replace=True)

    # Stage 2: for each sampled patient, bootstrap notes and collect sampled note indices.
    sampled_note_indices = []
    for pid in sampled_pids:
        orig = pid_to_indices[int(pid)]
        n    = len(orig)
        for i in rng.integers(0, n, size=n):
            sampled_note_indices.append(orig[i])

    # Pre-size the output arrays to avoid repeated reallocation.
    # Compute actual total sentences from the sampled notes.
    total = sum(len(sentences_col[ni]) for ni in sampled_note_indices)
    note_idx = np.empty(total, dtype=np.int32)
    sent_idx = np.empty(total, dtype=np.int16)

    pos = 0
    for ni in sampled_note_indices:
        sents = sentences_col[ni]
        ns    = len(sents)
        note_idx[pos:pos + ns] = ni
        sent_idx[pos:pos + ns] = np.arange(ns, dtype=np.int16)
        pos += ns

    return note_idx, sent_idx


# ------------------------------------------------------------------
# Writer.
# ------------------------------------------------------------------

def _write(
    out_path: str,
    note_idx: np.ndarray,
    sent_idx: np.ndarray,
    patient_groups: list,
    provider_groups: list,
    sentences_col: list,
) -> int:
    written = 0
    with open(out_path, 'w', encoding='utf-8') as fp:
        for ni, si in zip(note_idx.tolist(), sent_idx.tolist()):
            sent = sentences_col[ni][si]
            fp.write(f"{patient_groups[ni]} ||| {provider_groups[ni]} ||| {sent}\n")
            written += 1
    return written


# ------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MIMIC-III step 3: generate training text file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input',          default='mimic-iii-sents.parquet',
                        help='Sentences parquet from 02_to_sentences.py (default: mimic-iii-sents.parquet)')
    parser.add_argument('--out',            default='mimic-iii-sents.txt',
                        help='Output training text file (default: mimic-iii-sents.txt)')
    parser.add_argument('--bootstrap-seed', type=int, default=None, metavar='N',
                        help='Enable two-stage bootstrap sampling with this random seed.')
    parser.add_argument('--shuffle-seed',   type=int, default=None, metavar='N',
                        help='Seed for row shuffle (default: bootstrap seed if set, else random).')
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        print("Run 02_to_sentences.py first.", file=sys.stderr)
        sys.exit(1)

    # Bootstrap sampling and shuffle use separate generators so the two seeds
    # are fully orthogonal — changing one never affects the other's output.
    shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.bootstrap_seed
    boot_rng     = np.random.default_rng(args.bootstrap_seed)
    shuffle_rng  = np.random.default_rng(shuffle_seed)

    mode = (f"bootstrap (seed={args.bootstrap_seed})"
            if args.bootstrap_seed is not None else "default (all notes)")

    print("=== MIMIC-III Step 3: Generate Training Text ===")
    print(f"  Input        : {args.input}")
    print(f"  Output       : {args.out}")
    print(f"  Mode         : {mode}")
    print(f"  Shuffle seed : {shuffle_seed if shuffle_seed is not None else 'random'}")
    print()

    # ------------------------------------------------------------------
    # Load parquet → Python native lists, then free the DataFrame.
    # Working with plain lists avoids Polars string-object overhead and
    # prevents any accidental DataFrame copies downstream.
    # ------------------------------------------------------------------
    print("Loading sentences parquet...")
    t0  = time.time()
    df  = pl.read_parquet(args.input)

    subject_ids     = df['SUBJECT_ID'].to_list()
    patient_groups  = df['PatientGroup'].to_list()
    provider_groups = df['ProviderGroup'].to_list()
    sentences_col   = df['Sentences'].to_list()
    n_notes         = len(subject_ids)
    n_patients      = len(set(subject_ids))
    del df  # release Polars DataFrame memory before building index arrays

    print(f"  {n_notes:,} notes  |  {n_patients:,} unique patients")
    print()

    # ------------------------------------------------------------------
    # Build index arrays.
    # ------------------------------------------------------------------
    if args.bootstrap_seed is not None:
        print("Running two-stage bootstrap...")
        note_idx, sent_idx = _index_pairs_bootstrap(sentences_col, subject_ids, boot_rng)
    else:
        print("Indexing all notes...")
        note_idx, sent_idx = _index_pairs_default(sentences_col)

    # Shuffle index pairs in place (no string copies).
    perm     = shuffle_rng.permutation(len(note_idx))
    note_idx = note_idx[perm]
    sent_idx = sent_idx[perm]
    del perm

    # ------------------------------------------------------------------
    # Write.
    # ------------------------------------------------------------------
    total_sents = len(note_idx)
    print(f"Writing {total_sents:,} sentence rows to {args.out}...")
    written = _write(args.out, note_idx, sent_idx, patient_groups, provider_groups, sentences_col)

    elapsed = time.time() - t0
    out_mb  = Path(args.out).stat().st_size / 1_048_576

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Sentences written : {written:,}")
    print(f"  Output file       : {args.out}  ({out_mb:.1f} MB)")


if __name__ == '__main__':
    main()

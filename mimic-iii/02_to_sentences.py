"""
Step 2 of 2: NLP sentence segmentation and streaming export to training text.

Reads:
    mimic-iii-merged.parquet   (written by 01_merge.py)

Writes:
    mimic-iii-sents.txt   -- triple-pipe format for FastTextContext training
                             <PatientGroup> ||| <ProviderGroup> ||| <sentence>

Memory strategy:
    The merged parquet is scanned lazily and processed in batches of --batch-size
    rows (default 5000). Each batch is collected, NLP-processed, and its sentences
    are written to disk before the next batch is loaded. The exploded sentence list
    is never accumulated across batches, so peak RAM is:

        ~batch_size × avg_note_size × 2  (raw text + processed sentences)

    For batch_size=5000 and MIMIC-III average note length (~600 words) this is
    roughly 50–150 MB, independent of the full dataset size.

Run:
    python3 02_to_sentences.py [--input mimic-iii-merged.parquet] [--out mimic-iii-sents.txt] [--batch-size 5000] [--workers N]
"""

import argparse
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nltk
from tqdm import tqdm
import polars as pl

nltk.download('punkt_tab', quiet=True)

# ------------------------------------------------------------------
# Column definitions (must match 01_merge.py output).
# ------------------------------------------------------------------
PATIENT_COLS = [
    'MeSH', 'GENDER', 'ETHNICITY', 'LANGUAGE',
    'RELIGION', 'MARITAL_STATUS', 'INSURANCE',
]
PROVIDER_COLS = [
    'CG_TITLE', 'ADMISSION_TYPE',
]

# ------------------------------------------------------------------
# NLP helpers (module-level so multiprocessing workers can import them).
# ------------------------------------------------------------------
_DIGIT_PATTERN     = re.compile(r'\d')
_NON_ALPHA_PATTERN = re.compile(r'[^a-z-/\\]')
_TAG_PATTERN       = re.compile(r'\[\*\*.*?\*\*\]')
_WHITESPACE        = re.compile(r'\s+')


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = _WHITESPACE.sub(' ', text).strip()
    return ''.join(ch for ch in text if ch.isprintable())


def _post_process(sentences: list) -> list:
    out = []
    for sent in sentences:
        cleaned = _TAG_PATTERN.sub('', sent).lower()
        words = []
        for word in cleaned.split():
            if _DIGIT_PATTERN.search(word):
                continue
            alpha = _NON_ALPHA_PATTERN.sub('', word)
            if alpha:
                words.append(alpha)
        result = ' '.join(words)
        if result:
            out.append(result)
    return out


def process_text(text: str) -> list:
    """Full NLP pipeline for a single note. Called in worker processes."""
    cleaned   = _clean_text(text)
    sentences = nltk.sent_tokenize(cleaned) if cleaned else []
    return _post_process(sentences)


# ------------------------------------------------------------------
# Context string builder.
# ------------------------------------------------------------------
def _group_str(row: dict, cols: list) -> str:
    parts = []
    for col in cols:
        val = row.get(col)
        if val is None:
            continue
        s = str(val).replace(' ', '_')
        if 'unknown' not in s.lower():
            parts.append(s)
    return ' '.join(parts)


# ------------------------------------------------------------------
# Batch processor: NLP + write.
# ------------------------------------------------------------------
def process_and_write_batch(
    batch: pl.DataFrame,
    pool: Pool,
    fp,
    min_len: int,
) -> int:
    """
    Process one batch: run NLP in parallel, write qualifying sentences to fp.
    Returns the number of sentences written.
    """
    texts = batch['TEXT'].to_list()

    # imap preserves order so sentence lists align with dataframe rows.
    sentence_lists = list(pool.imap(process_text, texts, chunksize=256))

    written = 0
    all_cols = PATIENT_COLS + PROVIDER_COLS

    for row_idx, sentences in enumerate(sentence_lists):
        if not sentences:
            continue

        # Build context strings once per note, not once per sentence.
        row = {col: batch[col][row_idx] for col in all_cols}
        patient_str  = _group_str(row, PATIENT_COLS)
        provider_str = _group_str(row, PROVIDER_COLS)
        prefix       = f"{patient_str} ||| {provider_str} ||| "

        for sent in sentences:
            if len(sent) < min_len:
                continue
            fp.write(prefix + sent + '\n')
            written += 1

    return written


# ------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='MIMIC-III step 2: NLP sentence export.')
    parser.add_argument('--input',      default='mimic-iii-merged.parquet',
                        help='Merged parquet from 01_merge.py (default: mimic-iii-merged.parquet)')
    parser.add_argument('--out',        default='mimic-iii-sents.txt',
                        help='Output training text file (default: mimic-iii-sents.txt)')
    parser.add_argument('--batch-size', type=int, default=5_000,
                        help='Notes per batch (default: 5000). Lower = less RAM, more overhead.')
    parser.add_argument('--workers',    type=int, default=max(1, cpu_count() - 1),
                        help='NLP worker processes (default: cpu_count - 1)')
    parser.add_argument('--min-len',    type=int, default=20,
                        help='Minimum sentence character length to keep (default: 20)')
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        print("Run 01_merge.py first.", file=sys.stderr)
        sys.exit(1)

    # Count total rows cheaply without loading the data.
    total_rows = pl.scan_parquet(args.input).select(pl.len()).collect().item()
    total_batches = (total_rows + args.batch_size - 1) // args.batch_size

    print("=== MIMIC-III Step 2: NLP + Export ===")
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.out}")
    print(f"  Total rows : {total_rows:,}")
    print(f"  Batch size : {args.batch_size:,}  ({total_batches} batches)")
    print(f"  Workers    : {args.workers}")
    print(f"  Min sent.  : {args.min_len} chars")
    print()

    lf = pl.scan_parquet(args.input)
    t0 = time.time()
    total_written = 0

    # One Pool created here and reused across all batches — avoids repeated
    # process startup overhead which would dominate for small batch sizes.
    with Pool(processes=args.workers) as pool, \
         open(args.out, 'w', encoding='utf-8') as fp:

        for batch_idx in tqdm(range(total_batches), desc='Batches', unit='batch'):
            offset = batch_idx * args.batch_size

            # Collect only this slice — peak RAM is O(batch_size).
            batch = lf.slice(offset, args.batch_size).collect()

            written = process_and_write_batch(batch, pool, fp, args.min_len)
            total_written += written

            # Explicitly delete batch to release RAM before the next slice.
            del batch

    elapsed = time.time() - t0
    out_mb  = Path(args.out).stat().st_size / 1_048_576

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Sentences written : {total_written:,}")
    print(f"  Output file       : {args.out}  ({out_mb:.1f} MB)")


if __name__ == '__main__':
    main()

"""
Step 2 of 3: NLP sentence segmentation → sentences parquet.

Reads:
    mimic-iii-merged.parquet   (written by 01_merge.py)

Writes:
    mimic-iii-sents.parquet   -- one row per note, columns:
                                  SUBJECT_ID, ROW_ID,
                                  PatientGroup, ProviderGroup,
                                  Sentences (List[str])

    PatientGroup / ProviderGroup are pre-computed context strings used by
    03_to_text.py when writing the final training file.  Sentences have already
    been cleaned, de-identified tag-stripped, lowercased, digit-filtered, and
    filtered to --min-len characters, so 03_to_text.py can explode and write
    with no further NLP work.

Memory strategy:
    The merged parquet is scanned lazily and processed in batches of
    --batch-size rows.  Each batch is collected, NLP-processed in parallel
    worker processes, and its result rows are written to the output parquet
    via PyArrow's incremental ParquetWriter before the next batch is loaded.
    Peak RAM is O(batch_size × avg_note_size).

Run:
    python3 02_to_sentences.py [--input mimic-iii-merged.parquet]
                               [--out   mimic-iii-sents.parquet]
                               [--batch-size 5000]
                               [--workers N]
                               [--min-len 20]
"""

import argparse
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nltk
import pyarrow as pa
import pyarrow.parquet as pq
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

# Set by pool initializer in each worker process.
_MIN_LEN: int = 20


def _init_worker(min_len: int) -> None:
    global _MIN_LEN
    _MIN_LEN = min_len


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
        if result and len(result) >= _MIN_LEN:
            out.append(result)
    return out


def process_text(text: str) -> list:
    """Full NLP pipeline for a single note.  Called in worker processes."""
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
# Output schema (fixed so PyArrow writer stays consistent across batches).
# ------------------------------------------------------------------
OUTPUT_SCHEMA = pa.schema([
    pa.field('SUBJECT_ID',    pa.int64()),
    pa.field('ROW_ID',        pa.int64()),
    pa.field('PatientGroup',  pa.string()),
    pa.field('ProviderGroup', pa.string()),
    pa.field('Sentences',     pa.list_(pa.string())),
])


def process_batch(batch: pl.DataFrame, pool: Pool) -> pa.Table:
    """
    Run NLP on one batch of notes.
    Returns a PyArrow table ready to be written by ParquetWriter.
    Notes that produce zero qualifying sentences are dropped.
    """
    texts          = batch['TEXT'].to_list()
    sentence_lists = list(pool.imap(process_text, texts, chunksize=256))

    all_cols      = PATIENT_COLS + PROVIDER_COLS
    subject_ids   = []
    row_ids       = []
    patient_groups  = []
    provider_groups = []
    sentences_col   = []

    for i, sentences in enumerate(sentence_lists):
        if not sentences:
            continue
        row = {col: batch[col][i] for col in all_cols}
        subject_ids.append(batch['SUBJECT_ID'][i])
        row_ids.append(batch['ROW_ID'][i])
        patient_groups.append(_group_str(row, PATIENT_COLS))
        provider_groups.append(_group_str(row, PROVIDER_COLS))
        sentences_col.append(sentences)

    return pa.table(
        {
            'SUBJECT_ID':    pa.array(subject_ids,    type=pa.int64()),
            'ROW_ID':        pa.array(row_ids,        type=pa.int64()),
            'PatientGroup':  pa.array(patient_groups, type=pa.string()),
            'ProviderGroup': pa.array(provider_groups, type=pa.string()),
            'Sentences':     pa.array(sentences_col,  type=pa.list_(pa.string())),
        },
        schema=OUTPUT_SCHEMA,
    )


# ------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='MIMIC-III step 2: NLP sentence segmentation.')
    parser.add_argument('--input',      default='mimic-iii-merged.parquet',
                        help='Merged parquet from 01_merge.py (default: mimic-iii-merged.parquet)')
    parser.add_argument('--out',        default='mimic-iii-sents.parquet',
                        help='Output sentences parquet (default: mimic-iii-sents.parquet)')
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

    total_rows    = pl.scan_parquet(args.input).select(pl.len()).collect().item()
    total_batches = (total_rows + args.batch_size - 1) // args.batch_size

    print("=== MIMIC-III Step 2: Sentence Segmentation ===")
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.out}")
    print(f"  Total rows : {total_rows:,}")
    print(f"  Batch size : {args.batch_size:,}  ({total_batches} batches)")
    print(f"  Workers    : {args.workers}")
    print(f"  Min sent.  : {args.min_len} chars")
    print()

    lf           = pl.scan_parquet(args.input)
    writer       = None
    total_notes  = 0
    total_sents  = 0
    t0           = time.time()

    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(args.min_len,)) as pool:

        for batch_idx in tqdm(range(total_batches), desc='Batches', unit='batch'):
            offset = batch_idx * args.batch_size
            batch  = lf.slice(offset, args.batch_size).collect()

            result = process_batch(batch, pool)
            del batch

            if result.num_rows == 0:
                continue

            total_notes += result.num_rows
            total_sents += sum(len(s) for s in result['Sentences'].to_pylist())

            if writer is None:
                writer = pq.ParquetWriter(args.out, OUTPUT_SCHEMA, compression='zstd')
            writer.write_table(result)

    if writer:
        writer.close()

    elapsed = time.time() - t0
    out_mb  = Path(args.out).stat().st_size / 1_048_576 if Path(args.out).exists() else 0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Notes with sentences : {total_notes:,}")
    print(f"  Total sentences      : {total_sents:,}")
    print(f"  Output file          : {args.out}  ({out_mb:.1f} MB)")
    print("Run 03_to_text.py next to generate the training text file.")


if __name__ == '__main__':
    main()

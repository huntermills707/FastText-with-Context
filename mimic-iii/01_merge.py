"""
Step 1 of 2: Join MIMIC-III source tables and write a lean merged parquet.

Reads:
    ADMISSIONS.csv
    PATIENTS.csv
    CAREGIVERS.csv
    NOTEEVENTS.parquet  (convert from CSV with: python3 -c "import polars as pl; pl.read_csv('NOTEEVENTS.csv').write_parquet('NOTEEVENTS.parquet')")

Writes:
    mimic-iii-merged.parquet   -- context columns + raw TEXT, one row per note

Memory strategy:
    Everything is expressed as a Polars LazyFrame. sink_parquet() executes the
    plan in a streaming fashion so the full joined table is never resident in RAM.
    Peak usage is roughly one row-group worth of data at a time (~50–100 MB).

Run:
    python3 01_merge.py [--data-dir /path/to/csvs] [--out mimic-iii-merged.parquet]
"""

import argparse
import sys
import time
from pathlib import Path
import polars as pl


# Patient-level columns kept in the output.
PATIENT_COLS = [
    'MeSH', 'GENDER', 'ETHNICITY', 'LANGUAGE',
    'RELIGION', 'MARITAL_STATUS', 'INSURANCE',
]

# Provider-level columns kept in the output.
PROVIDER_COLS = [
    'CG_TITLE', 'ADMISSION_TYPE',
]


def build_lazy_pipeline(data_dir: Path) -> pl.LazyFrame:
    """Build the full join + transformation pipeline as a LazyFrame."""

    admissions = pl.scan_csv(data_dir / 'ADMISSIONS.csv')
    patients   = pl.scan_csv(data_dir / 'PATIENTS.csv')
    caregivers = pl.scan_csv(data_dir / 'CAREGIVERS.csv')
    notes      = pl.scan_parquet(data_dir / 'NOTEEVENTS.parquet')

    # ------------------------------------------------------------------
    # Admissions: derive DEATH flag, select needed columns.
    # ------------------------------------------------------------------
    admissions = admissions.with_columns(
        pl.when(pl.col('HOSPITAL_EXPIRE_FLAG') == 1)
          .then(pl.lit('Dead'))
          .otherwise(pl.lit('Alive'))
          .alias('DEATH')
    ).select([
        'HADM_ID', 'ADMITTIME', 'ADMISSION_TYPE',
        'INSURANCE', 'LANGUAGE', 'RELIGION',
        'MARITAL_STATUS', 'ETHNICITY', 'DEATH',
    ])

    caregivers = caregivers.select(['CGID', 'LABEL']).rename({'LABEL': 'CG_TITLE'})

    notes = notes.select(['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CGID', 'TEXT'])

    # ------------------------------------------------------------------
    # Cast join keys to consistent types before joining.
    # ------------------------------------------------------------------
    notes = notes.with_columns([
        pl.col('CGID').cast(pl.Int64),
        pl.col('HADM_ID').cast(pl.Int64),
        pl.col('SUBJECT_ID').cast(pl.Int64),
    ])

    # ------------------------------------------------------------------
    # Joins.
    # ------------------------------------------------------------------
    notes = notes.join(caregivers,  on='CGID',       how='left')
    notes = notes.join(patients,    on='SUBJECT_ID',  how='left')
    notes = notes.join(admissions,  on='HADM_ID',     how='left')

    # ------------------------------------------------------------------
    # Age calculation → MeSH category.
    # ------------------------------------------------------------------
    notes = notes.with_columns([
        pl.col('ADMITTIME').str.to_datetime(strict=False),
        pl.col('DOB').str.to_datetime(strict=False),
    ])

    notes = notes.with_columns(
        ((pl.col('ADMITTIME') - pl.col('DOB')).dt.total_days() / 365.0)
        .floor()
        .alias('age')
    )

    notes = notes.with_columns(
        pl.when(pl.col('age') >= 80).then(pl.lit('Elderly'))
        .when(pl.col('age')  >= 65).then(pl.lit('Aged'))
        .when(pl.col('age')  >= 45).then(pl.lit('Middle_Aged'))
        .when(pl.col('age')  >= 24).then(pl.lit('Adult'))
        .when(pl.col('age')  >= 19).then(pl.lit('Young_Adult'))
        .when(pl.col('age')  >= 13).then(pl.lit('Adolescent'))
        .when(pl.col('age')  >= 2) .then(pl.lit('Child'))
        .otherwise(pl.lit('Infant'))
        .alias('MeSH')
    )

    # ------------------------------------------------------------------
    # Normalise metadata: lowercase, spaces → underscores.
    # ------------------------------------------------------------------
    all_meta_cols = PATIENT_COLS + PROVIDER_COLS
    notes = notes.with_columns([
        pl.col(col).cast(pl.Utf8).str.to_lowercase().str.replace_all(' ', '_')
        for col in all_meta_cols
    ])

    # Keep only the columns the next script needs.
    return notes.select(['SUBJECT_ID', 'ROW_ID'] + all_meta_cols + ['TEXT'])


def main():
    parser = argparse.ArgumentParser(description='MIMIC-III step 1: merge source tables.')
    parser.add_argument('--data-dir', default='.', help='Directory containing MIMIC-III source files (default: .)')
    parser.add_argument('--out', default='mimic-iii-merged.parquet', help='Output parquet path (default: mimic-iii-merged.parquet)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = args.out

    required = ['ADMISSIONS.csv', 'PATIENTS.csv', 'CAREGIVERS.csv', 'NOTEEVENTS.parquet']
    missing  = [f for f in required if not (data_dir / f).exists()]
    if missing:
        print(f"ERROR: missing source files in '{data_dir}': {missing}", file=sys.stderr)
        print("Convert NOTEEVENTS.csv first:", file=sys.stderr)
        print("  python3 -c \"import polars as pl; pl.read_csv('NOTEEVENTS.csv').write_parquet('NOTEEVENTS.parquet')\"",
              file=sys.stderr)
        sys.exit(1)

    print("=== MIMIC-III Step 1: Merge ===")
    print(f"  Source dir : {data_dir}")
    print(f"  Output     : {out_path}")
    print()
    print("Building lazy pipeline...")

    pipeline = build_lazy_pipeline(data_dir)

    print("Streaming join result to parquet (no full collect)...")
    print("This streams one row-group at a time — peak RAM is low regardless of dataset size.")
    t0 = time.time()

    # sink_parquet executes the lazy plan in a streaming fashion.
    # row_group_size controls how many rows are held in RAM at once during the write.
    pipeline.sink_parquet(
        out_path,
        compression='zstd',
        row_group_size=50_000,
    )

    elapsed = time.time() - t0
    size_mb = Path(out_path).stat().st_size / 1_048_576

    print(f"\nDone in {elapsed:.1f}s  →  {out_path}  ({size_mb:.1f} MB)")
    print("Run 02_to_sentences.py next to segment notes into sentences.")


if __name__ == '__main__':
    main()

import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import polars as pl
import nltk

# Download once
nltk.download('punkt_tab', quiet=True)

N_WORKERS = max(1, cpu_count() - 1)

DIGIT_PATTERN = re.compile(r'\d')
NON_ALPHA_PATTERN = re.compile(r'[^a-z]')
TAG_PATTERN = re.compile(r'\[\*\*.*?\*\*\]')

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = ''.join(ch for ch in text if ch.isprintable())
    return text

def segment_sentences(text):
    """Segment text into sentences."""
    if not text:
        return []
    return nltk.sent_tokenize(text)

def post_process_sentences(sentences):
    """
    Post-segmenting transformations:
    1. Strip tags [**TAG_TEXT**]
    2. Lowercase
    3. Drop words with numbers
    4. Keep only alpha chars
    """
    processed = []
    for sentence in sentences:
        # Strip tags
        cleaned = TAG_PATTERN.sub('', sentence).lower()
        words = []
        for word in cleaned.split():
            if DIGIT_PATTERN.search(word):
                continue
            alpha_word = NON_ALPHA_PATTERN.sub('', word)
            if alpha_word:
                words.append(alpha_word)
        
        cleaned = ' '.join(words)
        if cleaned:
            processed.append(cleaned)
    return processed

def process_text(text):
    """Full pipeline: clean, segment, and post-process."""
    cleaned = clean_text(text)
    sentences = segment_sentences(cleaned)
    return post_process_sentences(sentences)

def process_column_parallel(df, column_name, n_workers=None):
    """
    Process a column in parallel using Polars.
    Returns a list of lists (sentences per row).
    """
    if n_workers is None:
        n_workers = N_WORKERS

    # Extract column as a list of strings
    texts = df[column_name].to_list()
    total = len(texts)

    print(f"Processing {total} rows with {n_workers} workers...")

    with Pool(n_workers) as pool:
        # imap preserves order, essential for DataFrame alignment
        results = list(tqdm(
            pool.imap(process_text, texts),
            total=total,
            desc=f"Processing '{column_name}'",
            unit="rows"
        ))

    return results

if __name__ == "__main__":
    print('MIMIC-iii FastText Context Data Preparation Script')
    print('Requires: ADMISSIONS.csv, PATIENTS.csv, CAREGIVERS.csv, and NOTEEVENTS.parquet (csv converted to parquet)')
    
    # Load Data
    print('Loading Data.')
    admission_df = pl.scan_csv('ADMISSIONS.csv')
    patients_df = pl.scan_csv('PATIENTS.csv')
    caregivers_df = pl.scan_csv('CAREGIVERS.csv')
    notes_df = pl.scan_parquet('NOTEEVENTS.parquet')

    # Create DEATH column based on HOSPITAL_EXPIRE_FLAG
    admission_df = admission_df.with_columns(
        pl.when(pl.col('HOSPITAL_EXPIRE_FLAG') == 1)
          .then(pl.lit('Dead'))
          .otherwise(pl.lit('Alive'))
          .alias('DEATH')
    )

    # Select needed columns
    admission_cols = [
        'HADM_ID', 'ADMITTIME', 'ADMISSION_TYPE',
        'INSURANCE', 'LANGUAGE', 'RELIGION',
        'MARITAL_STATUS', 'ETHNICITY', 'DEATH'
    ]
    admission_df = admission_df.select(admission_cols)
    caregivers_cols = ['CGID', 'LABEL']
    caregivers_df = caregivers_df.select(caregivers_cols)
    caregivers_df = caregivers_df.rename({'LABEL': 'CG_TITLE'})
    notes_cols = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CGID', 'TEXT']
    notes_df = notes_df.select(notes_cols)

    # Merge
    notes_df = notes_df.with_columns([
        pl.col("CGID").cast(pl.Int64),
        pl.col("HADM_ID").cast(pl.Int64),
        pl.col("SUBJECT_ID").cast(pl.Int64),
    ])
    notes_df = notes_df.join(caregivers_df, on='CGID', how='left')
    notes_df = notes_df.join(patients_df, on='SUBJECT_ID', how='left')
    notes_df = notes_df.join(admission_df, on='HADM_ID', how='left')

    # Calculate age
    notes_df = notes_df.with_columns([
        pl.col('ADMITTIME').str.to_datetime(),
        pl.col('DOB').str.to_datetime()
    ])
    notes_df = notes_df.with_columns(
        ((pl.col('ADMITTIME') - pl.col('DOB')).dt.total_days() / 365).floor().alias('age')
    )

    # Assign MeSH Categories
    notes_df = notes_df.with_columns(
        pl.when(pl.col('age') >= 80).then(pl.lit('Elderly'))
        .when(pl.col('age') >= 65).then(pl.lit('Aged'))
        .when(pl.col('age') >= 45).then(pl.lit('Middle Aged'))
        .when(pl.col('age') >= 24).then(pl.lit('Adult'))
        .when(pl.col('age') >= 19).then(pl.lit('Young Adult'))
        .when(pl.col('age') >= 13).then(pl.lit('Adolescent'))
        .when(pl.col('age') >= 2).then(pl.lit('Child'))
        .otherwise(pl.lit('Infant'))
        .alias('MeSH')
    )

    # Further reduce columns
    context_cols = [
        'MeSH', 'GENDER', 'ETHNICITY', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS',
        'CG_TITLE', 'ADMISSION_TYPE', 'INSURANCE', 'DEATH',
    ]

    # Apply lower and replace space with underscore
    exprs = [
        pl.col(col).str.to_lowercase().str.replace_all(' ', '_')
        for col in context_cols
    ]
    notes_df = notes_df.with_columns(exprs)

    final_cols = context_cols + ['TEXT']
    notes_df = notes_df.select(final_cols)

    print('Selecting Data.')

    # Collect to memory for processing
    notes_df = notes_df.collect()

    # Run the heavy NLP lifting
    print("Starting NLP processing...")
    processed_sentences = process_column_parallel(notes_df, 'TEXT')

    # Add the processed list back to the dataframe
    text_series = pl.Series('TEXT', processed_sentences)
    notes_df = notes_df.with_columns(text_series)

    print('Preparing data for export (exploding sentences)...')
    
    # Explode the TEXT column
    exploded_df = notes_df.explode('TEXT')
    
    # 2. Filter out short sentences and nulls in one go
    exploded_df = exploded_df.filter(
        (pl.col('TEXT').cast(pl.Utf8).is_not_null()) & 
        (pl.col('TEXT').cast(pl.Utf8).str.len_chars() >= 20)
    )
    
    print(f"Total sentences to write: {len(exploded_df)}")
    print('Exporting Data with progress bar...')

    # Build the context prefix vectorized
    context_exprs = [
        pl.col(col).cast(pl.Utf8).fill_null('').str.replace_all(' ', '_')
        for col in context_cols
    ]
    
    with open('mimic-iii-sents.txt', 'w', encoding='utf-8') as fp:
        # Iterate with progress bar
        for row in tqdm(exploded_df.iter_rows(named=True), total=len(exploded_df), desc="Writing output"):
            # Reconstruct the base string exactly as the original logic intended
            parts = []
            for col in context_cols:
                val = row[col]
                if val is None:
                    continue
                val_str = str(val).replace(' ', '_')
                if 'unknown' not in val_str.lower():
                    parts.append(val_str)
                
            base = '|'.join(parts) + '|'
            sentence = row['TEXT']
            
            if sentence:
                fp.write(base + sentence + '\n')

    print("Processing complete. Output written to mimic-iii-sents.txt")

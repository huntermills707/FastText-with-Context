#!/bin/bash
set -e

# Configuration
INPUT_PARQUET="mimic-iii/mimic-iii-sents.parquet"
ORIGINAL_TEXT="mimic-iii/mimic-iii-sents.txt"
ORIGINAL_MODEL="model.bin"
LR=0.02
WD=1e-5
NUM_BOOTSTRAPS=25

echo "=== Running Original Dataset Training ==="
python3 mimic-iii/03_to_text.py --input "$INPUT_PARQUET" --out "$ORIGINAL_TEXT"
./train -lr $LR -weight-decay $WD "$ORIGINAL_TEXT" "$ORIGINAL_MODEL"

echo "=== Running $NUM_BOOTSTRAPS Bootstraps ==="
for i in $(seq 1 $NUM_BOOTSTRAPS); do
    BOOT_TEXT="mimic-iii/mimic-iii-sents_boot_$i.txt"
    BOOT_MODEL="model_boot_$i.bin"
    SEED=$i # Use the loop index as the seed for repeatability

    echo "--- Bootstrap $i (Seed: $SEED) ---"
    python3 mimic-iii/03_to_text.py --input "$INPUT_PARQUET" --out "$BOOT_TEXT" --bootstrap-seed $SEED
    ./train -lr $LR -weight-decay $WD "$BOOT_TEXT" "$BOOT_MODEL"
    
    # Optionally remove the bootstrap text file to save space
    rm "$BOOT_TEXT"
done

echo "=== All tasks completed ==="

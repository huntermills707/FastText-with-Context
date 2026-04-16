import sys
from itertools import product
import os
import numpy as np
from typing import List, Tuple
from fasttext_context import FastTextContext

def parse_line(line: str) -> Tuple[List[str], List[str], List[str]]:
    """Parse a line with 'words' 'patients' 'providers' format."""
    parts = line.strip().split("|||")
    return (parts[0].strip().split(),
            parts[1].strip().split(),
            parts[2].strip().split())

def calculate_similarity(model: FastTextContext, base: Tuple, target: Tuple) -> float:
    v1 = model.get_combined_vector(base[0], base[1], base[2])
    v2 = model.get_combined_vector(target[0], target[1], target[2])
    return float(np.dot(v1, v2))

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_bootstrap.py <bases_file> <targets_file> [model_prefix]")
        sys.exit(1)

    bases_file = sys.argv[1]
    targets_file = sys.argv[2]
    model_prefix = sys.argv[3] if len(sys.argv) > 3 else "model"

    # Load original model
    original_model_path = f"{model_prefix}.bin"
    if not os.path.exists(original_model_path):
        print(f"Original model not found: {original_model_path}")
        sys.exit(1)
    
    print(f"Loading original model: {original_model_path}")
    orig_model = FastTextContext()
    orig_model.load_model(original_model_path)

    # Find bootstrap models
    boot_models = []
    i = 1
    while True:
        boot_path = f"{model_prefix}_boot_{i}.bin"
        if os.path.exists(boot_path):
            boot_models.append(boot_path)
            i += 1
        else:
            break
    
    print(f"Found {len(boot_models)} bootstrap models.")

    # Read entries
    with open(bases_file, 'r') as f:
        bases = [parse_line(line) for line in f if line.strip()]
    with open(targets_file, 'r') as f:
        targets = [parse_line(line) for line in f if line.strip()]

    # Read similarities
    all_boot_sims = [[] for _ in range(len(bases) * len(targets))]

    for boot_path in boot_models:
        print(f"Loading bootstrap model: {boot_path}")
        temp_model = FastTextContext()
        temp_model.load_model(boot_path)
        for i, (b, t) in enumerate(product(bases, targets)):
            all_boot_sims[i].append(calculate_similarity(temp_model, b, t))
        del temp_model

    print(f"{'Base':<40} | {'Target':<40} | {'Original':>8} | {'95% CI':<20}")
    print("-" * 120)

    for i, (b, t) in enumerate(product(bases, targets)):
        # Original similarity
        orig_sim = calculate_similarity(orig_model, b, t)
        
        boot_sims = all_boot_sims[i]
        if boot_sims:
            boot_sims.sort()
            # 95% CI: 2.5th and 97.5th percentiles
            low = np.percentile(boot_sims, 2.5)
            high = np.percentile(boot_sims, 97.5)
            ci_str = f"[{low:.4f}, {high:.4f}]"
        else:
            ci_str = "N/A"

        base_str = f"{' '.join(b[0]) if b[0] else ''} {b[1][0] if b[1] else ''} {b[2][0] if b[2] else ''}"
        target_str = f"{t[0][0] if t[0] else ''} {t[1][0] if t[1] else ''} {t[2][0] if t[2] else ''}"
        print(f"{base_str[:40]:<40} | {target_str[:40]:<40} | {orig_sim:8.4f} | {ci_str}")

if __name__ == "__main__":
    main()

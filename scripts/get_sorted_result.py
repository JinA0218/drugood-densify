import os
import re
from collections import defaultdict

def sort_and_save_blocks(input_path):
    # Extract input file name
    input_filename = os.path.basename(input_path)

    # Target keys (task names)
    target_keys = [
        "hivprot count", "hivprot bit",
        "dpp4 count", "dpp4 bit",
        "nk1 count", "nk1 bit"
    ]

    # Read full file
    with open(input_path, "r") as f:
        raw_lines = f.readlines()

    # Group all blocks (multiple per key possible)
    blocks_by_key = defaultdict(list)
    current_block = []
    current_key = None

    for line in raw_lines:
        header_match = next((key for key in target_keys if key in line), None)
        if header_match:
            if current_block and current_key:
                blocks_by_key[current_key].append(current_block)
            current_block = [line]
            current_key = header_match
        else:
            current_block.append(line)

    # Append the last block
    if current_block and current_key:
        blocks_by_key[current_key].append(current_block)

    # Extract mean from "last performance mu: ..." line
    def extract_last_mu_mean(block):
        for line in block:
            if line.strip().startswith("last performance mu:"):
                match = re.search(r"last performance mu:\s*([0-9.]+)", line)
                return float(match.group(1)) if match else float('inf')
        return float('inf')

    # Sort blocks within each key and save
    for key, block_list in blocks_by_key.items():
        sorted_blocks = sorted(block_list, key=extract_last_mu_mean)
        dir_name = f"experiments/final/sorted/{key.replace(' ', '_')}"
        os.makedirs(dir_name, exist_ok=True)
        out_path = os.path.join(dir_name, input_filename)
        with open(out_path, "w") as f:
            for block in sorted_blocks:
                f.writelines(block)
                f.write("\n")
        print(f"✅ Sorted: {key} → {out_path}")

# List of input files to process
input_files = [
    '/c2/jinakim/Drug_Discovery_j/experiments/S-results-dsets_mvalid-all-3real-ml0-xmix-mvdefMANIFOLD_MIXUP_BILEVEL4-mNctFalse-RYV1_MANIFOLD_MIXUP_BILEVEL_real_3.txt',
    '/c2/jinakim/Drug_Discovery_j/experiments/S-results-dsets_mvalid-all-3real-ml0-xmix-mvdefMIXUP_BILEVEL4-mNctFalse-RYV1_MIXUP_BILEVEL_real_3.txt',
    '/c2/jinakim/Drug_Discovery_j/experiments/S-results-strans_mvalid-all-3real-ml0-xmix-mvdefMANIFOLD_MIXUP_BILEVEL4-mNctFalse-RYV1_MANIFOLD_MIXUP_BILEVEL_real_3.txt',
    '/c2/jinakim/Drug_Discovery_j/experiments/S-results-strans_mvalid-all-3real-ml0-xmix-mvdefMIXUP_BILEVEL4-mNctFalse-RYV1_MIXUP_BILEVEL_real_3.txt'
    # '/c2/jinakim/Drug_Discovery_j/experiments/S-results-dsets_mvalid-all-3real-ml0-xmix-mvdefMANIFOLD_MIXUP_BILEVEL4-mNctFalse-RYV1_MANIFOLD_MIXUP_BILEVEL_real_3.txt',
    # '/c2/jinakim/Drug_Discovery_j/experiments/S-results-strans_mvalid-all-3real-ml0-xmix-mvdefMIXUP_BILEVEL4-mNctFalse-RYV1_MIXUP_BILEVEL_real_3.txt'
    # '/c2/jinakim/Drug_Discovery_j/experiments/S-results-strans_mvalid-all-3real-ml0-xmix-mvdef1-mNctFalse-RYV1_real.txt'
    # "/c2/jinakim/Drug_Discovery_j/experiments/S-results-dsets_mvalid-all-3real-ml0-xmix-mvdef1-mNctFalse-RYV1_real.txt",
    # "/c2/jinakim/Drug_Discovery_j/experiments/S-results-dsets_mvalid-exclude-all-3real.txt",
    # "/c2/jinakim/Drug_Discovery_j/experiments/S-results-strans_mvalid-all-3real.txt",
    # "/c2/jinakim/Drug_Discovery_j/experiments/S-results-strans_mvalid-exclude-all-3real.txt"
]

# Run processing
for file_path in input_files:
    sort_and_save_blocks(file_path)

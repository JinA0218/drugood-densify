import os
import re

# Types and experiment files
types = [
    "hivprot_count", "dpp4_count", "nk1_count",
    "hivprot_bit", "dpp4_bit", "nk1_bit"
]

exp_files = [
    "S-results-dsets_mvalid-all-3real.txt",
    "S-results-dsets_mvalid-exclude-all-3real.txt",
    "S-results-strans_mvalid-all-3real.txt",
    "S-results-strans_mvalid-exclude-all-3real.txt"
]

# Base input and output directories
base_input_dir = "/c2/jinakim/Drug_Discovery_j/experiments/final/sorted"
base_output_dir = "/c2/jinakim/Drug_Discovery_j/experiments/final/top30_mv_pairs"
os.makedirs(base_output_dir, exist_ok=True)

# Extract and save top 30 mv pairs per (type, exp)
for tname in types:
    for exp in exp_files:
        input_path = os.path.join(base_input_dir, tname, exp)
        if not os.path.exists(input_path):
            continue

        # Read and split blocks
        with open(input_path, "r") as f:
            lines = f.readlines()

        blocks = []
        current_block = []
        for line in lines:
            if tname.split("_")[0] in line and current_block:
                blocks.append(current_block)
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            blocks.append(current_block)

        # Extract top 30 mv pairs
        mv_pairs = []
        for block in blocks[:30]:
            for line in block:
                mv_match = re.search(r"mv\s*:\s*\[(.*?)\]", line)
                if mv_match:
                    mv = [x.strip().strip("'") for x in mv_match.group(1).split(",")]
                    if len(mv) == 2:
                        mv_pairs.append(mv)
                        break

        # Write output file
        output_filename = f"{tname}_{exp.replace('.txt', '')}_top30_mv_pairs.txt"
        output_path = os.path.join(base_output_dir, output_filename)
        with open(output_path, "w") as f:
            for pair in mv_pairs:
                f.write(f"{pair[0]}, {pair[1]}\n")


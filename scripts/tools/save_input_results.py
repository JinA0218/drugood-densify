import os
import re
from collections import defaultdict

def group_results_by_dataset(input_dataset_name):
    # Experiment file names
    exp_files = [
        "S-results-dsets_mvalid-all-3real.txt",
        "S-results-dsets_mvalid-exclude-all-3real.txt",
        "S-results-strans_mvalid-all-3real.txt",
        "S-results-strans_mvalid-exclude-all-3real.txt"
    ]

    # Model type priority order
    model_order = [
        "hivprot count", "hivprot bit",
        "dpp4 count", "dpp4 bit",
        "nk1 count", "nk1 bit"
    ]

    # Base input/output paths
    input_base_dir = "/c2/jinakim/Drug_Discovery_j/experiments"
    output_base_dir = f"/c2/jinakim/Drug_Discovery_j/experiments/final/per_dataset/{input_dataset_name}"
    os.makedirs(output_base_dir, exist_ok=True)

    for exp in exp_files:
        input_path = os.path.join(input_base_dir, exp)
        if not os.path.exists(input_path):
            print(f"Missing {input_path}")
            continue

        with open(input_path, "r") as f:
            lines = f.readlines()

        # Group by mv-pair
        grouped_blocks = defaultdict(list)

        current_block = []
        for line in lines:
            if any(model in line for model in ["hivprot count", "hivprot bit", "dpp4 count", "dpp4 bit", "nk1 count", "nk1 bit"]) and current_block:
                # Check previous block
                for subline in current_block:
                    mv_match = re.search(r"mv\s*:\s*\[(.*?)\]", subline)
                    if mv_match:
                        mv = [x.strip().strip("'") for x in mv_match.group(1).split(",")]
                        if input_dataset_name in mv:
                            mv_key = tuple(sorted(mv))
                            grouped_blocks[mv_key].append(current_block)
                            break
                current_block = [line]
            else:
                current_block.append(line)

        if current_block:
            for subline in current_block:
                mv_match = re.search(r"mv\s*:\s*\[(.*?)\]", subline)
                if mv_match:
                    mv = [x.strip().strip("'") for x in mv_match.group(1).split(",")]
                    if input_dataset_name in mv:
                        mv_key = tuple(sorted(mv))
                        grouped_blocks[mv_key].append(current_block)
                        break

        # Write grouped output
        output_path = os.path.join(output_base_dir, f"{exp}.txt")
        with open(output_path, "w") as out_f:
            for mv_pair, blocks in grouped_blocks.items():
                # First write the mv pair
                out_f.write(f"{list(mv_pair)}\n\n")
                # Then order blocks by model priority
                ordered_blocks = []
                for model in model_order:
                    for block in blocks:
                        if any(model in line for line in block):
                            ordered_blocks.append(block)
                for block in ordered_blocks:
                    for line in block:
                        out_f.write(line)
                    out_f.write("\n")

    print(f"✅ Saved results to {output_base_dir}")

# Example usage:
# group_results_by_dataset("logd")


# Example usage
dataset = input("Enter dataset: ").strip()
group_results_by_dataset(dataset)

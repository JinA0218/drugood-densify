import re
from collections import defaultdict

def parse_and_sort_last_performance(text_lines):
    model_blocks = defaultdict(list)
    current_model = None
    current_mv_pair = None

    for line in text_lines:
        # Detect model type
        for model_type in ["hivprot count", "hivprot bit", "dpp4 count", "dpp4 bit", "nk1 count", "nk1 bit"]:
            if model_type in line:
                current_model = model_type
                # Try to get mv pair from the same line
                mv_match = re.search(r"mv\s*:\s*(\[.*?\])", line)
                if mv_match:
                    mv_pair = eval(mv_match.group(1))
                    current_mv_pair = tuple(mv_pair)
                else:
                    current_mv_pair = None
                break

        # Extract last performance mu
        if current_model:
            match = re.search(r"last performance mu:\s*([0-9.eE+-]+)\s*\+-\s*([0-9.eE+-]+)", line)
            if match and current_mv_pair:
                mean_val = float(match.group(1))
                std_val = float(match.group(2))
                model_blocks[current_model].append((current_mv_pair, mean_val, std_val))

    return model_blocks

# Main
path = input("Enter the path to the file: ").strip()

with open(path, "r") as f:
    lines = f.readlines()

model_blocks = parse_and_sort_last_performance(lines)

# Write sorted output
output_path = path.replace(".txt.txt", "_SORTED.txt")
with open(output_path, "w") as out_f:
    for model_type, entries in model_blocks.items():
        out_f.write(f"\n=== {model_type.upper()} ===\n")
        sorted_entries = sorted(entries, key=lambda x: x[1])  # sort by mean
        for mv_pair, mean_val, std_val in sorted_entries:
            out_f.write(f"Pair: {mv_pair}, Mean: {mean_val:.6f}, Std: {std_val:.6f}\n")

print(f"✅ Sorted results saved to {output_path}")

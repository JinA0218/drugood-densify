import re
import numpy as np
from collections import defaultdict
import os

# Get file path from user
file_path = input("Enter the path to the result file: ").strip()

# Data containers
mu_data = defaultdict(list)
last_mu_data = defaultdict(list)

# Regex patterns
header_pattern = re.compile(r"^(hivprot|dpp4|nk1)\s+(count|bit)")
mu_pattern = re.compile(r"mu:\s+([\d.eE+-]+)\s+\+-\s+[\d.eE+-]+")
last_mu_pattern = re.compile(r"last performance mu:\s+([\d.eE+-]+)\s+\+-\s+[\d.eE+-]+")

# Track current experiment block
current_exp = None

# Parse the file
with open(file_path, "r") as f:
    for line in f:
        line = line.strip()  # Trim whitespace
        header_match = header_pattern.search(line)
        if header_match:
            current_exp = f"{header_match.group(1).upper()} {header_match.group(2)}"
        elif "mu:" in line and current_exp and "last performance" not in line:
            mu_match = mu_pattern.search(line)
            if mu_match:
                mu_data[current_exp].append(float(mu_match.group(1)))
        elif "last performance mu:" in line and current_exp:
            last_mu_match = last_mu_pattern.search(line)
            if last_mu_match:
                last_mu_data[current_exp].append(float(last_mu_match.group(1)))
            else:
                print(f"[WARN] Failed to parse: {line}")

# Prepare output path
output_dir = "experiments/final"
os.makedirs(output_dir, exist_ok=True)
base_name = os.path.basename(file_path)
output_path = os.path.join(output_dir, f"summary-{base_name}")

# Write summary
with open(output_path, "w") as out:
    for key in sorted(mu_data.keys()):
        mu_array = np.array(mu_data[key])
        last_mu_array = np.array(last_mu_data[key])

        out.write(f"=== {key} ===\n")

        if mu_array.size == 0:
            out.write("mu: [EMPTY]\n")
        else:
            out.write(f"mu: mean={mu_array.mean():.4f}, std={mu_array.std():.4f}, median={np.median(mu_array):.4f}\n")

        if last_mu_array.size == 0:
            out.write("last performance mu: [EMPTY]\n")
        else:
            out.write(f"last performance mu: mean={last_mu_array.mean():.4f}, std={last_mu_array.std():.4f}, median={np.median(last_mu_array):.4f}\n")

        out.write("\n")

print(f"✅ Summary saved to: {output_path}")

import os
import re
import numpy as np
import pandas as pd

def summarize_last_performance_mu(text_blocks):
    """Extracts and summarizes last performance mu values."""
    last_mu_means = []

    for line in text_blocks:
        match = re.search(r"last performance mu:\s*([0-9.eE+-]+)\s*\+\-", line)
        if match:
            mean_value = float(match.group(1))
            last_mu_means.append(mean_value)

    if not last_mu_means:
        return None

    mean_mu = np.mean(last_mu_means)
    std_mu = np.std(last_mu_means)
    median_mu = np.median(last_mu_means)

    return mean_mu, std_mu, median_mu, last_mu_means

def process_all_txt_files(input_dataset):
    base_input_dir = f"/c2/jinakim/Drug_Discovery_j/experiments/final/per_dataset/{input_dataset}"
    base_output_dir = f"/c2/jinakim/Drug_Discovery_j/experiments/final/per_dataset/{input_dataset}"

    # List of experiment files
    exp_files = [
        "S-results-dsets_mvalid-all-3real.txt.txt",
        "S-results-dsets_mvalid-exclude-all-3real.txt.txt",
        "S-results-strans_mvalid-all-3real.txt.txt",
        "S-results-strans_mvalid-exclude-all-3real.txt.txt"
    ]

    summary_data = []

    for exp_file in exp_files:
        file_path = os.path.join(base_input_dir, exp_file)
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        with open(file_path, "r") as f:
            lines = f.readlines()

        result = summarize_last_performance_mu(lines)
        if result:
            mean_mu, std_mu, median_mu, values = result
            summary_data.append({
                "experiment": exp_file,
                "mean": mean_mu,
                "std": std_mu,
                "median": median_mu,
                "count": len(values)
            })

    # Save the summary as a CSV
    df_summary = pd.DataFrame(summary_data)
    output_summary_path = os.path.join(base_output_dir, f"{input_dataset}_summary.csv")
    df_summary.to_csv(output_summary_path, index=False)
    print(f"✅ Summary saved to {output_summary_path}")

# Example usage:
process_all_txt_files("3a4")

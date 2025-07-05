import os

datasets = ["hivprot", "dpp4", "nk1"]
featurizations = ["count", "bit"]

# datasets = ["hivprot"]
# featurizations = ["bit"]
hyper_keys = range(48)
num_inner_dataset = 1  # replace if this changes
base_dirs = [
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_dsets_n_mvalid_real",
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_strans_n_mvalid_real",
    
    
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_dsets_n_mvalid_real_ryv1",
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_strans_n_mvalid_real_ryv1",
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_dsets_MIXUP_BILEVEL_n_mvalid_real",
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_dsets_MANIFOLD_MIXUP_BILEVEL_n_mvalid_real",
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_dsets_n_mvalid_real_ryv1",
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_strans_n_mvalid_real_ryv1"
    
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_dsets_MIXUP_real",
    # "/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_dsets_MANIFOLD_MIXUP_real",

    "/c2/jinakim/Drug_Discovery_j/experiments2/hyper_search_strans_n_mvalid_real_ryv1",
    # "/c2/jinakim/Drug_Discovery_j/experiments2/hyper_search_dsets_n_mvalid_real_ryv1"
]

missing_files = []

for base_dir in base_dirs:
    print(f"\nChecking in: {base_dir}")
    for dataset in datasets:
        for featurization in featurizations:
            for hyper_key in hyper_keys:
                fname = f"nmvalid-{dataset}-{featurization}-{hyper_key}--in{num_inner_dataset}.pkl"
                fpath = os.path.join(base_dir, fname)
                if not os.path.exists(fpath):
                    missing_files.append(fpath)

if missing_files:
    print(f"\n❌ {len(missing_files)} missing files:")
    for f in missing_files:
        print(f"  - {f}")
else:
    print("\n✅ All expected files exist.")

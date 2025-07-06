import os
import pickle
import re

EXPERIMENT_VERSION = "table2_row5"  # NOTE Change this to switch experiment type

EXPERIMENT_PATHS = {
    # <Table 1>
    "table1_row5": "experiments/hyper_search_{model}_MIXUP_real/",
    "table1_row6": "experiments/hyper_search_{model}_MIXUP_BILEVEL_n_mvalid_real/",
    "table1_row7": "experiments/hyper_search_{model}_MANIFOLD_MIXUP_real/",
    "table1_row8": "experiments/hyper_search_{model}_MANIFOLD_MIXUP_BILEVEL_n_mvalid_real/",
    
    # <Table 2>
    "table2_row1_2": "experiments/hyper_search_{model}_n_mvalid_real_ryv1_no_bilevel/",
    "table2_row3": "experiments/hyper_search_{model}_n_mvalid_real_ryv1_mlp_real2/",
    "table2_row4": "experiments/hyper_search_dsets_n_mvalid_real_ryv0/",
    "table2_row5": "experiments/hyper_search_strans_n_mvalid_real_ryv0/",
}

DATASETS = ["hivprot", "dpp4", "nk1"]
TYPES = ["count", "bit"]
MODELS = ["dsets"]
OURS_MODELS = ["dsets", "strans"]

TABLE_FILTERS = {"lr": 0.001, "num_layers": 3, "hidden_dim": 64, }


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def match_filters(metrics, filters):
    return all(metrics.get(k) == v for k, v in filters.items())

def filter_files(files, ds, t, filename_check=None):
    return [
        f for f in files
        if ds in f and t in f and (filename_check is None or filename_check(f))
    ]

def find_best_metrics(dir_path, filtered_files, filters):
    best_mse = float("inf")
    best_metrics = None
    for fname in filtered_files:
        metrics = load_pickle(os.path.join(dir_path, fname))
        if not match_filters(metrics, filters):
            continue
        if metrics["mse"] < best_mse:
            best_metrics = metrics
            best_mse = metrics["mse"]
    return best_metrics

# =============================
# Table 2, 3 experiments
# =============================

def run_table_rows():
    d_template = EXPERIMENT_PATHS[EXPERIMENT_VERSION]
    for model in MODELS: # TODO FINAL CHANGE
        d = d_template.format(model=model)
        print("=" * 20 + f"{EXPERIMENT_VERSION}" + "=" * 20)
        files = os.listdir(d)

        for ds in DATASETS:
            for t in TYPES:
                filtered = filter_files(files, ds, t)
                if not filtered:
                    print(f"no files for: {ds=} {t=}")
                    continue

                best_metrics = find_best_metrics(d, filtered, TABLE_FILTERS)
                print(f"\n{EXPERIMENT_VERSION} {ds=} {t=} files: {len(filtered)}\n{best_metrics=}\n\n")

# =============================
# OURS experiments
# =============================

def run_ours_models():
    for model in OURS_MODELS:
        d = f"experiments/hyper_search_{model}_n_mvalid_real_ryv1/"
        print("=" * 20 + f"OURS {model}" + "=" * 20)
        files = os.listdir(d)

        filters = {
            "lr": 0.001,
            "num_layers": 3,
            "hidden_dim": 64,
        }

        for ds in DATASETS:
            for t in TYPES:
                def filename_check(fname):
                    match = re.search(rf"{ds}-{t}-(\d+)--in1\.pkl", fname)
                    return match and 0 <= int(match.group(1)) <= 71

                filtered = filter_files(files, ds, t, filename_check)
                if not filtered:
                    print(f"no files for: {ds=} {t=}")
                    continue

                best_metrics = find_best_metrics(d, filtered, filters)
                print(f"\nOURS {ds=} {t=} files: {len(filtered)}\n{best_metrics=}\n\n")

if __name__ == "__main__":
    run_table_rows()
    run_ours_models()

    # TODO FINAL CHECK
    #### ecfp rdkit
    # for model in ["dsets", "strans"]:
    #     d = f"experiments/ce_hyper_search_{model}-ctx32/"
    #     print("=" * 20 + f"OURS CROSS ENTROPY {model}" + "=" * 20)
    #     files = os.listdir(d)
    #
    #     for split in ["spectral", "weight", "scaffold", "random"]:
    #         for fp in ["ecfp", "rdkit"]:
    #             filtered_files = [f for f in files if split in f and fp in f]
    #             if not filtered_files:
    #                 print(f"no files for: {split=} {fp=}")
    #                 continue
    #
    #             best_auc = 0
    #             best_brier = float("inf")
    #             best_auc_metrics = {}
    #             best_brier_metrics = {}
    #             for _f in filtered_files:
    #                 metrics = load_pickle(os.path.join(d, _f))
    #
    #                 if metrics["metrics"]["val_brier_brier"] < best_brier:
    #                     best_brier_metrics = metrics
    #                     best_brier = metrics["metrics"]["val_brier_brier"]
    #
    #                 if metrics["metrics"]["val_auc_auc"] > best_auc:
    #                     best_auc_metrics = metrics
    #                     best_auc = metrics["metrics"]["val_auc_auc"]
    #
    #             print(f"\n{split=} {fp=} files: {len(filtered_files)}")
    #             print(f"{best_auc_metrics=}\n\n{best_brier_metrics=}\n\n")
    #             print(f"{'-' * 50}")
    #
    #             print("best auc metrics:")
    #             for k, v in best_auc_metrics["metrics"].items():
    #                 print(f"{k}: {v}")
    #             print("\n\n")
    #
    #             print("best brier metrics:")
    #             for k, v in best_brier_metrics["metrics"].items():
    #                 print(f"{k}: {v}")
    #             print("\n\n")

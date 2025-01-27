import os
import pickle

if __name__ == "__main__":

    for model in ["dsets", "strans"]:
        d = f"experiments/hyper_search_{model}/"
        print("=" * 20 + f"OURS {model}" + "=" * 20)
        files = os.listdir(d)

        for ds in ["hivprot", "dpp4", "nk1"]:
            for t in ["count", "bit"]:
                filtered_files = [f for f in files if ds in f and t in f]
                if len(filtered_files) == 0:
                    print(f"no files for: {ds=} {t=}")
                    continue

                best_mse = float("inf")
                best_metrics = None
                for _f in filtered_files:
                    with open(os.path.join(f"{d}/{_f}"), "rb") as f:
                        metrics = pickle.load(f)

                    if metrics["mse"] < best_mse:
                        best_metrics = metrics
                        best_mse = metrics["mse"]

                # print(f"{ds=} {t=}\n\n{best_metrics=}")
                print(f"\n{ds=} {t=} files: {len(filtered_files)}\n{best_metrics=}\n\n")

    for model in ["dsets", "strans"]:
        d = f"experiments/ce_hyper_search_{model}/"
        print("=" * 20 + f"OURS CROSS ENTROPY {model}" + "=" * 20)
        files = os.listdir(d)

        for split in ["spectral", "weight", "scaffold", "random"]:
            for fp in ["ecfp", "rdkit"]:
                filtered_files = [f for f in files if split in f and fp in f]
                if len(filtered_files) == 0:
                    print(f"no files for: {split=} {fp=}")
                    continue

                best_auc = 0
                best_brier = float("inf")
                best_auc_metrics = {}
                best_brier_metrics = {}
                for _f in filtered_files:
                    with open(os.path.join(f"{d}/{_f}"), "rb") as f:
                        metrics = pickle.load(f)

                    # if metrics["metrics"]["val_brier_brier"] < best_brier:
                    #     best_brier_metrics = metrics
                    #     best_brier = metrics["metrics"]["val_brier_brier"]

                    # if metrics["metrics"]["val_auc_auc"] > best_auc:
                    #     best_auc_metrics = metrics
                    #     best_auc = metrics["metrics"]["val_auc_auc"]
                    if metrics["metrics"]["brier_brier"] < best_brier:
                        best_brier_metrics = metrics
                        best_brier = metrics["metrics"]["brier_brier"]

                    if metrics["metrics"]["auc_auc"] > best_auc:
                        best_auc_metrics = metrics
                        best_auc = metrics["metrics"]["auc_auc"]

                # print(f"\n{split=} {fp=} files: {len(filtered_files)}\n{best_auc_metrics=}\n\n{best_brier_metrics=}\n\n")
                # print(f"{'-' * 50}")

                # print("best auc metrics:")
                # for k, v in best_auc_metrics["metrics"].items():
                #     print(f"{k}: {v}")
                # print("\n\n")

                # print("best brier metrics:")
                # for k, v in best_brier_metrics["metrics"].items():
                #     print(f"{k}: {v}")
                # print("\n\n")

                print(f"\n{split=} {fp=} files: {len(filtered_files)}\n\n")
                print(f"auc: {best_auc_metrics['metrics']['auc_auc']=}")
                print(f"\n{best_auc_metrics}\n")
                print(f"brier: {best_brier_metrics['metrics']['brier_brier']=}")
                print(f"\n{best_brier_metrics}\n")

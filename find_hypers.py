import os
import pickle

import re


if __name__ == "__main__":

    # model = "strans"
    # d = f"/c2/jinakim/Drug_Discovery_j/experiments/hyper_search_{model}_n_mvalid_real_ryv1/"
    # files = os.listdir(d)

    # filtered_files = []
    # for ds in ["hivprot", "dpp4", "nk1"]:
    #     for t in ["count", "bit"]:
    #         for fname in files:
    #             if ds in fname and t in fname:
    #                 match = re.search(rf"{ds}-{t}-(\d+)--in1\.pkl", fname)
    #                 if match:
    #                     a = int(match.group(1))
    #                     if 0 <= a <= 71:
    #                         filtered_files.append(fname)

    # print("Filtered files:")
    # for f in filtered_files:
    #     print(f)

    for model in ["dsets"]: # "dsets", 
        # d = f"experiments/hyper_search_{model}_n_mvalid_real_ryv1/"
        # d = f"experiments/hyper_search_{model}_MANIFOLD_MIXUP_BILEVEL_n_mvalid_real/"
        d = f"experiments/hyper_search_{model}_MIXUP_BILEVEL_n_mvalid_real/"
        # d = f"experiments/hyper_search_{model}_MIXUP_real/"
        # d = f"experiments/hyper_search_{model}_MANIFOLD_MIXUP_real/"
        print("=" * 20 + f"OURS {model}" + "=" * 20)
        files = os.listdir(d)


        for ds in ["hivprot", "dpp4", "nk1"]: #  "hivprot", "dpp4", 
            for t in ["count", "bit"]: # "count", 
                # filtered_files = []
                # for fname in files:
                #     if ds in fname and t in fname:
                #         match = re.search(rf"{ds}-{t}-(\d+)--in1\.pkl", fname)
                #         if match:
                #             a = int(match.group(1))
                #             if 0 <= a <= 71:
                #                 filtered_files.append(fname)

                filtered_files = [f for f in files if ds in f and t in f]
                if len(filtered_files) == 0:
                    print(f"no files for: {ds=} {t=}")
                    continue

                best_mse = float("inf")
                best_metrics = None
                for _f in filtered_files:
                    with open(os.path.join(f"{d}/{_f}"), "rb") as f:
                        metrics = pickle.load(f)

                    if metrics.get("lr") != 0.001:
                        continue

                    if metrics.get("num_layers") != 3:
                        continue
                    
                    if metrics.get("hidden_dim") != 64:
                        continue

                    if metrics["mse"] < best_mse:
                        best_metrics = metrics
                        best_mse = metrics["mse"]

                # print(f"{ds=} {t=}\n\n{best_metrics=}")Z
                print(f"\n{ds=} {t=} files: {len(filtered_files)}\n{best_metrics=}\n\n")



######### FOR OURS
    # for model in ["strans"]: # "dsets", 
    #     d = f"experiments/hyper_search_{model}_n_mvalid_real_ryv1/"
    #     # d = f"experiments/hyper_search_{model}_MANIFOLD_MIXUP_BILEVEL_n_mvalid_real/"
    #     # d = f"experiments/hyper_search_{model}_MIXUP_real/"
    #     # d = f"experiments/hyper_search_{model}_MANIFOLD_MIXUP_real/"
    #     print("=" * 20 + f"OURS {model}" + "=" * 20)
    #     files = os.listdir(d)


    #     for ds in ["hivprot", "dpp4", "nk1"]: #  "hivprot", "dpp4", 
    #         for t in ["count", "bit"]: # "count", 
    #             filtered_files = []
    #             for fname in files:
    #                 if ds in fname and t in fname:
    #                     match = re.search(rf"{ds}-{t}-(\d+)--in1\.pkl", fname)
    #                     if match:
    #                         a = int(match.group(1))
    #                         if 0 <= a <= 71:
    #                             filtered_files.append(fname)

    #             if len(filtered_files) == 0:
    #                 print(f"no files for: {ds=} {t=}")
    #                 continue

    #             best_mse = float("inf")
    #             best_metrics = None
    #             for _f in filtered_files:
    #                 with open(os.path.join(f"{d}/{_f}"), "rb") as f:
    #                     metrics = pickle.load(f)

    #                 if metrics.get("lr") != 0.001:
    #                     continue

    #                 if metrics.get("num_layers") != 4:
    #                     continue
                    
    #                 if metrics.get("hidden_dim") != 32:
    #                     continue

    #                 if metrics["mse"] < best_mse:
    #                     best_metrics = metrics
    #                     best_mse = metrics["mse"]

    #             # print(f"{ds=} {t=}\n\n{best_metrics=}")Z
    #             print(f"\n{ds=} {t=} files: {len(filtered_files)}\n{best_metrics=}\n\n")
##########


####################333 ORIGIN
    # for model in ["strans"]: # "dsets", 
    #     d = f"experiments/hyper_search_{model}_n_mvalid_real_ryv1/"
    #     # d = f"experiments/hyper_search_{model}_MANIFOLD_MIXUP_BILEVEL_n_mvalid_real/"
    #     # d = f"experiments/hyper_search_{model}_MIXUP_real/"
    #     # d = f"experiments/hyper_search_{model}_MANIFOLD_MIXUP_real/"
    #     print("=" * 20 + f"OURS {model}" + "=" * 20)
    #     files = os.listdir(d)

    #     for ds in ["hivprot", "dpp4", "nk1"]: #  "hivprot", "dpp4", 
    #         for t in ["count", "bit"]: # "count", 
    #             filtered_files = [f for f in files if ds in f and t in f]
    #             if len(filtered_files) == 0:
    #                 print(f"no files for: {ds=} {t=}")
    #                 continue

    #             best_mse = float("inf")
    #             best_metrics = None
    #             for _f in filtered_files:
    #                 with open(os.path.join(f"{d}/{_f}"), "rb") as f:
    #                     metrics = pickle.load(f)

    #                 if metrics.get("lr") != 0.001:
    #                     continue

    #                 if metrics["mse"] < best_mse:
    #                     best_metrics = metrics
    #                     best_mse = metrics["mse"]

    #             # print(f"{ds=} {t=}\n\n{best_metrics=}")Z
    #             print(f"\n{ds=} {t=} files: {len(filtered_files)}\n{best_metrics=}\n\n")
##################33

    # for model in ["dsets"]:  # You can add other models here
    #     # d = f"experiments/hyper_search_{model}_MIXUP_real/"
    #     d = f"experiments/hyper_search_{model}_n_mvalid_real_ryv1/"
        
    #     print("=" * 20 + f"OURS {model}" + "=" * 20)
    #     files = os.listdir(d)

    #     for ds in ["hivprot", "dpp4", "nk1"]: # 
    #         for t in ["count", "bit"]:
    #             filtered_files = [f for f in files if ds in f and t in f]
    #             if len(filtered_files) == 0:
    #                 print(f"no files for: {ds=} {t=}")
    #                 continue

    #             best_mse = float("inf")
    #             best_metrics = None

    #             for _f in filtered_files:
    #                 with open(os.path.join(d, _f), "rb") as f:
    #                     metrics = pickle.load(f)

    #                 # Only consider if sencoder_layer == "max"
    #                 if metrics.get("sencoder_layer") != "pma":
    #                     continue

    #                 if metrics["mse"] < best_mse:
    #                     best_metrics = metrics
    #                     best_mse = metrics["mse"]

    #             print(f"\n{ds=} {t=} files: {len(filtered_files)}\n{best_metrics=}\n\n")







    # for model in ["dsets", "strans"]:
    #     d = f"experiments/ce_hyper_search_{model}-ctx32/"
    #     print("=" * 20 + f"OURS CROSS ENTROPY {model}" + "=" * 20)
    #     files = os.listdir(d)

    #     for split in ["spectral", "weight", "scaffold", "random"]:
    #         for fp in ["ecfp", "rdkit"]:
    #             filtered_files = [f for f in files if split in f and fp in f]
    #             if len(filtered_files) == 0:
    #                 print(f"no files for: {split=} {fp=}")
    #                 continue

    #             best_auc = 0
    #             best_brier = float("inf")
    #             best_auc_metrics = {}
    #             best_brier_metrics = {}
    #             for _f in filtered_files:
    #                 with open(os.path.join(f"{d}/{_f}"), "rb") as f:
    #                     metrics = pickle.load(f)

    #                 if metrics["metrics"]["val_brier_brier"] < best_brier:
    #                     best_brier_metrics = metrics
    #                     best_brier = metrics["metrics"]["val_brier_brier"]

    #                 if metrics["metrics"]["val_auc_auc"] > best_auc:
    #                     best_auc_metrics = metrics
    #                     best_auc = metrics["metrics"]["val_auc_auc"]
    #                 # if metrics["metrics"]["brier_brier"] < best_brier:
    #                 #     best_brier_metrics = metrics
    #                 #     best_brier = metrics["metrics"]["brier_brier"]

    #                 # if metrics["metrics"]["auc_auc"] > best_auc:
    #                 #     best_auc_metrics = metrics
    #                 #     best_auc = metrics["metrics"]["auc_auc"]

    #             print(f"\n{split=} {fp=} files: {len(filtered_files)}\n{best_auc_metrics=}\n\n{best_brier_metrics=}\n\n")
    #             print(f"{'-' * 50}")

    #             print("best auc metrics:")
    #             for k, v in best_auc_metrics["metrics"].items():
    #                 print(f"{k}: {v}")
    #             print("\n\n")

    #             print("best brier metrics:")
    #             for k, v in best_brier_metrics["metrics"].items():
    #                 print(f"{k}: {v}")
    #             print("\n\n")

                # print(f"\n{split=} {fp=} files: {len(filtered_files)}\n\n")
                # print(f"auc: {best_auc_metrics['metrics']['auc_auc']=}")
                # print(f"\n{best_auc_metrics}\n")
                # print(f"brier: {best_brier_metrics['metrics']['brier_brier']=}")
                # print(f"\n{best_brier_metrics}\n")

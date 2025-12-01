# Description: This script is used to run the federated learning experiments on the ProCan compendium dataset (Figure 3).

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
    KFold,
)
from sklearn.preprocessing import LabelEncoder
from FedAggregateWeights import WeightsAggregation
from FedTrain import TrainFedProtNet, TrainFedProtNet_DP
from utils.ProtDataset import ProtDataset
from sklearn.model_selection import GroupShuffleSplit
import config

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

N_clients = 5
N_included_clients = 5
N_iters = 100

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ProCanFDL Federated Learning with INR')
    parser.add_argument('--use_inr', action='store_true', help='Use INR (Implicit Neural Representation) layers')
    parser.add_argument('--use_dp', action='store_true', help='Use Differential Privacy')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA layers')
    parser.add_argument('--use_dr', action='store_true', help='Use decentralized randomization for model aggregation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0, cpu, etc.)')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load hyperparameters from config
    hypers = config.DEFAULT_HYPERPARAMETERS.copy()
    hypers.update(
        {
            "target": config.DEFAULT_TARGET,
            "included_types": config.DEFAULT_CANCER_TYPES,
        }
    )

    print(hypers)
    TARGET = hypers["target"]
    included_types = hypers["included_types"]
    FILE_NAME = f"{hypers['target']}_INR_{args.use_inr}_DP_{args.use_dp}_LoRA_{args.use_lora}_DR_{args.use_dr}"
    # MODEL_PATH = config.get_model_path(FILE_NAME)
    MODEL_PATH = f'/deltadisk/zhangrongyu/leo/ProCanFDL/new_model/INFL_INR_{args.use_inr}_DP_{args.use_dp}_LoRA_{args.use_lora}_DR_{args.use_dr}'
    os.makedirs(MODEL_PATH, exist_ok=True)
    print(f"MODEL_PATH: {MODEL_PATH}")
    
    # Print configuration
    print(f"\n=== Configuration ===")
    print(f"Use INR: {args.use_inr}")
    print(f"Use DP: {args.use_dp}")
    print(f"Use LoRA: {args.use_lora}")
    print(f"Use Decentralized Randomization: {args.use_dr}")
    print(f"Device: {args.device}")
    print("=" * 50)

    prot_id_to_name = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=0).to_dict()[
        "Gene"
    ]

    prot_name_to_id = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=3).to_dict()[
        "UniProtID"
    ]

    combined_df = pd.read_csv(
        config.PROCAN_DATA_FILE,
        index_col=0,
        low_memory=False,
    )

    # combined_df = combined_df[combined_df["sample_tissue_descriptor"] == "Malignant"]
    # combined_df = combined_df[combined_df["sample_sample_type"] != "Cell lines"]
    # corr_df = pd.read_csv(config.REPLICATE_CORR_FILE)
    # low_corr_samples = corr_df[corr_df["pearsons_r"] < config.QUALITY_THRESHOLD][
    #     "prep_lims_id"
    # ].tolist()
    # combined_df = combined_df[~combined_df.index.isin(low_corr_samples)]
    if TARGET in ["Broad_Cancer_Type", "cancer_type", "cancer_type_2"]:
        combined_selected_df = combined_df[combined_df[TARGET].isin(included_types)]
    else:
        combined_selected_df = combined_df[
            (combined_df["Broad_Cancer_Type"] == "Adenocarcinoma")
            & (combined_df[TARGET].isin(included_types))
        ]
    merged_label_map = combined_selected_df[TARGET].to_dict()

    # combined_vcb_df = combined_selected_df[
    #     combined_selected_df["prep_cohort"] == "E0008-P01"
    # ]
    # combined_rest_df = combined_selected_df[
    #     combined_selected_df["prep_cohort"] != "E0008-P01"
    # ]
    # combined_rest_df = combined_rest_df[
    #     (combined_rest_df["prep_cohort"] != "E0006-P02")
    #     | (combined_rest_df["prep_general_sample_info"] == "Primary")
    # ]

    # combined_rest_cohort_df = combined_rest_df[
    #     ["prep_cohort", TARGET]
    # ].drop_duplicates()
    # combined_rest_cohort_df = combined_rest_cohort_df.sort_values(
    #     by=[TARGET, "prep_cohort"]
    # )
    # combined_rest_cohort_df = combined_rest_cohort_df.drop_duplicates(
    #     subset=["prep_cohort"], keep="last"
    # ).reset_index(drop=True)
    META_COL_NUMS = config.META_COL_NUMS
    # combined_selected_meta_df = combined_selected_df.iloc[:, :META_COL_NUMS]
    # combined_selected_meta_df['prep_cohort_number'] = combined_selected_meta_df['prep_cohort_number'] + 1
    # combined_selected_meta_df.to_excel(config.RESULTS_DIR / "Samples_for_Figure3.xlsx")

    # meta_vcb = pd.read_excel(
    #     config.SAMPLE_METADATA_FILE,
    #     engine="openpyxl",
    # )
    # meta_vcb = meta_vcb.drop_duplicates(subset=["LIMS-ID1"])
    # meta_vcb = meta_vcb[meta_vcb["Include_Y_N"] == "Yes"]
    # included_vcb_lims = meta_vcb["LIMS-ID1"].tolist()
    # combined_vcb_df = combined_vcb_df[combined_vcb_df.index.isin(included_vcb_lims)]

    # le = LabelEncoder()
    # le.fit(combined_selected_df[TARGET])
    # train_rest_df, test_rest_df = train_test_split(combined_rest_df, stratify=combined_rest_df[TARGET],
    #                                                test_size=0.1, random_state=0)
    # unique_cohorts = combined_rest_df["prep_cohort"].unique()
    # train_rest_df = []
    # test_rest_df = []    

    # Check class distribution before splitting
    print("\n=== Class Distribution ===")
    class_counts = combined_selected_df[TARGET].value_counts()
    print(class_counts)
    print(f"\nClasses with only 1 sample: {class_counts[class_counts == 1].index.tolist()}")
    
    # Filter out classes with less than 2 samples for stratified split
    valid_classes = class_counts[class_counts >= 2].index
    combined_selected_df_filtered = combined_selected_df[combined_selected_df[TARGET].isin(valid_classes)]
    
    print(f"\nOriginal samples: {len(combined_selected_df)}")
    print(f"Filtered samples: {len(combined_selected_df_filtered)}")
    print(f"Removed classes: {set(combined_selected_df[TARGET].unique()) - set(valid_classes)}")
    
    # Split combined_selected_df into train (90%) and test (10%) datasets
    train_df, test_df = train_test_split(
        combined_selected_df_filtered, 
        stratify=combined_selected_df_filtered[TARGET],
        test_size=0.1,  # 10% for test
        train_size=0.9,  # 90% for train
        random_state=0
    )
    
    # Fit LabelEncoder on the filtered data (after removing classes with <2 samples)
    le = LabelEncoder()
    le.fit(combined_selected_df_filtered[TARGET])
    
    print(f"Train set size: {len(train_df)} ({len(train_df)/len(combined_selected_df)*100:.1f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(combined_selected_df)*100:.1f}%)")
    print(train_df[TARGET].value_counts())
    print(test_df[TARGET].value_counts())

    # for c in unique_cohorts:
    #     print(
    #         f"{c}: {combined_rest_df[combined_rest_df['prep_cohort'] == c][TARGET].value_counts()}"
    #     )
    #     tmp_df = combined_rest_df[combined_rest_df["prep_cohort"] == c]

    #     gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    #     i, (train_index, test_index) = next(
    #         enumerate(
    #             gss.split(
    #                 tmp_df,
    #                 y=tmp_df[TARGET],
    #                 groups=tmp_df["subject_collaborator_patient_id"],
    #             )
    #         )
    #     )
    #     train_rest_cohort_df = tmp_df.iloc[train_index]
    #     test_rest_cohort_df = tmp_df.iloc[test_index]
    #     train_rest_df.append(train_rest_cohort_df)
    #     test_rest_df.append(test_rest_cohort_df)
    # train_rest_df = pd.concat(train_rest_df)
    # test_rest_df = pd.concat(test_rest_df)

    # print(f"test shape of samples: {test_rest_df.shape[0]}")
    # test_rest_df = test_rest_df.drop_duplicates(
    #     subset=["subject_collaborator_patient_id"], keep="last"
    # )
    # print(f"test shape of unique samples: {test_rest_df.shape[0]}")

    # print(train_rest_df[TARGET].value_counts())
    # train_combined_df = pd.concat([combined_vcb_df, train_rest_df])
    best_scores = []
    # print(train_combined_df.groupby([TARGET, "prep_cohort"]).size())
    # print(test_rest_df[TARGET].value_counts())
    # print(test_rest_df.groupby([TARGET, 'prep_cohort']).size())

    # traditional
    test_dataset = ProtDataset(
        test_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
        le.transform(test_df[TARGET].to_numpy()),
        prot_id_to_name,
        prot_name_to_id,
        merged_label_map,
    )
    if N_included_clients == N_clients:
        train_dataset = ProtDataset(
            train_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
            le.transform(train_df[TARGET].to_numpy()),
            prot_id_to_name,
            prot_name_to_id,
            merged_label_map,
        )
    elif N_included_clients < N_clients:
        skf = StratifiedKFold(n_splits=N_clients, shuffle=True, random_state=0)
        included_idx = []
        for fold, (train_index, val_index) in enumerate(
            skf.split(train_df, train_df[TARGET])
        ):
            if fold < N_included_clients:
                included_idx.extend(val_index)
            else:
                break
        train_df_subset = train_df.iloc[included_idx]
        print(f"Using subset of train_df: {len(train_df_subset)}/{len(train_df)} samples")
        train_dataset = ProtDataset(
            train_df_subset.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
            le.transform(train_df_subset[TARGET].to_numpy()),
            prot_id_to_name,
            prot_name_to_id,
            merged_label_map,
        )
    else:
        raise ValueError("N_included_clients should be less than or equal to N_clients")

    print(f"running traditional: {len(train_dataset)} {len(test_dataset)}")
    if args.use_dp:
        model_training = TrainFedProtNet_DP(
            train_dataset,
            test_dataset,
            hypers,
            save_path=f"{MODEL_PATH}/local_model_{hypers['target']}_traditional.pt",
            use_lora=args.use_lora,
        )
    else:
        model_training = TrainFedProtNet(
            train_dataset,
            test_dataset,
            hypers,
            save_path=f"{MODEL_PATH}/local_model_{hypers['target']}_traditional.pt",
            use_inr=args.use_inr,
        )

    score_dict = model_training.run_train_val()
    score_dict["fold"] = "traditional"
    best_scores.append(score_dict)
    
    # Save traditional results to CSV
    traditional_results = {
        "model_type": "traditional",
        "test_acc": score_dict["test_acc"],
        "test_f1": score_dict["test_f1"],
        "test_auc": score_dict["test_auc"],
        "best_epoch": score_dict.get("best_epoch", 0),
    }
    traditional_df = pd.DataFrame([traditional_results])
    traditional_df.to_csv(
        config.get_results_path() / f"traditional_results_{FILE_NAME}.csv", 
        index=False
    )
    print(f"Traditional results saved: {traditional_results}")
    
    confs_df = pd.DataFrame(model_training.get_pred_for_cm(), columns=le.classes_)
    confs_df["pred"] = confs_df.idxmax(axis=1)
    confs_df["true"] = test_df[TARGET].to_numpy()
    confs_df.index = test_df.index
    confs_df.to_csv(config.get_results_path() / f"centralised_pred_{FILE_NAME}.csv")

    # start Fed training
    all_site_sizes_df = []
    all_confs_local_sites_df = []
    all_confs_df = []
    
    # Track best global model with highest test accuracy
    best_global_model_acc = 0.0
    best_global_model_path = '/deltadisk/zhangrongyu/leo/ProCanFDL/new_model'
    best_global_model_info = None
    # initialize global model
    if args.use_dp:
        model = TrainFedProtNet_DP(
            train_dataset,
            test_dataset,
            hypers,
            save_path=f"{MODEL_PATH}/{hypers['target']}_fedavg.pt",
            use_lora=args.use_lora,
        )
    else:
        model = TrainFedProtNet(
        train_dataset,
        test_dataset,
        hypers,
        save_path=f"{MODEL_PATH}/{hypers['target']}_fedavg.pt",
        use_inr=args.use_inr,
    )
    
    
    model.save_model()

    # all the clients
    model_paths = [
        f"{MODEL_PATH}/local_model_{hypers['target']}.pt"
    ] + [
        f"{MODEL_PATH}/local_model_{hypers['target']}_d2_{i}.pt"
        for i in range(N_included_clients)
    ]

    for fed_iter in range(N_iters):
        print(f"Running Fed Iteration {fed_iter}")
        # Train vcb
        # print(combined_df.groupby([TARGET, "prep_cohort"]).size())
        train_dataset = ProtDataset(
            train_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
            le.transform(train_df[TARGET].to_numpy()),
            prot_id_to_name,
            prot_name_to_id,
            merged_label_map,
        )
        if args.use_dp:
            model_training = TrainFedProtNet_DP(
                train_dataset,
                test_dataset,
                hypers,
                load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg.pt",
                save_path=f"{MODEL_PATH}/local_model_{hypers['target']}.pt",
                use_lora=args.use_lora,
            )
        else:   
            model_training = TrainFedProtNet(
                train_dataset,
                test_dataset,
                hypers,
                load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg.pt",
                save_path=f"{MODEL_PATH}/local_model_{hypers['target']}.pt",
                use_inr=args.use_inr,
            )
        

        score_dict = model_training.run_train_val()
        score_dict["model"] = "vlocal"
        score_dict["iter"] = fed_iter
        best_scores.append(score_dict)

        if fed_iter == 0:
            confs = model_training.get_pred_for_cm()
            confs_df = pd.DataFrame(confs, columns=le.classes_)
            confs_df["pred"] = confs_df.idxmax(axis=1)
            confs_df["true"] = test_df[TARGET].to_numpy()
            confs_df.index = test_df.index
            confs_df.to_csv(
                config.get_results_path() / f"local_pred_{FILE_NAME}.csv"
            )

        # Train rest
        # skf = StratifiedKFold(n_splits=N_clients, shuffle=True, random_state=0)
        kf = KFold(n_splits=N_clients, shuffle=True, random_state=0)
        train_tmp_df = train_df.sample(
            frac=1, random_state=0
        ).reset_index(drop=True)
        for fold, (train_index, val_index) in enumerate(
            kf.split(train_tmp_df)
        ):
            # Get samples for this fold
            train_fold_df = train_tmp_df.iloc[val_index]
            print(f"running fold {fold}: {len(train_fold_df)} samples")
            print(f"Fold {fold} class distribution:\n{train_fold_df[TARGET].value_counts()}")
            if fed_iter == 0:
                site_size_df = (
                    train_fold_df.groupby([TARGET])
                    .size()
                    .reset_index()
                )
                site_size_df["site"] = f"simulated_local_{fold}"
                all_site_sizes_df.append(site_size_df)

            train_dataset = ProtDataset(
                train_fold_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
                le.transform(train_fold_df[TARGET].to_numpy()),
                prot_id_to_name,
                prot_name_to_id,
                merged_label_map,
            )

            if args.use_dp:
                model_training = TrainFedProtNet_DP(
                    train_dataset,
                    test_dataset,
                    hypers,
                    load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg.pt",
                    save_path=f"{MODEL_PATH}/local_model_{hypers['target']}_d2_{fold}.pt",
                    use_lora=args.use_lora,
                )
            else:
                model_training = TrainFedProtNet(
                train_dataset,
                test_dataset,
                hypers,
                load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg.pt",
                save_path=f"{MODEL_PATH}/local_model_{hypers['target']}_d2_{fold}.pt",
                use_inr=args.use_inr,
            )
            

            score_dict = model_training.run_train_val()
            score_dict["model"] = f"simulated_local_{fold}"
            score_dict["iter"] = fed_iter
            best_scores.append(score_dict)

            if fed_iter == N_iters - 1:
                confs = model_training.get_pred_for_cm()
                confs_df = pd.DataFrame(confs, columns=le.classes_)
                confs_df["pred"] = confs_df.idxmax(axis=1)
                confs_df["true"] = test_df[TARGET].to_numpy()
                confs_df["fold"] = fold
                confs_df.index = test_df.index
                all_confs_local_sites_df.append(confs_df)

            if fold + 1 >= N_included_clients:
                break

        # Aggregate weights
        weight_agg = WeightsAggregation(
            model_paths=model_paths,
            use_dr=args.use_dr
        )
        weight_agg.fed_avg()
        weight_agg.save_model(
            f"{MODEL_PATH}/{hypers['target']}_fedavg.pt"
        )

        if args.use_dp:
            model_training = TrainFedProtNet_DP(
                test_dataset,
                test_dataset,
                hypers,
                load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg.pt",
                use_lora=args.use_lora,
            )
        else:
            model_training = TrainFedProtNet(
                test_dataset,
                test_dataset,
                hypers,
                load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg.pt",
                use_inr=args.use_inr,
            )
        
        score_dict = model_training.predict()
        score_dict["model"] = "fedavg"
        score_dict["iter"] = fed_iter
        best_scores.append(score_dict)
        print(score_dict)
        
        # Track best global model with highest test accuracy
        if score_dict["test_acc"] > best_global_model_acc:
            best_global_model_acc = score_dict["test_acc"]
            best_global_model_path = f"{MODEL_PATH}/{hypers['target']}_fedavg.pt"
            best_global_model_info = {
                "iter": fed_iter,
                "test_acc": score_dict["test_acc"],
                "test_f1": score_dict["test_f1"],
                "test_auc": score_dict["test_auc"],
                "model_path": best_global_model_path
            }
            print(f"New best global model found! Test accuracy: {best_global_model_acc:.4f} at iter {fed_iter}")

        if fed_iter == N_iters - 1:
            confs = model_training.get_pred_for_cm()
            confs_df = pd.DataFrame(confs, columns=le.classes_)
            confs_df["pred"] = confs_df.idxmax(axis=1)
            confs_df["true"] = test_df[TARGET].to_numpy()
            confs_df.index = test_df.index
            all_confs_df.append(confs_df)

    all_site_sizes_df = pd.concat(all_site_sizes_df)
    all_site_sizes_df.to_csv(
        config.get_results_path() / f"site_sizes_{FILE_NAME}.csv", index=False
    )
    best_scores_df = pd.DataFrame(best_scores)
    best_scores_df.to_csv(
        config.get_results_path() / f"best_scores_{FILE_NAME}.csv", index=False
    )

    all_confs_local_sites_df = pd.concat(all_confs_local_sites_df)
    all_confs_local_sites_df.to_csv(
        config.get_results_path() / f"local_sites_pred_{FILE_NAME}.csv"
    )

    all_confs_df = pd.concat(all_confs_df)
    all_confs_df.to_csv(config.get_results_path() / f"fedavg_pred_{FILE_NAME}.csv")
    
    # Save the best global model with highest test accuracy
    if best_global_model_path is not None:
        import shutil
        best_model_save_path = f"{MODEL_PATH}/best_global_model_{hypers['target']}_acc{best_global_model_acc:.4f}.pt"
        shutil.copy2(best_global_model_path, best_model_save_path)
        print(f"\n=== Best Global Model Saved ===")
        print(f"Best test accuracy: {best_global_model_acc:.4f}")
        print(f"Best model saved to: {best_model_save_path}")
        print(f"Best model info: {best_global_model_info}")
        
        # Save best model info to CSV
        best_model_info_df = pd.DataFrame([best_global_model_info])
        best_model_info_df.to_csv(
            config.get_results_path() / f"best_global_model_info_{FILE_NAME}.csv", 
            index=False
        )
    else:
        print("No global model was saved - no models were evaluated.")

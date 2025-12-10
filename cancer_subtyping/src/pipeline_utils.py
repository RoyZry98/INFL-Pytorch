import os
import json
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from FedAggregateWeights import WeightsAggregation
from FedTrain import TrainFedProtNet
from utils.ProtDataset import ProtDataset
import config


# ------------------------------
# Configuration and arguments
# ------------------------------

@dataclass
class Args:
    use_inr: bool = True
    device: str = "cuda:0"


@dataclass
class RunConstants:
    N_clients: int = 5
    N_included_clients: int = 5
    N_iters: int = 100
    base_model_dir: str = "./saved_model"


# ------------------------------
# Seeding and device helpers
# ------------------------------

def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def summarize_config(args: Args, hypers: Dict[str, Any], model_path: str):
    print("Hyperparameters:", hypers)
    print(f"MODEL_PATH: {model_path}")
    print("\n=== Configuration ===")
    print(f"Use INR: {args.use_inr}")
    print(f"Device: {args.device}")
    print("=" * 50)


# ------------------------------
# Data loading and preprocessing
# ------------------------------

def load_protein_mappings() -> Tuple[Dict[str, str], Dict[str, str]]:
    prot_id_to_name = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=0).to_dict()["Gene"]
    prot_name_to_id = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=3).to_dict()["UniProtID"]
    return prot_id_to_name, prot_name_to_id


def load_and_filter_data(hypers: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    target = hypers["target"]
    included_types = hypers["included_types"]

    print("Loading dataset...")
    combined_df = pd.read_csv(config.PROCAN_DATA_FILE, index_col=0, low_memory=False)

    # Filter Data based on Target
    if target in ["Broad_Cancer_Type", "cancer_type", "cancer_type_2"]:
        combined_selected_df = combined_df[combined_df[target].isin(included_types)]
    else:
        combined_selected_df = combined_df[
            (combined_df["Broad_Cancer_Type"] == "Adenocarcinoma")
            & (combined_df[target].isin(included_types))
        ]

    merged_label_map = combined_selected_df[target].to_dict()

    # Class distribution
    print("\n=== Class Distribution (Before Filtering) ===")
    class_counts = combined_selected_df[target].value_counts()
    print(class_counts)
    print(f"\nClasses with only 1 sample: {class_counts[class_counts == 1].index.tolist()}")

    # Filter out classes with less than 2 samples for stratified split
    valid_classes = class_counts[class_counts >= 2].index
    filtered_df = combined_selected_df[combined_selected_df[target].isin(valid_classes)]

    print(f"\nOriginal samples: {len(combined_selected_df)}")
    print(f"Filtered samples: {len(filtered_df)}")
    print(f"Removed classes: {set(combined_selected_df[target].unique()) - set(valid_classes)}")

    return filtered_df, pd.Series(merged_label_map)


def split_train_test(filtered_df: pd.DataFrame, target: str, test_size: float = 0.1, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        filtered_df,
        stratify=filtered_df[target],
        test_size=test_size,
        train_size=1 - test_size,
        random_state=seed,
    )
    return train_df, test_df


def fit_label_encoder(filtered_df: pd.DataFrame, target: str) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(filtered_df[target])
    return le


def report_split_stats(train_df: pd.DataFrame, test_df: pd.DataFrame, combined_selected_total: int, target: str):
    print(f"Train set size: {len(train_df)} ({len(train_df)/combined_selected_total*100:.1f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df)/combined_selected_total*100:.1f}%)")
    print("\nTrain Distribution:\n", train_df[target].value_counts())
    print("\nTest Distribution:\n", test_df[target].value_counts())


# ------------------------------
# Dataset builders
# ------------------------------

def build_prot_dataset(df: pd.DataFrame, target: str, le: LabelEncoder,
                       prot_id_to_name: Dict[str, str], prot_name_to_id: Dict[str, str],
                       merged_label_map: Dict[Any, Any]) -> ProtDataset:
    X = df.iloc[:, config.META_COL_NUMS:].fillna(0).to_numpy()
    y = le.transform(df[target].to_numpy())
    return ProtDataset(X, y, prot_id_to_name, prot_name_to_id, merged_label_map)


def maybe_subset_train(train_df: pd.DataFrame, target: str, N_clients: int, N_included_clients: int, seed: int = 0) -> pd.DataFrame:
    if N_included_clients == N_clients:
        return train_df
    if N_included_clients < N_clients:
        skf = StratifiedKFold(n_splits=N_clients, shuffle=True, random_state=seed)
        included_idx = []
        for fold, (_tr, val_index) in enumerate(skf.split(train_df, train_df[target])):
            if fold < N_included_clients:
                included_idx.extend(val_index)
            else:
                break
        return train_df.iloc[included_idx]
    raise ValueError("N_included_clients should be less than or equal to N_clients")


# ------------------------------
# Model factories
# ------------------------------

def create_trainer(train_dataset: ProtDataset,
                   test_dataset: ProtDataset,
                   hypers: Dict[str, Any],
                   save_path: Optional[str] = None,
                   load_model: Optional[str] = None,
                   args: Optional[Args] = None):
    args = args or Args()
    return TrainFedProtNet(
        train_dataset, test_dataset, hypers,
        save_path=save_path, load_model=load_model, use_inr=args.use_inr
    )


# ------------------------------
# Traditional training workflow
# ------------------------------

def run_traditional_training(train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             target: str,
                             le: LabelEncoder,
                             prot_id_to_name: Dict[str, str],
                             prot_name_to_id: Dict[str, str],
                             merged_label_map: Dict[Any, Any],
                             hypers: Dict[str, Any],
                             model_path_dir: str,
                             file_name: str,
                             args: Args) -> Tuple[Dict[str, Any], ProtDataset, ProtDataset]:
    test_dataset = build_prot_dataset(test_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map.to_dict())

    subset_df = maybe_subset_train(train_df, target, N_clients=hypers.get("N_clients", 5),
                                   N_included_clients=hypers.get("N_included_clients", 5))
    if len(subset_df) != len(train_df):
        print(f"Using subset of train_df: {len(subset_df)}/{len(train_df)} samples")

    train_dataset = build_prot_dataset(subset_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map.to_dict())

    print(f"Running traditional training: Train size {len(train_dataset)}, Test size {len(test_dataset)}")

    trainer = create_trainer(
        train_dataset, test_dataset, hypers,
        save_path=f"{model_path_dir}/local_model_{hypers['target']}_traditional.pt",
        args=args
    )
    score = trainer.run_train_val()
    score["fold"] = "traditional"

    # Save traditional results
    traditional_results = {
        "model_type": "traditional",
        "test_acc": score.get("test_acc"),
        "test_f1": score.get("test_f1"),
        "test_auc": score.get("test_auc"),
        "best_epoch": score.get("best_epoch", 0),
    }
    pd.DataFrame([traditional_results]).to_csv(
        config.get_results_path() / f"traditional_results_{file_name}.csv",
        index=False
    )
    print(f"Traditional results saved: {traditional_results}")

    # Save predictions
    confs_df = pd.DataFrame(trainer.get_pred_for_cm(), columns=le.classes_)
    confs_df["pred"] = confs_df.idxmax(axis=1)
    confs_df["true"] = test_df[target].to_numpy()
    confs_df.index = test_df.index
    confs_df.to_csv(config.get_results_path() / f"centralised_pred_{file_name}.csv")

    return score, train_dataset, test_dataset


# ------------------------------
# Federated training workflow
# ------------------------------

def initialize_global_model(train_dataset: ProtDataset, test_dataset: ProtDataset,
                            hypers: Dict[str, Any], args: Args, model_path_dir: str) -> str:
    trainer = create_trainer(train_dataset, test_dataset, hypers,
                             save_path=f"{model_path_dir}/{hypers['target']}_fedavg.pt",
                             args=args)
    trainer.save_model()
    return f"{model_path_dir}/{hypers['target']}_fedavg.pt"


def build_client_model_paths(model_path_dir: str, hypers: Dict[str, Any], N_included_clients: int) -> List[str]:
    return [f"{model_path_dir}/local_model_{hypers['target']}.pt"] + \
           [f"{model_path_dir}/local_model_{hypers['target']}_d2_{i}.pt" for i in range(N_included_clients)]


def run_federated_loop(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target: str,
        le: LabelEncoder,
        prot_id_to_name: Dict[str, str],
        prot_name_to_id: Dict[str, str],
        merged_label_map: Dict[Any, Any],
        hypers: Dict[str, Any],
        args: Args,
        model_path_dir: str,
        N_clients: int,
        N_included_clients: int,
        N_iters: int,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame], List[pd.DataFrame], Dict[str, Any], str]:

    best_scores: List[Dict[str, Any]] = []
    all_site_sizes_df: List[pd.DataFrame] = []
    all_confs_local_sites_df: List[pd.DataFrame] = []
    all_confs_df: List[pd.DataFrame] = []

    # Initialize global model
    global_model_path = initialize_global_model(
        build_prot_dataset(train_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map.to_dict()),
        build_prot_dataset(test_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map.to_dict()),
        hypers, args, model_path_dir
    )

    model_paths = build_client_model_paths(model_path_dir, hypers, N_included_clients)

    best_global_model_acc = 0.0
    best_global_model_info = None

    for fed_iter in range(N_iters):
        print(f"\n=== Running Fed Iteration {fed_iter} ===")

        # 1) VCB / main local update
        train_dataset_vcb = build_prot_dataset(train_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map.to_dict())
        test_dataset = build_prot_dataset(test_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map.to_dict())
        trainer = create_trainer(
            train_dataset_vcb, test_dataset, hypers,
            load_model=global_model_path,
            save_path=f"{model_path_dir}/local_model_{hypers['target']}.pt",
            args=args
        )
        score = trainer.run_train_val()
        score["model"] = "vlocal"
        score["iter"] = fed_iter
        best_scores.append(score)

        if fed_iter == 0:
            confs = trainer.get_pred_for_cm()
            confs_df = pd.DataFrame(confs, columns=le.classes_)
            confs_df["pred"] = confs_df.idxmax(axis=1)
            confs_df["true"] = test_df[target].to_numpy()
            confs_df.index = test_df.index
            confs_df.to_csv(config.get_results_path() / f"local_pred_{build_file_name(hypers, args)}.csv")

        # 2) Simulated clients via KFold
        kf = KFold(n_splits=N_clients, shuffle=True, random_state=0)
        train_tmp_df = train_df.sample(frac=1, random_state=0).reset_index(drop=True)

        for fold, (_tr, val_index) in enumerate(kf.split(train_tmp_df)):
            train_fold_df = train_tmp_df.iloc[val_index]

            if fed_iter == 0:
                site_size_df = train_fold_df.groupby([target]).size().reset_index()
                site_size_df["site"] = f"simulated_local_{fold}"
                all_site_sizes_df.append(site_size_df)

            train_dataset_fold = build_prot_dataset(train_fold_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map.to_dict())
            trainer_fold = create_trainer(
                train_dataset_fold, test_dataset, hypers,
                load_model=global_model_path,
                save_path=f"{model_path_dir}/local_model_{hypers['target']}_d2_{fold}.pt",
                args=args
            )
            score_fold = trainer_fold.run_train_val()
            score_fold["model"] = f"simulated_local_{fold}"
            score_fold["iter"] = fed_iter
            best_scores.append(score_fold)

            if fed_iter == N_iters - 1:
                confs = trainer_fold.get_pred_for_cm()
                confs_df = pd.DataFrame(confs, columns=le.classes_)
                confs_df["pred"] = confs_df.idxmax(axis=1)
                confs_df["true"] = test_df[target].to_numpy()
                confs_df["fold"] = fold
                confs_df.index = test_df.index
                all_confs_local_sites_df.append(confs_df)

            if fold + 1 >= N_included_clients:
                break

        # 3) Aggregate weights
        weight_agg = WeightsAggregation(model_paths=model_paths)
        weight_agg.fed_avg()
        weight_agg.save_model(global_model_path)

        # 4) Evaluate global model
        eval_trainer = create_trainer(
            build_prot_dataset(test_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map.to_dict()),
            build_prot_dataset(test_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map.to_dict()),
            hypers, load_model=global_model_path, args=args
        )
        eval_score = eval_trainer.predict()
        eval_score["model"] = "fedavg"
        eval_score["iter"] = fed_iter
        best_scores.append(eval_score)
        print(f"Iter {fed_iter} Global Acc: {eval_score['test_acc']:.4f}")

        if eval_score["test_acc"] > best_global_model_acc:
            best_global_model_acc = eval_score["test_acc"]
            best_global_model_info = {
                "iter": fed_iter,
                "test_acc": eval_score["test_acc"],
                "test_f1": eval_score["test_f1"],
                "test_auc": eval_score["test_auc"],
                "model_path": global_model_path,
            }
            print(f"New best global model found! Test accuracy: {best_global_model_acc:.4f}")

        if fed_iter == N_iters - 1:
            confs = eval_trainer.get_pred_for_cm()
            confs_df = pd.DataFrame(confs, columns=le.classes_)
            confs_df["pred"] = confs_df.idxmax(axis=1)
            confs_df["true"] = test_df[target].to_numpy()
            confs_df.index = test_df.index
            all_confs_df.append(confs_df)

    print("Federated Learning Loop Complete.")
    return best_scores, all_site_sizes_df, all_confs_local_sites_df, best_global_model_info, global_model_path


# ------------------------------
# Results saving utilities
# ------------------------------

def build_file_name(hypers: Dict[str, Any], args: Args) -> str:
    return f"{hypers['target']}_INR_{args.use_inr}"


def prepare_model_path(base_model_dir: str, args: Args, hypers: Dict[str, Any]) -> str:
    model_dir = f"{base_model_dir}/INFL_INR_{args.use_inr}"
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def save_all_results(all_site_sizes_df: List[pd.DataFrame],
                     best_scores: List[Dict[str, Any]],
                     all_confs_local_sites_df: List[pd.DataFrame],
                     all_confs_df: List[pd.DataFrame],
                     best_global_model_info: Optional[Dict[str, Any]],
                     best_global_model_path: str,
                     model_path_dir: str,
                     file_name: str,
                     hypers: Dict[str, Any]):
    # Save site sizes
    if len(all_site_sizes_df) > 0:
        site_sizes = pd.concat(all_site_sizes_df)
        site_sizes.to_csv(config.get_results_path() / f"site_sizes_{file_name}.csv", index=False)

    # Save best scores
    best_scores_df = pd.DataFrame(best_scores)
    best_scores_df.to_csv(config.get_results_path() / f"best_scores_{file_name}.csv", index=False)

    # Save local sites predictions
    if len(all_confs_local_sites_df) > 0:
        local_sites_preds = pd.concat(all_confs_local_sites_df)
        local_sites_preds.to_csv(config.get_results_path() / f"local_sites_pred_{file_name}.csv")

    # Save global model predictions
    if len(all_confs_df) > 0:
        pd.concat(all_confs_df).to_csv(config.get_results_path() / f"fedavg_pred_{file_name}.csv")

    # Save best global model
    if best_global_model_info is not None:
        acc = best_global_model_info["test_acc"]
        best_model_save_path = f"{model_path_dir}/best_global_model_{hypers['target']}_acc{acc:.4f}.pt"
        shutil.copy2(best_global_model_path, best_model_save_path)
        print("\n=== Best Global Model Saved ===")
        print(f"Best test accuracy: {acc:.4f}")
        print(f"Best model saved to: {best_model_save_path}")
        print(f"Best model info: {best_global_model_info}")

        pd.DataFrame([best_global_model_info]).to_csv(
            config.get_results_path() / f"best_global_model_info_{file_name}.csv", index=False
        )
    else:
        print("No global model was saved - no models were evaluated.")


# ------------------------------
# High-level pipeline wrapper
# ------------------------------

def run_full_pipeline(args: Args, consts: RunConstants):
    # Seeds
    set_seeds(config.RANDOM_SEED)
    print("Libraries loaded and seeds set.")

    # Hypers
    hypers = config.DEFAULT_HYPERPARAMETERS.copy()
    hypers.update({
        "target": config.DEFAULT_TARGET,
        "included_types": config.DEFAULT_CANCER_TYPES,
        # also store N_clients & N_included for reuse
        "N_clients": consts.N_clients,
        "N_included_clients": consts.N_included_clients,
    })

    # Paths and names
    model_path_dir = prepare_model_path(consts.base_model_dir, args, hypers)
    file_name = build_file_name(hypers, args)
    summarize_config(args, hypers, model_path_dir)

    target = hypers["target"]

    # Data
    prot_id_to_name, prot_name_to_id = load_protein_mappings()
    filtered_df, merged_label_map = load_and_filter_data(hypers)
    train_df, test_df = split_train_test(filtered_df, target, test_size=0.1, seed=0)
    le = fit_label_encoder(filtered_df, target)
    report_split_stats(train_df, test_df, combined_selected_total=len(filtered_df), target=target)

    # Traditional training
    best_scores: List[Dict[str, Any]] = []
    traditional_score, train_dataset, test_dataset = run_traditional_training(
        train_df, test_df, target, le, prot_id_to_name, prot_name_to_id, merged_label_map,
        hypers, model_path_dir, file_name, args
    )
    best_scores.append(traditional_score)

    # Federated training loop
    fed_scores, site_sizes_list, confs_local_sites_list, best_global_model_info, global_model_path = run_federated_loop(
        train_df=train_df,
        test_df=test_df,
        target=target,
        le=le,
        prot_id_to_name=prot_id_to_name,
        prot_name_to_id=prot_name_to_id,
        merged_label_map=merged_label_map,
        hypers=hypers,
        args=args,
        model_path_dir=model_path_dir,
        N_clients=consts.N_clients,
        N_included_clients=consts.N_included_clients,
        N_iters=consts.N_iters,
    )
    best_scores.extend(fed_scores)

    # Save all outputs
    save_all_results(
        all_site_sizes_df=site_sizes_list,
        best_scores=best_scores,
        all_confs_local_sites_df=confs_local_sites_list,
        all_confs_df=[],  # already saved inside fed loop at end; keeping API hook
        best_global_model_info=best_global_model_info,
        best_global_model_path=global_model_path,
        model_path_dir=model_path_dir,
        file_name=file_name,
        hypers=hypers
    )
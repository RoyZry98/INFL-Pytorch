from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import config
from FedModel import FedProtNet, FedProtNet_DP
from FedTrain import FedProtNet_KeyedINR, make_coordinates
from utils.ProtDataset import ProtDataset


GradList = List[torch.Tensor]

METHOD_ORDER = [
    "FL",
    "FL-DP",
    "PPML",
    "INFL-wrong-key",
    "INFL-without-inr",
]

COLORS = {
    "FL": "#0072B2",
    "FL-DP": "#E69F00",
    "PPML": "#009E73",
    "INFL-wrong-key": "#CC79A7",
    "INFL-without-inr": "#D55E00",
}

METRICS = [
    ("topk_overlap", "Top-50 overlap", "Top-50 overlap"),
    ("best_grad_loss", "Gradient loss", "Best gradient loss"),
    ("pearson_corr", "|Pearson r|", "|Pearson r|"),
    ("spearman_corr", "|Spearman rho|", "|Spearman rho|"),
]

ABS_METRICS = {"pearson_corr", "spearman_corr"}


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_methods(value: str) -> List[str]:
    methods = [x.strip() for x in value.split(",") if x.strip()]
    valid = {"plain", "dp", "ppml", "keyed"}
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {sorted(valid)}")
    return methods


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable. Using CPU.")
        return torch.device("cpu")
    return dev


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._module if hasattr(model, "_module") else model


def set_coordinates_any(model: nn.Module, coords: torch.Tensor) -> None:
    m = unwrap_model(model)
    if hasattr(m, "set_coordinates"):
        m.set_coordinates(coords)


def is_without_inr_shared_param_name(name: str) -> bool:
    excluded_tokens = (
        "key_encoder",
        "key_to_mod",
        "_film",
        ".film",
        "act_proj",
        "fc2_key_bias",
    )
    return not any(token in name for token in excluded_tokens)


def get_trainable_params(
    model: nn.Module,
    without_inr_shared_only: bool = False,
) -> List[torch.Tensor]:
    m = unwrap_model(model)
    if not without_inr_shared_only:
        return [p for p in m.parameters() if p.requires_grad]
    return [
        p
        for name, p in m.named_parameters()
        if p.requires_grad and is_without_inr_shared_param_name(name)
    ]


def load_state_dict_flexible(model: nn.Module, model_path: Optional[str], device: torch.device, allow_untrained: bool) -> bool:
    if model_path is None:
        if allow_untrained:
            print("[warning] No checkpoint path supplied; using randomly initialized model.")
            return False
        raise FileNotFoundError("No checkpoint path supplied. Use --allow_untrained for a smoke run.")

    path = Path(model_path)
    if not path.exists():
        if allow_untrained:
            print(f"[warning] Checkpoint not found: {path}. Using randomly initialized model.")
            return False
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state_dict = torch.load(path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint is not a state_dict dict: {path}")

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[8:] if key.startswith("_module.") else key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warning] Missing keys while loading {path}: {len(missing)}")
    if unexpected:
        print(f"[warning] Unexpected keys while loading {path}: {len(unexpected)}")
    print(f"[loaded] {path}")
    return True


def load_data() -> Dict[str, Any]:
    target = config.DEFAULT_TARGET
    included_types = config.DEFAULT_CANCER_TYPES

    prot_id_to_name = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=0).to_dict()["Gene"]
    prot_name_to_id = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=3).to_dict()["UniProtID"]

    combined_df = pd.read_csv(config.PROCAN_DATA_FILE, index_col=0, low_memory=False)
    if target in ["Broad_Cancer_Type", "cancer_type", "cancer_type_2"]:
        combined_selected_df = combined_df[combined_df[target].isin(included_types)]
    else:
        combined_selected_df = combined_df[
            (combined_df["Broad_Cancer_Type"] == "Adenocarcinoma")
            & (combined_df[target].isin(included_types))
        ]

    class_counts = combined_selected_df[target].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    filtered_df = combined_selected_df[combined_selected_df[target].isin(valid_classes)]

    train_df, test_df = train_test_split(
        filtered_df,
        stratify=filtered_df[target],
        test_size=0.1,
        train_size=0.9,
        random_state=0,
    )

    le = LabelEncoder()
    le.fit(filtered_df[target])
    merged_label_map = combined_selected_df[target].to_dict()

    train_dataset = ProtDataset(
        train_df.iloc[:, config.META_COL_NUMS :].fillna(0).to_numpy(),
        le.transform(train_df[target].to_numpy()),
        prot_id_to_name,
        prot_name_to_id,
        merged_label_map,
    )
    test_dataset = ProtDataset(
        test_df.iloc[:, config.META_COL_NUMS :].fillna(0).to_numpy(),
        le.transform(test_df[target].to_numpy()),
        prot_id_to_name,
        prot_name_to_id,
        merged_label_map,
    )

    return {
        "target": target,
        "train_df": train_df,
        "test_df": test_df,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "le": le,
    }


def build_model(method_type: str, dataset: ProtDataset, num_classes: int, args: argparse.Namespace, device: torch.device, coords: torch.Tensor) -> nn.Module:
    if method_type == "dp":
        return FedProtNet_DP(
            input_dim=dataset.feature_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        ).to(device)

    if method_type == "keyed":
        return FedProtNet_KeyedINR(
            input_dim=dataset.feature_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            coord_dim=args.coord_dim,
            coord_points=args.coord_points,
            key_dim=args.key_dim,
            key_hidden=args.key_hidden,
            key_strength=args.key_strength,
            key_only_fc2_bias=args.key_only_fc2_bias,
            coords=coords,
            device=device,
        ).to(device)

    return FedProtNet(
        input_dim=dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)


def evaluate_model(model: nn.Module, dataset: ProtDataset, batch_size: int, device: torch.device) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_data)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")
    return {"test_acc": float(acc), "test_f1": float(f1), "test_auc": float(auc)}


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def compute_target_gradients(
    model: nn.Module,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    without_inr: bool = False,
    without_inr_shared_only: bool = False,
) -> GradList:
    model.eval()
    model.zero_grad(set_to_none=True)
    params = get_trainable_params(model, without_inr_shared_only=without_inr_shared_only)
    if not params:
        raise RuntimeError("No trainable parameters found.")

    m = unwrap_model(model)
    logits = m.forward_without_inr(target_x) if without_inr and hasattr(m, "forward_without_inr") else model(target_x)
    loss = F.cross_entropy(logits, target_y)
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach().clone() for p, g in zip(params, grads)]


def gradient_matching_loss(pred_grads: GradList, target_grads: GradList, loss_type: str = "l2+cosine", eps: float = 1e-12) -> torch.Tensor:
    if len(pred_grads) != len(target_grads):
        raise ValueError(f"Gradient list length mismatch: pred={len(pred_grads)}, target={len(target_grads)}")

    device = pred_grads[0].device
    l2 = torch.zeros((), device=device)
    numel = 0
    dot = torch.zeros((), device=device)
    pred_norm_sq = torch.zeros((), device=device)
    tgt_norm_sq = torch.zeros((), device=device)

    for pred_grad, target_grad in zip(pred_grads, target_grads):
        target_grad = target_grad.to(device=pred_grad.device, dtype=pred_grad.dtype)
        diff = pred_grad - target_grad
        l2 = l2 + diff.pow(2).sum()
        numel += diff.numel()
        dot = dot + (pred_grad * target_grad).sum()
        pred_norm_sq = pred_norm_sq + pred_grad.pow(2).sum()
        tgt_norm_sq = tgt_norm_sq + target_grad.pow(2).sum()

    l2 = l2 / max(numel, 1)
    cosine = 1.0 - dot / (torch.sqrt(pred_norm_sq) * torch.sqrt(tgt_norm_sq) + eps)
    if loss_type == "l2":
        return l2
    if loss_type == "cosine":
        return cosine
    if loss_type == "l2+cosine":
        return l2 + cosine
    raise ValueError(f"Unknown grad loss type: {loss_type}")


def gradient_inversion_attack_tabular(
    model: nn.Module,
    target_grads: GradList,
    input_dim: int,
    batch_size: int,
    num_classes: int,
    iters: int,
    lr: float,
    device: torch.device,
    grad_loss_type: str,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    l2_x_prior_weight: float = 0.0,
    without_inr: bool = False,
    without_inr_shared_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    model.eval()
    params = get_trainable_params(model, without_inr_shared_only=without_inr_shared_only)
    if not params:
        raise RuntimeError("No trainable parameters found for GIA.")

    dummy_x = nn.Parameter(torch.randn(batch_size, input_dim, device=device) * 0.01)
    dummy_label_logits = nn.Parameter(torch.zeros(batch_size, num_classes, device=device))
    nn.init.normal_(dummy_label_logits, mean=0.0, std=0.01)
    optimizer = torch.optim.Adam([dummy_x, dummy_label_logits], lr=lr)

    loss_log: List[float] = []
    best_loss = float("inf")
    best_x = None
    best_label_logits = None

    for _ in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        if clamp_min is not None or clamp_max is not None:
            x_for_model = dummy_x.clamp(
                min=clamp_min if clamp_min is not None else -float("inf"),
                max=clamp_max if clamp_max is not None else float("inf"),
            )
        else:
            x_for_model = dummy_x

        label_prob = F.softmax(dummy_label_logits, dim=1)
        m = unwrap_model(model)
        logits = m.forward_without_inr(x_for_model) if without_inr and hasattr(m, "forward_without_inr") else model(x_for_model)
        cls_loss = soft_cross_entropy(logits, label_prob)
        raw_pred_grads = torch.autograd.grad(cls_loss, params, create_graph=True, retain_graph=True, allow_unused=True)
        pred_grads = [torch.zeros_like(p) if g is None else g for p, g in zip(params, raw_pred_grads)]
        grad_loss = gradient_matching_loss(pred_grads, target_grads, loss_type=grad_loss_type)
        prior_loss = x_for_model.pow(2).mean() if l2_x_prior_weight > 0 else torch.zeros((), device=device)
        total_loss = grad_loss + float(l2_x_prior_weight) * prior_loss
        total_loss.backward()
        optimizer.step()

        grad_loss_value = float(grad_loss.detach().cpu().item())
        loss_log.append(grad_loss_value)
        if grad_loss_value < best_loss:
            best_loss = grad_loss_value
            with torch.no_grad():
                best_x = x_for_model.detach().cpu().clone()
                best_label_logits = dummy_label_logits.detach().cpu().clone()

    if best_x is None or best_label_logits is None:
        best_x = dummy_x.detach().cpu()
        best_label_logits = dummy_label_logits.detach().cpu()
    return best_x, best_label_logits.argmax(dim=1).cpu(), loss_log


def run_attack_case_tabular(
    model: nn.Module,
    target_grads: GradList,
    target_x: torch.Tensor,
    true_label: int,
    input_dim: int,
    num_classes: int,
    seed: int,
    args: argparse.Namespace,
    without_inr: bool = False,
    without_inr_shared_only: bool = False,
) -> Dict[str, Any]:
    seed_everything(seed)
    rec_x, rec_label, loss_log = gradient_inversion_attack_tabular(
        model=model,
        target_grads=target_grads,
        input_dim=input_dim,
        batch_size=target_x.shape[0],
        num_classes=num_classes,
        iters=args.gia_iters,
        lr=args.gia_lr,
        device=target_x.device,
        grad_loss_type=args.gia_grad_loss,
        clamp_min=args.gia_clamp_min,
        clamp_max=args.gia_clamp_max,
        l2_x_prior_weight=args.gia_l2_x_prior_weight,
        without_inr=without_inr,
        without_inr_shared_only=without_inr_shared_only,
    )
    rec_label_int = int(rec_label.item())
    return {
        "rec_x": rec_x,
        "label": rec_label_int,
        "label_correct": rec_label_int == int(true_label),
        "loss_log": loss_log,
        "best_grad_loss": float(np.min(loss_log)) if loss_log else float("nan"),
        "final_grad_loss": float(loss_log[-1]) if loss_log else float("nan"),
    }


def compute_reconstruction_metrics(rec_x: torch.Tensor, target_x: torch.Tensor, feature_std: np.ndarray, top_k: int) -> Dict[str, float]:
    rec = rec_x.detach().cpu().numpy().reshape(-1).astype(np.float64)
    target = target_x.detach().cpu().numpy().reshape(-1).astype(np.float64)
    safe_std = np.where(feature_std > 1e-8, feature_std, 1.0)
    diff = rec - target
    norm_diff = diff / safe_std
    input_mse = float(np.mean(diff ** 2))
    normalized_mse = float(np.mean(norm_diff ** 2))
    normalized_rmse = float(np.sqrt(normalized_mse))
    pearson_corr = float(np.corrcoef(rec, target)[0, 1]) if np.std(rec) > 1e-12 and np.std(target) > 1e-12 else float("nan")
    rec_rank = pd.Series(rec).rank(method="average").to_numpy()
    target_rank = pd.Series(target).rank(method="average").to_numpy()
    spearman_corr = float(np.corrcoef(rec_rank, target_rank)[0, 1]) if np.std(rec_rank) > 1e-12 and np.std(target_rank) > 1e-12 else float("nan")
    k = min(int(top_k), len(target))
    rec_top = set(np.argsort(np.abs(rec))[-k:])
    target_top = set(np.argsort(np.abs(target))[-k:])
    topk_overlap = float(len(rec_top & target_top) / max(k, 1))
    return {
        "input_mse": input_mse,
        "normalized_mse": normalized_mse,
        "normalized_rmse": normalized_rmse,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "topk_overlap": topk_overlap,
    }


def infer_label_from_fc2_bias(model: nn.Module, grads: GradList, shared_only: bool = False) -> int:
    named_params = [
        (name, p)
        for name, p in unwrap_model(model).named_parameters()
        if p.requires_grad and (not shared_only or is_without_inr_shared_param_name(name))
    ]
    for (name, _), grad in zip(named_params, grads):
        if name == "fc2.bias":
            return int(grad.detach().cpu().float().argmin().item())
    return -1


def method_label(method: str, attack_case: str, ppml_source: str, fldp_source: str) -> Optional[str]:
    if method == "plain-fedavg" and attack_case == "standard":
        return "FL"
    if method == fldp_source and attack_case == "standard":
        return "FL-DP"
    if method == ppml_source and attack_case == "standard":
        return "PPML"
    if method == "keyed-inr" and attack_case == "wrong-coordinate":
        return "INFL-wrong-key"
    if method == "keyed-inr" and attack_case == "without-inr":
        return "INFL-without-inr"
    return None


def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.6, length=2.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)


def prepare_plot_df(df: pd.DataFrame, ppml_source: str, fldp_source: str) -> pd.DataFrame:
    work = df.copy()
    work["display_method"] = [
        method_label(row.method, row.attack_case, ppml_source, fldp_source)
        for row in work.itertuples(index=False)
    ]
    work = work[work["display_method"].notna()].copy()
    work["display_method"] = pd.Categorical(work["display_method"], categories=METHOD_ORDER, ordered=True)
    return work.sort_values("display_method")


def present_methods(df: pd.DataFrame) -> List[str]:
    present = set(df["display_method"].dropna().astype(str))
    return [method for method in METHOD_ORDER if method in present]


def display_method(method: str) -> str:
    return method.replace("INFL-", "INFL\n")


def metric_values(df: pd.DataFrame, metric: str) -> np.ndarray:
    values = df[metric].dropna().to_numpy(dtype=float)
    if metric in ABS_METRICS:
        values = np.abs(values)
    return values


def mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean, mean
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    half = 1.96 * sem
    return mean, mean - half, mean + half


def save_all_figures(fig: plt.Figure, path: Path, dpi: int) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")


def save_metric_boxplots(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int) -> List[Path]:
    methods = present_methods(df)
    paths: List[Path] = []
    if not methods:
        return paths
    for metric, title, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(3.25, 2.35))
        data = [metric_values(df[df["display_method"] == method], metric) for method in methods]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55)
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.85)
            patch.set_edgecolor("#333333")
        for key in ["whiskers", "caps", "medians"]:
            for line in bp[key]:
                line.set_color("#333333")
                line.set_linewidth(0.6)
        ax.set_title(title, pad=3)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels([display_method(m) for m in methods], rotation=25, ha="right")
        finish_axes(ax)
        fig.tight_layout()
        path = out_dir / f"{prefix}_{metric}_boxplot.svg"
        save_all_figures(fig, path, dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def save_metric_ci_bars(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int) -> List[Path]:
    methods = present_methods(df)
    paths: List[Path] = []
    if not methods:
        return paths
    x = np.arange(len(methods))
    for metric, title, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(3.25, 2.35))
        means, lows, highs = [], [], []
        for method in methods:
            mean, low, high = mean_ci(metric_values(df[df["display_method"] == method], metric))
            means.append(mean)
            lows.append(low)
            highs.append(high)
        yerr = np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
        ax.bar(
            x,
            means,
            yerr=yerr,
            capsize=2.5,
            color=[COLORS[m] for m in methods],
            alpha=0.9,
            width=0.65,
            edgecolor="#333333",
            linewidth=0.5,
            error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#333333"},
        )
        ax.set_title(title, pad=3)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([display_method(m) for m in methods], rotation=25, ha="right")
        finish_axes(ax)
        fig.tight_layout()
        path = out_dir / f"{prefix}_{metric}_ci_bar.svg"
        save_all_figures(fig, path, dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def save_loss_curves(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int) -> Optional[Path]:
    methods = present_methods(df)
    if not methods:
        return None
    grouped = df.groupby(["display_method", "iteration"], observed=False)["grad_loss"].agg(["mean", "std", "count"]).reset_index()
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
    grouped["ci_low"] = grouped["mean"] - 1.96 * grouped["sem"].fillna(0.0)
    grouped["ci_high"] = grouped["mean"] + 1.96 * grouped["sem"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for method in methods:
        mdf = grouped[grouped["display_method"] == method].dropna(subset=["mean"])
        if mdf.empty:
            continue
        x = mdf["iteration"].to_numpy(dtype=float)
        mean = mdf["mean"].to_numpy(dtype=float)
        low = mdf["ci_low"].to_numpy(dtype=float)
        high = mdf["ci_high"].to_numpy(dtype=float)
        ax.plot(x, mean, label=method, color=COLORS[method], linewidth=1.2)
        ax.fill_between(x, low, high, color=COLORS[method], alpha=0.12, linewidth=0)
    ax.set_xlabel("GIA iteration")
    ax.set_ylabel("Gradient loss")
    ax.set_title("GIA optimization loss", pad=3)
    finish_axes(ax)
    ax.legend(frameon=False, handlelength=1.4)
    fig.tight_layout()
    path = out_dir / f"{prefix}_loss_curves.svg"
    save_all_figures(fig, path, dpi)
    plt.close(fig)
    return path


def save_privacy_radar(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int) -> Optional[Path]:
    methods = present_methods(df)
    if not methods:
        return None
    metrics = ["best_grad_loss", "topk_protection", "corr_protection", "spearman_protection"]
    labels = ["Grad loss", "1 - TopK", "1 - |Pearson|", "1 - |Spearman|"]
    work = df.copy()
    work["topk_protection"] = 1.0 - work["topk_overlap"].clip(0, 1)
    work["corr_protection"] = 1.0 - work["pearson_corr"].abs().clip(0, 1)
    work["spearman_protection"] = 1.0 - work["spearman_corr"].abs().clip(0, 1)
    means = work.groupby("display_method", observed=False)[metrics].mean().reindex(methods)
    scaled = means.copy()
    for metric in metrics:
        vals = means[metric].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            scaled[metric] = 0.5
            continue
        low = float(np.min(finite))
        high = float(np.max(finite))
        scaled[metric] = (means[metric] - low) / (high - low) if high > low else 0.5
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(3.2, 3.2))
    ax = plt.subplot(111, polar=True)
    for method in methods:
        values = scaled.loc[method, metrics].to_numpy(dtype=float).tolist()
        values += values[:1]
        ax.plot(angles, values, color=COLORS[method], linewidth=1.1, label=method)
        ax.fill(angles, values, color=COLORS[method], alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.set_title("Relative GIA Privacy Profile", y=1.10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.10), frameon=False, handlelength=1.4)
    path = out_dir / f"{prefix}_privacy_radar.svg"
    save_all_figures(fig, path, dpi)
    plt.close(fig)
    return path


def save_method_table(df: pd.DataFrame, out_dir: Path, prefix: str) -> Path:
    rows = []
    for method in present_methods(df):
        mdf = df[df["display_method"] == method]
        row: Dict[str, Any] = {"display_method": method, "n_attacks": int(len(mdf))}
        for metric in ["pearson_corr", "spearman_corr", "topk_overlap", "best_grad_loss"]:
            mean, low, high = mean_ci(metric_values(mdf, metric))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    path = out_dir / f"{prefix}_visualization_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_visualizations(summary_df: pd.DataFrame, loss_df: pd.DataFrame, out_dir: Path, prefix: str, args: argparse.Namespace) -> List[Path]:
    apply_nature_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_summary = prepare_plot_df(summary_df, args.ppml_source, args.fldp_source)
    plot_loss = prepare_plot_df(loss_df, args.ppml_source, args.fldp_source)
    if plot_summary.empty:
        print("[warning] No mapped methods available for visualization.")
        return []
    paths: List[Path] = []
    paths.extend(save_metric_boxplots(plot_summary, out_dir, prefix, args.dpi))
    paths.extend(save_metric_ci_bars(plot_summary, out_dir, prefix, args.dpi))
    loss_path = save_loss_curves(plot_loss, out_dir, prefix, args.dpi)
    radar_path = save_privacy_radar(plot_summary, out_dir, prefix, args.dpi)
    if loss_path is not None:
        paths.append(loss_path)
    if radar_path is not None:
        paths.append(radar_path)
    paths.append(save_method_table(plot_summary, out_dir, prefix))
    mapping_path = out_dir / f"{prefix}_method_mapping.csv"
    plot_summary[["display_method", "method", "attack_case", "epsilon"]].drop_duplicates().sort_values("display_method").to_csv(mapping_path, index=False)
    paths.append(mapping_path)
    return paths


def get_args() -> argparse.Namespace:
    project_root = config.PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Run cancer subtyping GIA and create gia_visualizations outputs.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--methods", type=str, default="keyed", help="Comma list: plain,dp,ppml,keyed")
    parser.add_argument("--target_indices", type=str, default="0")
    parser.add_argument("--seeds", type=str, default="5001")
    parser.add_argument("--gia_iters", type=int, default=30)
    parser.add_argument("--gia_lr", type=float, default=0.1)
    parser.add_argument("--gia_grad_loss", type=str, default="l2+cosine", choices=["l2", "cosine", "l2+cosine"])
    parser.add_argument("--gia_clamp_min", type=float, default=None)
    parser.add_argument("--gia_clamp_max", type=float, default=None)
    parser.add_argument("--gia_l2_x_prior_weight", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--coord_dim", type=int, default=2)
    parser.add_argument("--coord_points", type=int, default=128)
    parser.add_argument("--correct_coord_seed", type=int, default=0)
    parser.add_argument("--wrong_coord_seed", type=int, default=config.RANDOM_SEED + 12345)
    parser.add_argument("--correct_coord_mode", type=str, default="uniform", choices=["uniform", "normal", "ones", "zeros", "negones", "constant"])
    parser.add_argument("--wrong_coord_mode", type=str, default="normal", choices=["uniform", "normal", "ones", "zeros", "negones", "constant"])
    parser.add_argument("--coord_constant", type=float, default=1.0)
    parser.add_argument("--key_dim", type=int, default=16)
    parser.add_argument("--key_hidden", type=int, default=8)
    parser.add_argument("--key_strength", type=float, default=20.0)
    parser.add_argument("--key_only_fc2_bias", action="store_true")
    parser.add_argument("--plain_model_path", type=str, default=str(project_root / "saved_model/INFL_INR_False/cancer_type_fedavg.pt"))
    parser.add_argument("--dp_model_path", type=str, default=None)
    parser.add_argument("--ppml_model_path", type=str, default=None)
    parser.add_argument("--keyed_model_path", type=str, default=str(project_root / "saved_model/INFL_INR_True/cancer_type_fedavg.pt"))
    parser.add_argument("--dp_epsilon", type=float, default=1.29)
    parser.add_argument("--ppml_epsilon", type=float, default=2.36)
    parser.add_argument("--allow_untrained", action="store_true", help="Use randomly initialized models when checkpoints are missing. Useful for a smoke test only.")
    parser.add_argument("--output_dir", type=str, default=str(config.RESULTS_DIR / "gia_visualizations"))
    parser.add_argument("--prefix", type=str, default="gia_methods_keydim16_keyhidden8_strength20")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ppml_source", type=str, default="ppml", choices=["ppml", "dp-fedavg"])
    parser.add_argument("--fldp_source", type=str, default="dp-fedavg", choices=["ppml", "dp-fedavg"])
    return parser.parse_args()


def main() -> None:
    args = get_args()
    seed_everything(config.RANDOM_SEED)
    methods = parse_methods(args.methods)
    target_indices = parse_int_list(args.target_indices)
    seeds = parse_int_list(args.seeds)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    train_df = data["train_df"]
    test_df = data["test_df"]
    test_dataset = data["test_dataset"]
    le = data["le"]
    target_name = data["target"]
    num_classes = len(le.classes_)
    feature_std = train_df.iloc[:, config.META_COL_NUMS :].fillna(0).to_numpy(dtype=np.float64).std(axis=0)

    correct_coords = make_coordinates(
        seed=args.correct_coord_seed,
        mode=args.correct_coord_mode,
        num_points=args.coord_points,
        coord_dim=args.coord_dim,
        device=device,
        constant=args.coord_constant,
    )
    wrong_coords = make_coordinates(
        seed=args.wrong_coord_seed,
        mode=args.wrong_coord_mode,
        num_points=args.coord_points,
        coord_dim=args.coord_dim,
        device=device,
        constant=args.coord_constant,
    )

    specs = []
    if "plain" in methods:
        specs.append({"method": "plain-fedavg", "type": "plain", "path": args.plain_model_path, "epsilon": np.nan})
    if "dp" in methods:
        specs.append({"method": "dp-fedavg", "type": "dp", "path": args.dp_model_path, "epsilon": float(args.dp_epsilon)})
    if "ppml" in methods:
        specs.append({"method": "ppml", "type": "plain", "path": args.ppml_model_path, "epsilon": float(args.ppml_epsilon)})
    if "keyed" in methods:
        specs.append({"method": "keyed-inr", "type": "keyed", "path": args.keyed_model_path, "epsilon": np.nan})

    models: Dict[str, nn.Module] = {}
    accuracy_rows = []
    for spec in specs:
        coords = correct_coords if spec["type"] == "keyed" else correct_coords
        model = build_model(spec["type"], test_dataset, num_classes, args, device, coords)
        loaded = load_state_dict_flexible(model, spec["path"], device, args.allow_untrained)
        if spec["type"] == "keyed":
            set_coordinates_any(model, correct_coords)
        metrics = evaluate_model(model, test_dataset, args.batch_size, device)
        models[spec["method"]] = model
        accuracy_rows.append({"method": spec["method"], "epsilon": spec["epsilon"], "model_path": spec["path"], "checkpoint_loaded": loaded, **metrics})
        print(f"[accuracy] {spec['method']}: {metrics}")

    input_dim = int(test_dataset.feature_dim)
    attack_rows = []
    loss_rows = []

    for raw_idx in target_indices:
        target_index = int(raw_idx % len(test_df))
        target_row = test_df.iloc[target_index]
        target_x_np = target_row.iloc[config.META_COL_NUMS :].fillna(0).to_numpy(dtype=np.float32)
        target_label_name = str(target_row[target_name])
        target_label = int(le.transform([target_label_name])[0])
        target_x = torch.tensor(target_x_np, dtype=torch.float32, device=device).unsqueeze(0)
        target_y = torch.tensor([target_label], dtype=torch.long, device=device)
        print(f"\n[target] index={target_index} sample={target_row.name} label={target_label_name}")

        for spec in specs:
            model = models[spec["method"]].to(device).eval()
            if spec["type"] == "keyed":
                set_coordinates_any(model, correct_coords)
                target_grads = compute_target_gradients(model, target_x, target_y, without_inr=False)
                shared_target_grads = compute_target_gradients(model, target_x, target_y, without_inr=False, without_inr_shared_only=True)
                attack_cases = [
                    ("correct-coordinate", target_grads, False, False, correct_coords),
                    ("wrong-coordinate", target_grads, False, False, wrong_coords),
                    ("without-inr", shared_target_grads, True, True, correct_coords),
                ]
            else:
                target_grads = compute_target_gradients(model, target_x, target_y, without_inr=False)
                attack_cases = [("standard", target_grads, False, False, None)]

            inferred_label = infer_label_from_fc2_bias(model, target_grads)
            for seed in seeds:
                for attack_case, grads, without_inr, shared_only, coords in attack_cases:
                    if coords is not None:
                        set_coordinates_any(model, coords)
                    result = run_attack_case_tabular(
                        model=model,
                        target_grads=grads,
                        target_x=target_x,
                        true_label=target_label,
                        input_dim=input_dim,
                        num_classes=num_classes,
                        seed=seed,
                        args=args,
                        without_inr=without_inr,
                        without_inr_shared_only=shared_only,
                    )
                    metrics = compute_reconstruction_metrics(result["rec_x"], target_x, feature_std, args.top_k)
                    rec_label_id = int(result["label"])
                    attack_rows.append(
                        {
                            "method": spec["method"],
                            "attack_case": attack_case,
                            "epsilon": spec["epsilon"],
                            "target_index": target_index,
                            "target_sample_id": str(target_row.name),
                            "true_label_name": target_label_name,
                            "true_label_id": int(target_label),
                            "seed": int(seed),
                            "optimized_rec_label_id": rec_label_id,
                            "optimized_label_correct": rec_label_id == target_label,
                            "inferred_rec_label_id": int(inferred_label),
                            "inferred_label_correct": int(inferred_label) == target_label,
                            "input_mse": metrics["input_mse"],
                            "normalized_mse": metrics["normalized_mse"],
                            "normalized_rmse": metrics["normalized_rmse"],
                            "pearson_corr": metrics["pearson_corr"],
                            "spearman_corr": metrics["spearman_corr"],
                            "topk_overlap": metrics["topk_overlap"],
                            "best_grad_loss": float(result["best_grad_loss"]),
                            "final_grad_loss": float(result["final_grad_loss"]),
                        }
                    )
                    for iteration, grad_loss in enumerate(result["loss_log"], start=1):
                        loss_rows.append(
                            {
                                "method": spec["method"],
                                "attack_case": attack_case,
                                "epsilon": spec["epsilon"],
                                "target_index": target_index,
                                "seed": int(seed),
                                "iteration": int(iteration),
                                "grad_loss": float(grad_loss),
                            }
                        )
                    print(
                        f"  {spec['method']}/{attack_case} seed={seed}: "
                        f"label={rec_label_id == target_label} mse={metrics['input_mse']:.4f} "
                        f"grad={float(result['best_grad_loss']):.4f}"
                    )

    accuracy_df = pd.DataFrame(accuracy_rows)
    summary_df = pd.DataFrame(attack_rows)
    loss_df = pd.DataFrame(loss_rows)

    accuracy_path = out_dir / f"{args.prefix}_accuracy.csv"
    summary_path = out_dir / f"{args.prefix}_summary.csv"
    loss_path = out_dir / f"{args.prefix}_loss_curves.csv"
    metadata_path = out_dir / f"{args.prefix}_metadata.json"
    accuracy_df.to_csv(accuracy_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    loss_df.to_csv(loss_path, index=False)
    plot_paths = make_visualizations(summary_df, loss_df, out_dir, args.prefix, args)

    metadata = {
        "methods": methods,
        "target_indices": target_indices,
        "seeds": seeds,
        "gia_iters": int(args.gia_iters),
        "gia_lr": float(args.gia_lr),
        "gia_grad_loss": args.gia_grad_loss,
        "top_k": int(args.top_k),
        "coord_dim": int(args.coord_dim),
        "coord_points": int(args.coord_points),
        "correct_coord_seed": int(args.correct_coord_seed),
        "wrong_coord_seed": int(args.wrong_coord_seed),
        "key_dim": int(args.key_dim),
        "key_hidden": int(args.key_hidden),
        "key_strength": float(args.key_strength),
        "key_only_fc2_bias": bool(args.key_only_fc2_bias),
        "allow_untrained": bool(args.allow_untrained),
        "classes": [str(x) for x in le.classes_],
        "outputs": {
            "accuracy": str(accuracy_path),
            "summary": str(summary_path),
            "loss_curves": str(loss_path),
            "plots": [str(path) for path in plot_paths],
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[saved] {accuracy_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {loss_path}")
    for path in plot_paths:
        print(f"[saved] {path}")
    print(f"[saved] {metadata_path}")


if __name__ == "__main__":
    main()

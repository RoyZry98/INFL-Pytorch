from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent


def task_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else TASK_ROOT / resolved


def load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.nanmean((pred - target) ** 2))


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    normalized = np.zeros_like(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        normalized[:] = np.nan
        return normalized
    lo = float(np.nanmin(values[valid]))
    hi = float(np.nanmax(values[valid]))
    if hi <= lo:
        return normalized
    normalized[valid] = (values[valid] - lo) / (hi - lo)
    normalized[~valid] = np.nan
    return normalized


def normalize_by_group(values: np.ndarray, group_info) -> np.ndarray:
    normalized = np.zeros_like(values, dtype=float)
    for group in group_info:
        mask = group["mask"]
        normalized[mask] = minmax_normalize(values[mask])
    return normalized


def packed_coordinate_groups(patient: np.ndarray, section: np.ndarray, pixel: np.ndarray, pad: float = 0.08):
    groups = sorted(set(zip(patient.tolist(), section.tolist())))
    ncols = int(np.ceil(np.sqrt(len(groups))))
    packed = np.zeros((pixel.shape[0], 2), dtype=float)
    group_info = []
    for idx, group in enumerate(groups):
        mask = (patient == group[0]) & (section == group[1])
        coords = pixel[mask].astype(float)
        if coords.size == 0:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = np.maximum(maxs - mins, 1.0)
        local = (coords - mins) / span
        local[:, 1] = 1.0 - local[:, 1]
        row = idx // ncols
        col = idx % ncols
        x0 = col * (1.0 + pad)
        y0 = row * (1.0 + pad)
        packed[mask, 0] = x0 + local[:, 0]
        packed[mask, 1] = y0 + local[:, 1]
        group_info.append({
            "patient": str(group[0]),
            "section": str(group[1]),
            "mask": mask,
            "text_x": x0 + 0.03,
            "text_y": y0 + 0.97,
        })
    return packed, group_info


def packed_coordinates(patient: np.ndarray, section: np.ndarray, pixel: np.ndarray, pad: float = 0.08) -> np.ndarray:
    packed, _ = packed_coordinate_groups(patient, section, pixel, pad=pad)
    return packed


def annotate_group_mse(ax, group_info, values: np.ndarray, true_values: np.ndarray) -> None:
    for group in group_info:
        mask = group["mask"]
        if not np.any(mask):
            continue
        group_mse = mse(values[mask], true_values[mask])
        ax.text(
            group["text_x"],
            group["text_y"],
            f"MSE={group_mse:.3f}",
            color="white",
            fontsize=5.2,
            ha="left",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 0.8},
        )


def section_mse_rows(method: str, gene: str, data, gene_idx: int) -> list[dict[str, object]]:
    patient = data["patient"].astype(str)
    section = data["section"].astype(str)
    true_values = data["counts"][:, gene_idx]
    pred_values = data["predictions"][:, gene_idx]
    rows = []
    for patient_id, section_id in sorted(set(zip(patient.tolist(), section.tolist()))):
        mask = (patient == patient_id) & (section == section_id)
        rows.append({
            "method": method,
            "gene": gene,
            "patient": patient_id,
            "section": section_id,
            "spots": int(mask.sum()),
            "mse": mse(pred_values[mask], true_values[mask]),
        })
    return rows


def find_gene(gene_names: np.ndarray, gene: str) -> int:
    names = np.asarray(gene_names).astype(str)
    matches = np.where(names == gene)[0]
    if len(matches) == 0:
        matches = np.where(np.char.upper(names) == gene.upper())[0]
    if len(matches) == 0:
        raise ValueError(f"Gene {gene} not found")
    return int(matches[0])


def parse_method_paths(method_paths: str) -> dict[str, Path]:
    paths = {}
    for item in method_paths.split(","):
        item = item.strip()
        if not item:
            continue
        method, path = item.split("=", 1)
        paths[method.strip()] = task_path(path.strip())
    return paths


def save_pred_vs_true(outdir: Path, method: str, epoch: str, counts: np.ndarray, preds: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(counts.ravel(), preds.ravel(), s=2, alpha=0.12, linewidths=0)
    lo = float(np.nanmin([counts.min(), preds.min()]))
    hi = float(np.nanmax([counts.max(), preds.max()]))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title(f"{method} epoch {epoch}")
    ax.set_xlabel("True log expression")
    ax.set_ylabel("Predicted log expression")
    ax.text(0.02, 0.98, f"MSE={mse(preds, counts):.6f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(outdir / f"{method}_epoch{epoch}_pred_vs_true.png", dpi=220)
    fig.savefig(outdir / f"{method}_epoch{epoch}_pred_vs_true.pdf")
    plt.close(fig)


def save_fasn_combined(outdir: Path, epoch: str, gene: str, loaded: list[tuple[str, object, int]]) -> None:
    fig, axes = plt.subplots(len(loaded), 2, figsize=(7.2, 3.0 * len(loaded)), constrained_layout=True)
    if len(loaded) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, (method, data, gene_idx) in enumerate(loaded):
        patient = data["patient"].astype(str)
        section = data["section"].astype(str)
        coords, group_info = packed_coordinate_groups(patient, section, data["pixel"])
        true_values = data["counts"][:, gene_idx]
        pred_values = data["predictions"][:, gene_idx]
        for col, (values, label) in enumerate([
            (true_values, "true"),
            (pred_values, "pred"),
        ]):
            panel_mse = mse(values, true_values)
            plot_values = normalize_by_group(values, group_info)
            sc = axes[row, col].scatter(coords[:, 0], coords[:, 1], c=plot_values, s=9, marker="s", cmap="magma", vmin=0.0, vmax=1.0, linewidths=0)
            axes[row, col].set_aspect("equal")
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{method} {gene} {label}\nMSE={panel_mse:.4f}")
            if label == "pred":
                annotate_group_mse(axes[row, col], group_info, values, true_values)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.65, label=f"section-normalized {gene} expression")
    fig.savefig(outdir / f"spatial_epoch{epoch}_{gene}_packed_heatmap.png", dpi=260)
    fig.savefig(outdir / f"spatial_epoch{epoch}_{gene}_packed_heatmap.pdf")
    plt.close(fig)


def save_method_fasn(outdir: Path, method: str, epoch: str, gene: str, data, gene_idx: int) -> None:
    patient = data["patient"].astype(str)
    section = data["section"].astype(str)
    coords, group_info = packed_coordinate_groups(patient, section, data["pixel"])
    true_values = data["counts"][:, gene_idx]
    pred_values = data["predictions"][:, gene_idx]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for ax, values, title in [(axes[0], true_values, "true"), (axes[1], pred_values, "pred")]:
        panel_mse = mse(values, true_values)
        plot_values = normalize_by_group(values, group_info)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=plot_values, s=9, marker="s", cmap="magma", vmin=0.0, vmax=1.0, linewidths=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{method} {gene} {title}\nMSE={panel_mse:.4f}")
        if title == "pred":
            annotate_group_mse(ax, group_info, values, true_values)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.78, label=f"section-normalized {gene} expression")
    fig.savefig(outdir / f"{method}_epoch{epoch}_{gene}_packed_heatmap.png", dpi=260)
    fig.savefig(outdir / f"{method}_epoch{epoch}_{gene}_packed_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize spatial transcriptomics predictions and packed FASN heatmaps.")
    parser.add_argument("--root", default="spatial_fed_e5_c30_fixedsplit")
    parser.add_argument("--outdir", default="evaluation_spatial_fed_e5_c30_fixedsplit")
    parser.add_argument("--epoch", default="5")
    parser.add_argument("--gene", default="FASN")
    parser.add_argument("--methods", default="normal,dp,ppml,inr")
    parser.add_argument("--method-paths", default="", help="Optional comma-separated method=npz_path overrides")
    args = parser.parse_args()

    root = task_path(args.root)
    outdir = task_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    method_paths = parse_method_paths(args.method_paths)
    rows = []
    section_rows = []
    loaded = []
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        path = method_paths.get(method, root / method / "pred_root" / f"{args.epoch}.npz")
        data = load_npz(path)
        counts = data["counts"]
        preds = data["predictions"]
        gene_idx = find_gene(data["gene_names"], args.gene)
        row = {
            "method": method,
            "path": str(path),
            "spots": int(counts.shape[0]),
            "genes": int(counts.shape[1]),
            "nan_predictions": int(np.isnan(preds).sum()),
            "pred_mse": mse(preds, counts),
            f"{args.gene}_mse": mse(preds[:, gene_idx], counts[:, gene_idx]),
        }
        rows.append(row)
        section_rows.extend(section_mse_rows(method, args.gene, data, gene_idx))
        loaded.append((method, data, gene_idx))
        save_pred_vs_true(outdir, method, args.epoch, counts, preds)
        save_method_fasn(outdir, method, args.epoch, args.gene, data, gene_idx)
    save_fasn_combined(outdir, args.epoch, args.gene, loaded)
    with open(outdir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(outdir / "metrics.md", "w") as handle:
        handle.write("| method | spots | genes | nan_predictions | pred_mse | " + args.gene + "_mse |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(f"| {row['method']} | {row['spots']} | {row['genes']} | {row['nan_predictions']} | {row['pred_mse']} | {row[args.gene + '_mse']} |\n")
    section_fields = ["method", "gene", "patient", "section", "spots", "mse"]
    with open(outdir / f"{args.gene}_section_mse.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=section_fields)
        writer.writeheader()
        writer.writerows(section_rows)
    with open(outdir / f"{args.gene}_section_mse.md", "w") as handle:
        handle.write("| method | gene | patient | section | spots | mse |\n")
        handle.write("|---|---|---|---|---:|---:|\n")
        for row in section_rows:
            handle.write(f"| {row['method']} | {row['gene']} | {row['patient']} | {row['section']} | {row['spots']} | {row['mse']} |\n")
    print(f"outputs written to {outdir}")


if __name__ == "__main__":
    main()

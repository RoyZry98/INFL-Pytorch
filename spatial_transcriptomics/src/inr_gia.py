from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Subset

import gia_core as demo
from train import load_attack_defs


class ActivationFilmLeakyLeNetST(nn.Module):
    def __init__(self, outputs: int, args: argparse.Namespace, coords: torch.Tensor, defs: Dict) -> None:
        super().__init__()
        self.key_encoder = defs["CoordinateKeyEncoder"](
            args.key_coord_dim,
            args.key_coord_points,
            args.key_dim,
            coords,
        )
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.Sigmoid(),
        )
        base_fc = nn.Linear(12 * 112 * 112, outputs)
        self.fc = defs["KeyedINRLinear"](
            base_fc,
            self.key_encoder,
            args.key_dim,
            args.key_hidden,
            args.key_strength,
        )

    def set_coordinates(self, coords: torch.Tensor) -> None:
        self.key_encoder.set_coordinates(coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.fc(h.flatten(start_dim=1))

    def forward_without_inr(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.fc.forward_without_inr(h.flatten(start_dim=1))


def load_activation_film_defs() -> Dict:
    return load_attack_defs()


def make_coords(defs: Dict, seed: int, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return defs["make_key_inr_coordinates"](
        seed,
        args.key_coord_mode,
        args.key_coord_points,
        args.key_coord_dim,
        device,
        args.key_coord_constant,
    )


def make_inr_model(defs: Dict, outputs: int, args: argparse.Namespace, device: torch.device) -> nn.Module:
    coords = make_coords(defs, args.key_coord_seed, args, device)
    return ActivationFilmLeakyLeNetST(outputs, args, coords, defs).to(device)


def write_metrics(out_dir: Path, train_mse: float, test_mse: float, results: Dict, release_stats: Dict, args: argparse.Namespace) -> None:
    fieldnames = [
        "case",
        "train_mse",
        "test_mse",
        "image_mse",
        "final_grad_loss",
        "param_scope",
        "restarts",
        "noise_multiplier",
        "original_grad_norm",
        "clip_coef",
        "noise_norm",
    ]
    with open(out_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, item in results.items():
            stats = release_stats[name]
            writer.writerow({
                "case": name,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "image_mse": item["image_mse"],
                "final_grad_loss": item["final_grad_loss"],
                "param_scope": args.param_scope,
                "restarts": args.restarts,
                "noise_multiplier": stats["noise_multiplier"],
                "original_grad_norm": stats["original_grad_norm"],
                "clip_coef": stats["clip_coef"],
                "noise_norm": stats["noise_norm"],
            })

    with open(out_dir / "metrics.md", "w") as handle:
        handle.write("| case | train_mse | test_mse | image_mse | final_grad_loss | param_scope | restarts | noise_multiplier | original_grad_norm | clip_coef | noise_norm |\n")
        handle.write("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for name, item in results.items():
            stats = release_stats[name]
            handle.write(
                f"| {name} | {train_mse} | {test_mse} | {item['image_mse']} | {item['final_grad_loss']} | "
                f"{args.param_scope} | {args.restarts} | {stats['noise_multiplier']} | {stats['original_grad_norm']} | "
                f"{stats['clip_coef']} | {stats['noise_norm']} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activation-FiLM FL-INR spatial transcriptomics GIA cases.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model", default="leaky_lenet", choices=["leaky_lenet"])
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--key-hidden", type=int, default=8)
    parser.add_argument("--key-strength", type=float, default=1.5)
    parser.add_argument("--key-coord-dim", type=int, default=2)
    parser.add_argument("--key-coord-points", type=int, default=128)
    parser.add_argument("--key-coord-seed", type=int, default=20260508)
    parser.add_argument("--wrong-key-coord-seed", type=int, default=20260509)
    parser.add_argument("--key-coord-mode", choices=["uniform", "normal", "ones", "zeros", "constant"], default="uniform")
    parser.add_argument("--key-coord-constant", type=float, default=1.0)
    parser.add_argument("--fixed-test-patients", default=demo.FIXED_TEST_PATIENTS)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--target-split", choices=["train", "test"], default="train")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--attack-lr", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--l2-weight", type=float, default=1e-6)
    parser.add_argument("--grad-loss", choices=["l2", "cosine", "l2+cosine"], default="l2+cosine")
    parser.add_argument("--param-scope", choices=["all", "first_linear", "last_linear"], default="all")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--save-intermediate-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    demo.seed_everything(args.seed)
    device = demo.resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / "run_args.json", "w") as handle:
        payload = vars(args).copy()
        payload["inr_defs_source"] = "train.py"
        json.dump(payload, handle, indent=2)

    train_loader, test_loader, train_set, test_set, model_info, _ = demo.make_dataloaders(args, device)
    st_mod, outputs = model_info
    normalize = train_loader.dataset.dataset.transform.transforms[-1] if isinstance(train_loader.dataset, Subset) else train_loader.dataset.transform.transforms[-1]
    image_mean = normalize.mean
    image_std = normalize.std

    print(f"device={device}")
    print(f"train samples={len(train_set)} test samples={len(test_set)} outputs={outputs}")
    defs = load_activation_film_defs()
    base_model = demo.make_model(st_mod, outputs, device, args.model, args)
    base_state = copy.deepcopy(base_model.state_dict())
    model = make_inr_model(defs, outputs, args, device)
    demo.load_base_initialization(model, base_state, keyed=True)

    histories = {
        "fl_inr": demo.train_model(
            "fl_inr",
            model,
            train_loader,
            device,
            args.epochs,
            args.lr,
            "normal",
            args.dp_clip_norm,
            0.0,
            args.seed + 4000,
            args.seed + 5000,
        )
    }
    demo.save_training_curve(histories, out_dir / "training_curve.png")

    train_mse = demo.evaluate_model(model, train_loader, device)
    test_mse = demo.evaluate_model(model, test_loader, device)
    print(f"[fl_inr] train_mse={train_mse:.6f} test_mse={test_mse:.6f}")

    target_dataset = train_set if args.target_split == "train" else test_set
    target_x, target_gene, real_index = demo.get_target_sample(target_dataset, args.target_index, device)
    print(f"target_split={args.target_split} target_index={real_index}")

    attack_cases = [
        ("fl_inr_correct", "normal"),
        ("fl_inr_wrong", "normal"),
        ("fl_inr_without", "without_inr"),
    ]
    target_grad_sets = {}
    release_stats = {}
    for case_name, _ in attack_cases:
        seed = args.wrong_key_coord_seed if case_name == "fl_inr_wrong" else args.key_coord_seed
        demo.set_coordinates_any(model, make_coords(defs, seed, args, device))
        clean_grads = demo.compute_target_gradients(model, target_x, target_gene, args.param_scope, "normal")
        if case_name == "fl_inr_wrong":
            demo.set_coordinates_any(model, make_coords(defs, args.key_coord_seed, args, device))
        target_grad_sets[case_name] = clean_grads
        release_stats[case_name] = {
            "noise_multiplier": 0.0,
            "original_grad_norm": float(demo.grad_global_l2_norm(clean_grads).item()),
            "clip_coef": 1.0,
            "noise_norm": 0.0,
        }
        print(f"[{case_name} released gradients] {json.dumps(release_stats[case_name], indent=2)}")

    results = {}
    for case_name, attack_variant in attack_cases:
        seed = args.wrong_key_coord_seed if case_name == "fl_inr_wrong" else args.key_coord_seed
        demo.set_coordinates_any(model, make_coords(defs, seed, args, device))
        print(f"[attack] {case_name}")
        recon, history, snapshots = demo.gradient_inversion_attack(
            model,
            target_grad_sets[case_name],
            target_gene,
            tuple(target_x.shape),
            image_mean,
            image_std,
            args.iters,
            args.attack_lr,
            args.tv_weight,
            args.l2_weight,
            args.grad_loss,
            args.param_scope,
            attack_variant,
            args.restarts,
            args.seed + 12000,
            device,
            out_dir / "intermediates" / case_name if args.save_intermediate_every > 0 else None,
            args.save_intermediate_every,
        )
        mse = demo.image_mse(recon, target_x, image_mean, image_std)
        results[case_name] = {"recon": recon, "history": history, "snapshots": snapshots, "image_mse": mse, "final_grad_loss": history[-1]["grad_loss"]}
        plt.imsave(out_dir / f"{case_name}_recon.png", demo.tensor_to_image(recon, image_mean, image_std))

    demo.save_reconstruction_compare(target_x, results, image_mean, image_std, out_dir / "reconstruction_compare.png")
    demo.save_progress_grid(target_x, results, image_mean, image_std, out_dir / "attack_progress_grid.png")
    write_metrics(out_dir, train_mse, test_mse, results, release_stats, args)
    print(f"outputs written to {out_dir}")


if __name__ == "__main__":
    main()

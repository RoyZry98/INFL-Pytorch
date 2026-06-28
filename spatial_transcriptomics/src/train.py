from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


FIXED_TEST_PATIENTS = "BT23269,BT23277,BT23288,BT23377,BT23901,BT23944"
SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent


def script_path(name: str) -> str:
    return str(SCRIPT_DIR / name)


def task_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else TASK_ROOT / resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_st_module(args: argparse.Namespace, device: torch.device) -> Dict:
    mod = runpy.run_path(script_path("simulation_core.py"))
    setattr(sys.modules["__main__"], "IdentityDict", mod["IdentityDict"])
    st_args = SimpleNamespace(
        fixed_test_patients=args.fixed_test_patients,
        test=0.25,
        window=224,
        gene_filter="tumor",
        downsample=1,
        norm=None,
        gene_transform="log",
        model=args.backbone,
        batch=args.batch_size,
        workers=args.num_workers,
        gpu=device.type == "cuda",
        task="gene",
        load=None,
        gene_mask=None,
        average=None,
    )
    globals_ref = mod["getSTDataset"].__globals__
    globals_ref["args"] = st_args
    globals_ref["device"] = device
    globals_ref["logger"] = logging.getLogger("spatial_transcriptomics_train")
    return mod


def load_attack_defs() -> Dict:
    defs = runpy.run_path(script_path("attack.py"))
    st_defs = runpy.run_path(script_path("simulation_core.py"))
    for name in [
        "make_key_inr_coordinates",
        "CoordinateKeyEncoder",
        "KeyControlledLinear",
        "ActivationDependentFiLM1d",
        "KeyedINRLinear",
    ]:
        defs[name] = st_defs[name]
    return defs


def patient_section_from_patch(path: str) -> Tuple[str, str]:
    patch = Path(path)
    return patch.parent.name, patch.name.split("_")[0]


def make_eval_transform_from_train(st_mod: Dict, transform):
    transforms = st_mod["torchvision"].transforms
    normalize = None
    for item in getattr(transform, "transforms", []):
        if item.__class__.__name__ == "Normalize":
            normalize = item
    if normalize is None:
        return transforms.ToTensor()
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize.mean, std=normalize.std),
    ])


def split_validation_from_train(st_mod: Dict, train_datasets, fulltrain_dataset, args: argparse.Namespace):
    if args.val_fraction <= 0:
        return None
    all_files = list(fulltrain_dataset.dataset)
    groups = sorted({patient_section_from_patch(path) for path in all_files})
    if len(groups) < 2:
        return None
    rng = random.Random(args.seed + args.val_seed_offset)
    val_count = max(1, int(round(len(groups) * args.val_fraction)))
    val_count = min(val_count, len(groups) - 1)
    val_groups = set(rng.sample(groups, val_count))
    val_files = {path for path in all_files if patient_section_from_patch(path) in val_groups}
    if not val_files:
        return None

    val_dataset = copy.copy(fulltrain_dataset)
    val_dataset.dataset = sorted(val_files)
    val_dataset.load_image = True
    val_dataset.transform = make_eval_transform_from_train(st_mod, fulltrain_dataset.transform)

    for dataset in train_datasets:
        dataset.dataset = [path for path in dataset.dataset if path not in val_files]
    fulltrain_dataset.dataset = [path for path in fulltrain_dataset.dataset if path not in val_files]
    train_datasets[:] = [dataset for dataset in train_datasets if len(dataset) > 0]
    logging.info("Validation split: %d section groups, %d spots", len(val_groups), len(val_dataset))
    return val_dataset


def make_dataloaders(args: argparse.Namespace, device: torch.device):
    st_mod = load_st_module(args, device)
    train_datasets, test_dataset, fulltrain_dataset = st_mod["getSTDataset"](args.clients)
    val_dataset = split_validation_from_train(st_mod, train_datasets, fulltrain_dataset, args)
    train_loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for dataset in train_datasets
    ]
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    fulltrain_loader = DataLoader(
        fulltrain_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    outputs = int(fulltrain_dataset[0][2].shape[0])
    return st_mod, train_datasets, train_loaders, val_dataset, val_loader, test_dataset, test_loader, fulltrain_dataset, fulltrain_loader, outputs


def compute_mean_expression(fulltrain_dataset, fulltrain_loader: DataLoader, device: torch.device):
    mean_expression = torch.zeros(fulltrain_dataset[0][2].shape)
    mean_expression_tumor = torch.zeros_like(mean_expression)
    mean_expression_normal = torch.zeros_like(mean_expression)
    tumor = 0.0
    normal = 0.0
    load_image = getattr(fulltrain_loader.dataset, "load_image", True)
    fulltrain_loader.dataset.load_image = False
    for _, y, gene, *_ in tqdm(fulltrain_loader, desc="mean expression"):
        mean_expression += torch.sum(gene, 0)
        mean_expression_tumor += torch.sum(y.float() * gene, 0)
        mean_expression_normal += torch.sum((1 - y).float() * gene, 0)
        tumor += float(torch.sum(y).item())
        normal += float(torch.sum(1 - y).item())
    fulltrain_loader.dataset.load_image = load_image
    mean_expression /= max(tumor + normal, 1.0)
    mean_expression_tumor /= max(tumor, 1.0)
    mean_expression_normal /= max(normal, 1.0)
    return mean_expression.to(device), mean_expression_tumor.to(device), mean_expression_normal.to(device)


def make_keyed_coords(defs: Dict, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return defs["make_key_inr_coordinates"](
        args.key_coord_seed,
        args.key_coord_mode,
        args.key_coord_points,
        args.key_coord_dim,
        device,
        args.key_coord_constant,
    )


def make_torchvision_model(st_mod: Dict, args: argparse.Namespace):
    models = st_mod["torchvision"].models.__dict__
    if args.backbone not in models:
        raise ValueError(f"Unknown torchvision/ST-Net backbone: {args.backbone}")
    try:
        return models[args.backbone](pretrained=bool(args.pretrained))
    except TypeError:
        weights = "DEFAULT" if args.pretrained else None
        return models[args.backbone](weights=weights)


def set_last_linear_to_mean(model: nn.Module, mean_expression: torch.Tensor) -> None:
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        last = model.classifier
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        last = model.fc
    else:
        raise NotImplementedError("Only final classifier/fc Linear replacement is supported for ST-Net backbones")
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(mean_expression)


def replace_final_linear_with_keyed_inr(defs: Dict, model: nn.Module, args: argparse.Namespace, device: torch.device) -> nn.Module:
    coords = make_keyed_coords(defs, args, device)
    key_encoder = defs["CoordinateKeyEncoder"](coords.shape[1], coords.shape[0], args.key_dim, coords).to(device)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        source = model.classifier
        model.key_encoder = key_encoder
        model.classifier = defs["KeyedINRLinear"](source, model.key_encoder, args.key_dim, args.key_hidden, args.key_strength, args.key_init_std)
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        source = model.fc
        model.key_encoder = key_encoder
        model.fc = defs["KeyedINRLinear"](source, model.key_encoder, args.key_dim, args.key_hidden, args.key_strength, args.key_init_std)
    else:
        raise NotImplementedError("Only final classifier/fc Linear replacement is supported for keyed-INR")
    return model.to(device)


def apply_freeze_backbone(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not args.freeze_backbone or args.backbone in {"lenet", "leaky_lenet"}:
        return model
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    if hasattr(model, "key_encoder"):
        for param in model.key_encoder.parameters():
            param.requires_grad = True
    return model


def make_model(defs: Dict, st_mod: Dict, outputs: int, mean_expression: torch.Tensor, args: argparse.Namespace, device: torch.device, keyed: bool = False) -> nn.Module:
    if args.backbone in {"lenet", "leaky_lenet"}:
        if keyed:
            coords = make_keyed_coords(defs, args, device)
            model = defs["KeyedLeNet"](
                coords,
                key_dim=args.key_dim,
                key_hidden=args.key_hidden,
                key_strength=args.key_strength,
                init_std=args.key_init_std,
                num_classes=outputs,
            ).to(device)
            last = model.fc[0].linear
            with torch.no_grad():
                last.weight.zero_()
                last.bias.copy_(mean_expression)
            return model
        model = defs["LeNet"](num_classes=outputs).to(device)
        with torch.no_grad():
            model.fc[0].weight.zero_()
            model.fc[0].bias.copy_(mean_expression)
        return model

    model = make_torchvision_model(st_mod, args)
    st_mod["set_out_features"](model, outputs)
    set_last_linear_to_mean(model, mean_expression)
    model = model.to(device)
    if keyed:
        model = replace_final_linear_with_keyed_inr(defs, model, args, device)
    return apply_freeze_backbone(model, args)


def copy_base_to_keyed(base_state: Dict[str, torch.Tensor], keyed_model: nn.Module) -> None:
    state = keyed_model.state_dict()
    for dst, src in {
        "body.0.weight": "body.0.weight",
        "body.0.bias": "body.0.bias",
        "fc.0.linear.weight": "fc.0.weight",
        "fc.0.linear.bias": "fc.0.bias",
        "classifier.linear.weight": "classifier.weight",
        "classifier.linear.bias": "classifier.bias",
        "fc.linear.weight": "fc.weight",
        "fc.linear.bias": "fc.bias",
    }.items():
        if dst in state and src in base_state and state[dst].shape == base_state[src].shape:
            state[dst].copy_(base_state[src])
    for name, value in base_state.items():
        if name in state and state[name].shape == value.shape:
            state[name].copy_(value)
    keyed_model.load_state_dict(state, strict=True)


def clone_model(defs: Dict, model: nn.Module, outputs: int, args: argparse.Namespace, device: torch.device, keyed: bool) -> nn.Module:
    return copy.deepcopy(model).to(device)


def old_make_model(defs: Dict, outputs: int, mean_expression: torch.Tensor, args: argparse.Namespace, device: torch.device, keyed: bool = False) -> nn.Module:
    if keyed:
        coords = make_keyed_coords(defs, args, device)
        model = defs["KeyedLeNet"](
            coords,
            key_dim=args.key_dim,
            key_hidden=args.key_hidden,
            key_strength=args.key_strength,
            init_std=args.key_init_std,
            num_classes=outputs,
        ).to(device)
        last = model.fc[0].linear
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(mean_expression)
        return model
    model = defs["LeNet"](num_classes=outputs).to(device)
    with torch.no_grad():
        model.fc[0].weight.zero_()
        model.fc[0].bias.copy_(mean_expression)
    return model


def compute_grad_update(old_model: nn.Module, new_model: nn.Module, lr: float) -> List[torch.Tensor]:
    return [(new.data - old.data) / (-lr) for old, new in zip(old_model.parameters(), new_model.parameters())]


def add_update_to_model(model: nn.Module, update: Iterable[torch.Tensor], lr: float) -> None:
    for param, grad in zip(model.parameters(), update):
        if param.requires_grad:
            param.data += -lr * grad.data


def analytic_sigma(st_mod: Dict, clip_norm: float, epsilon: float, delta: float) -> float:
    return float(st_mod["calibrateAnalyticGaussianMechanism"](epsilon=epsilon, delta=delta, GS=clip_norm))


def update_l2_norm(update: List[torch.Tensor]) -> torch.Tensor:
    total = None
    for grad in update:
        value = torch.sum(grad.detach() ** 2)
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return torch.sqrt(total)


def clip_update(update: List[torch.Tensor], clip_norm: float) -> Tuple[List[torch.Tensor], float, float]:
    norm = update_l2_norm(update)
    norm_value = float(norm.detach().cpu().item())
    coef = min(1.0, float(clip_norm) / (norm_value + 1e-12))
    if coef < 1.0:
        update = [grad * coef for grad in update]
    return update, norm_value, coef


def add_dp_noise(update: List[torch.Tensor], sigma: float, device: torch.device, trainable_mask: List[bool] = None) -> List[torch.Tensor]:
    noisy = []
    for idx, grad in enumerate(update):
        if trainable_mask is not None and not trainable_mask[idx]:
            noisy.append(grad)
            continue
        noise = torch.tensor(np.random.normal(0, sigma, grad.shape), dtype=grad.dtype, device=device)
        noisy.append(grad + noise)
    return noisy


def local_train(model: nn.Module, loader: DataLoader, outputs: int, device: torch.device, lr: float, args: argparse.Namespace) -> float:
    model.train()
    if args.freeze_backbone and hasattr(model, "features"):
        model.features.eval()
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    total = 0.0
    n = 0
    for X, _, gene, *_ in tqdm(loader, leave=False):
        X = X.to(device, non_blocking=True)
        gene = gene.to(device, non_blocking=True)
        pred = model(X)
        loss = torch.sum((pred - gene) ** 2) / outputs
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu().item())
        n += X.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def evaluate_mse(model: nn.Module, loader: DataLoader, outputs: int, device: torch.device, desc: str = "eval") -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total = 0.0
    n = 0
    for X, _, gene, *_ in tqdm(loader, desc=desc, leave=False):
        X = X.to(device, non_blocking=True)
        gene = gene.to(device, non_blocking=True)
        pred = model(X)
        loss = torch.sum((pred - gene) ** 2) / outputs
        total += float(loss.detach().cpu().item())
        n += X.shape[0]
    return total / max(n, 1)


def scheduled_lr(epoch: int, args: argparse.Namespace) -> float:
    milestones = []
    for item in str(args.lr_milestones).split(","):
        item = item.strip()
        if item:
            milestones.append(int(item))
    factor = args.lr_gamma ** sum(epoch > milestone for milestone in milestones)
    return args.lr * factor


def method_epsilon(method: str, args: argparse.Namespace) -> float:
    if method == "dp":
        return args.dp_epsilon
    if method == "ppml":
        return args.ppml_epsilon
    return args.epsilon


def method_sigma(method: str, st_mod: Dict, args: argparse.Namespace) -> float:
    if method == "dp" and args.dp_noise is not None:
        return float(args.dp_noise)
    if method == "ppml" and args.ppml_noise is not None:
        return float(args.ppml_noise)
    if method in {"dp", "ppml"}:
        return analytic_sigma(st_mod, args.l2_clip, method_epsilon(method, args), args.delta)
    return 0.0


@torch.no_grad()
def evaluate_and_save(
    model: nn.Module,
    loader: DataLoader,
    test_dataset,
    outputs: int,
    device: torch.device,
    epoch: int,
    pred_root: Path,
    mean_expression: torch.Tensor,
    mean_expression_tumor: torch.Tensor,
    mean_expression_normal: torch.Tensor,
) -> float:
    model.eval()
    predictions, counts, tumor, coord, pixel = [], [], [], [], []
    patient, section = [], []
    total = 0.0
    n = 0
    for X, y, gene, c, _, pat, sec, pix, _ in tqdm(loader, desc=f"test {epoch}", leave=False):
        X = X.to(device, non_blocking=True)
        gene_device = gene.to(device, non_blocking=True)
        pred = model(X)
        loss = torch.sum((pred - gene_device) ** 2) / outputs
        total += float(loss.detach().cpu().item())
        n += X.shape[0]
        predictions.append(pred.detach().cpu().numpy())
        counts.append(gene.numpy())
        tumor.append(y.numpy())
        coord.append(c.numpy())
        pixel.append(pix.numpy())
        patient += list(pat)
        section += list(sec)
    pred_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_root / str(epoch),
        task="gene",
        tumor=np.concatenate(tumor),
        counts=np.concatenate(counts),
        predictions=np.concatenate(predictions),
        coord=np.concatenate(coord),
        patient=np.asarray(patient),
        section=np.asarray(section),
        pixel=np.concatenate(pixel),
        mean_expression=mean_expression.detach().cpu().numpy(),
        mean_expression_tumor=mean_expression_tumor.detach().cpu().numpy(),
        mean_expression_normal=mean_expression_normal.detach().cpu().numpy(),
        ensg_names=test_dataset.ensg_names,
        gene_names=test_dataset.gene_names,
    )
    return total / max(n, 1)


def train_method(
    method: str,
    defs: Dict,
    st_mod: Dict,
    train_loaders: List[DataLoader],
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_dataset,
    outputs: int,
    mean_expression: torch.Tensor,
    mean_expression_tumor: torch.Tensor,
    mean_expression_normal: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, float]:
    keyed = method == "inr"
    server = make_model(defs, st_mod, outputs, mean_expression, args, device, keyed=False)
    if keyed:
        base_state = copy.deepcopy(server.state_dict())
        server = make_model(defs, st_mod, outputs, mean_expression, args, device, keyed=True)
        copy_base_to_keyed(base_state, server)
    best_val_mse = float("inf")
    best_test_mse = float("inf")
    best_epoch = 0
    epsilon = method_epsilon(method, args)
    sigma = method_sigma(method, st_mod, args)
    pred_root = out_dir / method / "pred_root"
    ckpt_dir = out_dir / method / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainable_mask = [param.requires_grad for param in server.parameters()]
    history = []
    for epoch in range(1, args.epochs + 1):
        lr = scheduled_lr(epoch, args)
        logging.info("[%s] epoch %s/%s lr=%g", method, epoch, args.epochs, lr)
        loader_order = list(train_loaders)
        if method == "ppml":
            random.shuffle(loader_order)
        aggregated = None
        train_loss = 0.0
        update_norms = []
        clip_coefs = []
        for loader in loader_order:
            client = clone_model(defs, server, outputs, args, device, keyed)
            train_loss += local_train(client, loader, outputs, device, lr, args)
            update = compute_grad_update(server, client, lr)
            if method in {"dp", "ppml"}:
                update, norm_value, clip_coef = clip_update(update, args.l2_clip)
                update_norms.append(norm_value)
                clip_coefs.append(clip_coef)
            if aggregated is None:
                aggregated = [item.clone() for item in update]
            else:
                for acc, item in zip(aggregated, update):
                    acc.add_(item)
            del client
            if device.type == "cuda":
                torch.cuda.empty_cache()
        aggregated = [item / len(loader_order) for item in aggregated]
        if method in {"dp", "ppml"}:
            aggregated = add_dp_noise(aggregated, sigma, device, trainable_mask=trainable_mask)
        add_update_to_model(server, aggregated, lr)
        test_mse = evaluate_and_save(server, test_loader, test_dataset, outputs, device, epoch, pred_root, mean_expression, mean_expression_tumor, mean_expression_normal)
        val_mse = evaluate_mse(server, val_loader, outputs, device, desc=f"val {method} {epoch}") if val_loader is not None else test_mse
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss / len(loader_order),
            "val_mse": val_mse,
            "test_mse": test_mse,
            "sigma": sigma,
            "epsilon": epsilon,
            "mean_update_norm": float(np.mean(update_norms)) if update_norms else 0.0,
            "mean_clip_coef": float(np.mean(clip_coefs)) if clip_coefs else 1.0,
        }
        history.append(row)
        logging.info("[%s] epoch=%d lr=%g train_loss=%.6f val_mse=%.6f test_mse=%.6f sigma=%.6f", method, epoch, lr, row["train_loss"], val_mse, test_mse, sigma)
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save({"model": server.state_dict(), "epoch": epoch, "history": history}, ckpt_dir / f"{epoch}.pt")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_test_mse = test_mse
            best_epoch = epoch
            torch.save({"model": server.state_dict(), "epoch": epoch, "history": history}, out_dir / method / "modelbest.pt")
            evaluate_and_save(server, test_loader, test_dataset, outputs, device, "best", pred_root, mean_expression, mean_expression_tumor, mean_expression_normal)
    with open(out_dir / method / "history.json", "w") as handle:
        json.dump(history, handle, indent=2)
    return {
        "method": method,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "best_test_mse": best_test_mse,
        "final_val_mse": history[-1]["val_mse"],
        "final_test_mse": history[-1]["test_mse"],
        "epsilon": epsilon,
        "sigma": sigma,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated ST training with ST-Net/LeNet backbone and activation-FiLM keyed-INR.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="spatial_fed_e5_c30_fixedsplit")
    parser.add_argument("--methods", default="normal,dp,ppml,inr")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--clients", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-milestones", default="8,15")
    parser.add_argument("--lr-gamma", type=float, default=0.3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--dp-epsilon", type=float, default=2.36)
    parser.add_argument("--ppml-epsilon", type=float, default=1.29)
    parser.add_argument("--dp-noise", type=float, default=None)
    parser.add_argument("--ppml-noise", type=float, default=None)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--l2-clip", type=float, default=1.0)
    parser.add_argument("--backbone", default="densenet121")
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--freeze-backbone", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--val-seed-offset", type=int, default=9173)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--fixed-test-patients", default=FIXED_TEST_PATIENTS)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--key-hidden", type=int, default=8)
    parser.add_argument("--key-strength", type=float, default=1.0)
    parser.add_argument("--key-init-std", type=float, default=0.02)
    parser.add_argument("--key-coord-dim", type=int, default=2)
    parser.add_argument("--key-coord-points", type=int, default=128)
    parser.add_argument("--key-coord-seed", type=int, default=20260508)
    parser.add_argument("--key-coord-mode", default="uniform", choices=["uniform", "normal", "ones", "zeros", "constant"])
    parser.add_argument("--key-coord-constant", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    out_dir = task_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    st_mod, train_datasets, train_loaders, val_dataset, val_loader, test_dataset, test_loader, fulltrain_dataset, fulltrain_loader, outputs = make_dataloaders(args, device)
    mean_expression, mean_expression_tumor, mean_expression_normal = compute_mean_expression(fulltrain_dataset, fulltrain_loader, device)
    defs = load_attack_defs()
    rows = []
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        rows.append(train_method(method, defs, st_mod, train_loaders, val_loader, test_loader, test_dataset, outputs, mean_expression, mean_expression_tumor, mean_expression_normal, args, device, out_dir))
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(rows, handle, indent=2)
    logging.info("outputs written to %s", out_dir)


if __name__ == "__main__":
    main()

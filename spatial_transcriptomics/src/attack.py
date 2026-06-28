import argparse
import csv
import logging
import os
import random
import runpy
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent


def script_path(name: str) -> str:
    return str(SCRIPT_DIR / name)


def task_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else TASK_ROOT / resolved


def make_key_inr_coordinates(seed, mode, num_points, coord_dim, device, constant=1.0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if mode == "uniform":
        coords = torch.rand(num_points, coord_dim, generator=generator) * 2.0 - 1.0
    elif mode == "normal":
        coords = torch.randn(num_points, coord_dim, generator=generator)
    elif mode == "ones":
        coords = torch.ones(num_points, coord_dim)
    elif mode == "zeros":
        coords = torch.zeros(num_points, coord_dim)
    elif mode == "constant":
        coords = torch.full((num_points, coord_dim), float(constant))
    else:
        raise ValueError("Unknown key-INR coordinate mode: {}".format(mode))
    return coords.to(device)


class CoordinateKeyEncoder(nn.Module):
    def __init__(self, coord_dim, num_points, key_dim, coords):
        super().__init__()
        if tuple(coords.shape) != (num_points, coord_dim):
            raise ValueError("coords shape mismatch")
        self.register_buffer("coords", coords.detach().clone())
        self.net = nn.Sequential(
            nn.Linear(coord_dim, key_dim),
            nn.SiLU(),
            nn.Linear(key_dim, key_dim),
            nn.SiLU(),
            nn.LayerNorm(key_dim),
        )

    def set_coordinates(self, coords):
        if coords.shape != self.coords.shape:
            raise ValueError("coords shape mismatch")
        with torch.no_grad():
            self.coords.copy_(coords.to(device=self.coords.device, dtype=self.coords.dtype))

    def forward(self):
        return self.net(self.coords).mean(dim=0)


class KeyControlledLinear(nn.Module):
    def __init__(self, in_features, out_features, key_dim, hidden_dim, key_strength=1.0, bias=True, init_std=0.02):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.key_strength = float(key_strength)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.key_to_mod = nn.Sequential(
            nn.Linear(key_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features * 2),
        )
        self.reset_parameters()
        last = self.key_to_mod[-1]
        nn.init.normal_(last.weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(last.bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            bound = 1 / np.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def copy_from_linear(self, linear):
        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if self.bias is not None and linear.bias is not None:
                self.bias.copy_(linear.bias)

    def make_effective_params(self, key_vec):
        scale_raw, bias_raw = self.key_to_mod(key_vec).chunk(2, dim=0)
        scale = 1.0 + self.key_strength * torch.tanh(scale_raw).view(self.out_features, 1)
        weight = self.weight * scale
        bias = None if self.bias is None else self.bias + self.key_strength * torch.tanh(bias_raw)
        return weight, bias

    def forward(self, x, key_vec):
        weight, bias = self.make_effective_params(key_vec)
        return F.linear(x, weight, bias)

    def forward_without_key(self, x):
        return F.linear(x, self.weight, self.bias)


class KeyedINRLinear(nn.Module):
    def __init__(self, source, key_encoder, key_dim, key_hidden, key_strength, init_std):
        super().__init__()
        object.__setattr__(self, "key_encoder", key_encoder)
        self.linear = KeyControlledLinear(
            source.in_features,
            source.out_features,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
            bias=source.bias is not None,
            init_std=init_std,
        )
        self.linear.copy_from_linear(source)

    def forward(self, x):
        return self.linear(x, self.key_encoder())

    def forward_without_inr(self, x):
        return self.linear.forward_without_key(x)


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=150528, num_classes=1):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(hideen, num_classes))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class KeyedLeNet(nn.Module):
    def __init__(self, coords, key_dim=16, key_hidden=8, key_strength=1.0, init_std=0.02, channel=3, hideen=150528, num_classes=1):
        super().__init__()
        base = LeNet(channel=channel, hideen=hideen, num_classes=num_classes)
        self.body = base.body
        self.key_encoder = CoordinateKeyEncoder(coords.shape[1], coords.shape[0], key_dim, coords)
        self.fc = nn.Sequential(
            KeyedINRLinear(base.fc[0], self.key_encoder, key_dim, key_hidden, key_strength, init_std)
        )

    def set_coordinates(self, coords):
        self.key_encoder.set_coordinates(coords)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)

    def forward_without_inr(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc[0].forward_without_inr(out)


def forward_variant(model, x, variant):
    if variant == "without_inr" and hasattr(model, "forward_without_inr"):
        return model.forward_without_inr(x)
    return model(x)


def clone_model(model, device):
    if isinstance(model, KeyedLeNet):
        coords = model.key_encoder.coords.detach().clone().to(device)
        cloned = KeyedLeNet(
            coords,
            key_dim=model.key_encoder.net[0].out_features,
            key_hidden=model.fc[0].linear.key_to_mod[0].out_features,
            key_strength=model.fc[0].linear.key_strength,
            init_std=0.02,
            num_classes=model.fc[0].linear.out_features,
        ).to(device)
    else:
        cloned = LeNet(num_classes=model.fc[0].out_features).to(device)
    cloned.load_state_dict(model.state_dict())
    return cloned


def tensor_to_image(x):
    return np.transpose(x.detach().cpu().numpy().squeeze(), (1, 2, 0))


def parameter_gradients(loss, parameters, create_graph):
    params = list(parameters)
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )
    return [torch.zeros_like(param) if grad is None else grad for param, grad in zip(params, grads)]


def run_attack_for_sample(target_model, attack_model, gt_data, gene, outputs, mode, dp_fn, device, epsilon, delta, lr, iterations, save_path, sample_idx, method_name, attack_variant):
    pred = target_model(gt_data)
    y = torch.sum((pred - gene) ** 2) / outputs
    dy_dx = parameter_gradients(y, target_model.parameters(), create_graph=False)
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    if mode == "DP":
        original_dy_dx = [dp_fn(grad, device=device, epsilon=epsilon, delta=delta) for grad in original_dy_dx]

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([dummy_data], lr=lr)
    history = []
    history_iters = []
    rows = []
    print("lr =", lr)
    for iters in range(iterations):
        print("idx: {}, {}/{}".format(sample_idx, iters, iterations))

        def closure():
            optimizer.zero_grad()
            pred_dummy = forward_variant(attack_model, dummy_data, attack_variant)
            dummy_loss = torch.sum((pred_dummy - gene) ** 2) / outputs
            dummy_dy_dx = parameter_gradients(dummy_loss, attack_model.parameters(), create_graph=True)
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        current_mse = torch.mean((dummy_data - gt_data) ** 2).item()
        rows.append({"iter": iters, "loss": current_loss, "mse": current_mse})
        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
        print(current_time, iters, "loss = %.8f, mse = %.8f" % (current_loss, current_mse))

        if iters % 100 == 0 and iters >= 100:
            history.append(tensor_to_image(dummy_data))
            history_iters.append(iters)
            plt.figure(figsize=(12, 8), dpi=300)
            plt.subplot(3, 10, 1)
            plt.imshow(tensor_to_image(gt_data))
            plt.title("original")
            plt.axis("off")
            for j in range(min(len(history), 29)):
                plt.subplot(3, 10, j + 2)
                plt.imshow(history[j])
                plt.title("iter=%d" % (history_iters[j]))
                plt.axis("off")
            plt.savefig("%s/iDLG_%s_on_%s_%05d.png" % (save_path, method_name, sample_idx, iters))
            plt.close()
            if current_loss < 0.0000001:
                break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Official-style LeNet iDLG with keyed INR")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=100)
    parser.add_argument("--delta", type=float, default=10e-5)
    parser.add_argument("--mode", default="SGD")
    parser.add_argument("--client", type=int, default=3)
    parser.add_argument("--nprocess", type=int, default=100)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--shuffle_model", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=901)
    parser.add_argument("--samples", default="0,50,100")
    parser.add_argument("--methods", default="baseline,correct,wrong,without", help="baseline,correct,wrong,without")
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--key-hidden", type=int, default=8)
    parser.add_argument("--key-strength", type=float, default=1.0)
    parser.add_argument("--key-init-std", type=float, default=0.02)
    parser.add_argument("--key-seed", type=int, default=20260508)
    parser.add_argument("--wrong-key-seed", type=int, default=20260509)
    parser.add_argument("--key-points", type=int, default=128)
    parser.add_argument("--key-coord-dim", type=int, default=2)
    args_cli = parser.parse_args()

    device_name = args_cli.device
    if "cuda" in device_name:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_name.split(":")[1]
        device_name = "cuda:0"
    device = torch.device(device_name)

    mod = runpy.run_path(script_path("attack_core.py"))
    setattr(sys.modules["__main__"], "IdentityDict", mod["IdentityDict"])
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    globals_ref = mod["getSTDataset"].__globals__
    dataset_args = SimpleNamespace(
        window_raw=None,
        test=0.25,
        window=224,
        gene_filter="tumor",
        downsample=1,
        gene_transform="log",
        model="vgg11",
        norm=None,
        batch=args_cli.batch_size,
        workers=args_cli.nprocess,
        gpu=torch.cuda.is_available(),
        pretrained=True,
        task="gene",
        load=None,
        gene_mask=None,
        epochs=50,
        trainpatients=None,
        testpatients=None,
        restart=None,
        save_pred_every=None,
        average=None,
        checkpoint_every=10,
        keep_checkpoints=None,
    )
    globals_ref["args"] = dataset_args
    globals_ref["device"] = device
    globals_ref["logger"] = logging.getLogger("spatial_transcriptomics_attack")

    save_path = task_path(args_cli.expname)
    save_path.mkdir(parents=True, exist_ok=True)
    print("root_path:", save_path)
    print("save_path:", save_path)
    logging.info("Mode: %s", args_cli.mode)
    logging.info("device: %s", device_name)
    logging.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES", "not defined"))
    logging.info("CPUs: %s", os.sched_getaffinity(0))
    logging.info("GPUs: %s", torch.cuda.device_count())
    logging.info("Hostname: %s", socket.gethostname())

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    train_datasets, test_dataset, fulltrain_dataset = mod["getSTDataset"](args_cli.client)
    loader = DataLoader(test_dataset, args_cli.batch_size, False)
    globals_ref["trainDatasets"] = train_datasets
    globals_ref["testDataset"] = test_dataset
    globals_ref["fulltrainDataset"] = fulltrain_dataset
    globals_ref["testLoader"] = loader
    globals_ref["fulltrainLoader"] = DataLoader(fulltrain_dataset, args_cli.batch_size, False)
    mod["compute_mean_expression"]()

    outputs = fulltrain_dataset[0][2].shape[0]
    selected_samples = {int(item) for item in args_cli.samples.split(",") if item.strip()}
    selected_methods = {item.strip() for item in args_cli.methods.split(",") if item.strip()}
    correct_coords = make_key_inr_coordinates(args_cli.key_seed, "uniform", args_cli.key_points, args_cli.key_coord_dim, device)
    wrong_coords = make_key_inr_coordinates(args_cli.wrong_key_seed, "uniform", args_cli.key_points, args_cli.key_coord_dim, device)
    methods = {
        "baseline": {"target_keyed": False, "attack_variant": "normal", "wrong_key": False},
        "correct": {"target_keyed": True, "attack_variant": "normal", "wrong_key": False},
        "wrong": {"target_keyed": True, "attack_variant": "normal", "wrong_key": True},
        "without": {"target_keyed": True, "attack_variant": "without_inr", "wrong_key": False},
    }

    metrics = []
    for sample_idx, batch in enumerate(tqdm(loader)):
        if sample_idx not in selected_samples:
            continue
        X, y, gene, c, ind, pat, s, pix, f = batch
        print("iDLG on idx: {}".format(sample_idx))
        gt_data = X.to(device)
        gene = gene.to(device)
        print(gt_data.shape, gene.shape)
        baseline_target = LeNet(num_classes=outputs).to(device)
        keyed_target = KeyedLeNet(
            correct_coords,
            key_dim=args_cli.key_dim,
            key_hidden=args_cli.key_hidden,
            key_strength=args_cli.key_strength,
            init_std=args_cli.key_init_std,
            num_classes=outputs,
        ).to(device)
        for method_name in ["baseline", "correct", "wrong", "without"]:
            if method_name not in selected_methods:
                continue
            config = methods[method_name]
            if config["target_keyed"]:
                target_model = keyed_target
            else:
                target_model = baseline_target
            attack_model = clone_model(target_model, device)
            if config["wrong_key"]:
                attack_model.set_coordinates(wrong_coords)
            print("running method {} on sample {}".format(method_name, sample_idx))
            torch.manual_seed(100000 + sample_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(100000 + sample_idx)
            rows = run_attack_for_sample(
                target_model,
                attack_model,
                gt_data,
                gene,
                outputs,
                args_cli.mode,
                mod["dp"],
                device,
                args_cli.epsilon,
                args_cli.delta,
                args_cli.lr,
                args_cli.iterations,
                str(save_path),
                sample_idx,
                method_name,
                config["attack_variant"],
            )
            if rows:
                final = rows[-1]
                metrics.append({"sample": sample_idx, "method": method_name, **final})
                with open(save_path / "iDLG_{}_on_{}_history.csv".format(method_name, sample_idx), "w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
        if selected_samples and sample_idx >= max(selected_samples):
            break

    if metrics:
        with open(save_path / "metrics.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
        with open(save_path / "metrics.md", "w") as handle:
            handle.write("| sample | method | iter | loss | mse |\n")
            handle.write("|---:|---|---:|---:|---:|\n")
            for row in metrics:
                handle.write("| {sample} | {method} | {iter} | {loss} | {mse} |\n".format(**row))


if __name__ == "__main__":
    main()

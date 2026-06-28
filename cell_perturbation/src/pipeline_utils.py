# federated_gears_utils.py
import os
import copy
import math
import warnings
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

from src import PertData, GEARS
from src.inference import (evaluate, compute_metrics,
                             deeper_analysis, non_dropout_analysis)

warnings.filterwarnings("ignore")


# =========================
# Config dataclasses
# =========================
@dataclass
class FLArgs:
    gpu: int = 0
    n_client: int = 10
    dataset: str = "adamson"
    alpha: float = 0.3
    global_round: int = 200
    local_epoch: int = 2
    frac: float = 0.5
    lr: float = 1e-3
    inr: bool = False
    data_dir: str = "./data"   # gears 数据根目录
    batch_size: int = 32
    seed: int = 1
    out_dir_prefix: str = "INFL"
    posi_method: str = "norman"
    coord_dim: int = 2
    coord_points: int = 128
    coord_seed: int = 0
    coord_mode: str = "uniform"
    coord_constant: float = 1.0
    key_dim: int = 16
    key_hidden: int = 8
    key_strength: float = 5.0
    disable_film: bool = False


# =========================
# Keyed-INR modules
# =========================
def make_coordinates(
    seed: int,
    mode: str,
    num_points: int,
    coord_dim: int,
    device: torch.device,
    constant: float = 1.0,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    if mode == "uniform":
        coords = torch.rand(num_points, coord_dim, generator=g) * 2.0 - 1.0
    elif mode == "normal":
        coords = torch.randn(num_points, coord_dim, generator=g)
    elif mode == "ones":
        coords = torch.ones(num_points, coord_dim)
    elif mode == "zeros":
        coords = torch.zeros(num_points, coord_dim)
    elif mode == "negones":
        coords = -torch.ones(num_points, coord_dim)
    elif mode == "constant":
        coords = torch.full((num_points, coord_dim), float(constant))
    else:
        raise ValueError(f"Unknown coordinate mode: {mode}")

    return coords.to(device)


class CoordinateKeyEncoder(nn.Module):
    def __init__(
        self,
        coord_dim: int,
        num_points: int,
        key_dim: int,
        coords: torch.Tensor,
    ) -> None:
        super().__init__()
        if coords.shape != (num_points, coord_dim):
            raise ValueError(
                f"coords shape mismatch: expected {(num_points, coord_dim)}, "
                f"got {tuple(coords.shape)}"
            )

        self.coord_dim = int(coord_dim)
        self.num_points = int(num_points)
        self.key_dim = int(key_dim)
        self.register_buffer("coords", coords.detach().clone())
        self.net = nn.Sequential(
            nn.Linear(coord_dim, key_dim),
            nn.SiLU(),
            nn.Linear(key_dim, key_dim),
            nn.SiLU(),
            nn.LayerNorm(key_dim),
        )

    def set_coordinates(self, coords: torch.Tensor) -> None:
        if coords.shape != self.coords.shape:
            raise ValueError(
                f"coords shape mismatch: expected {tuple(self.coords.shape)}, "
                f"got {tuple(coords.shape)}"
            )
        with torch.no_grad():
            self.coords.copy_(coords.to(device=self.coords.device, dtype=self.coords.dtype))

    def forward(self) -> torch.Tensor:
        return self.net(self.coords).mean(dim=0)


class KeyControlledLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        key_dim: int,
        hidden_dim: int,
        key_strength: float,
        bias: bool = True,
    ) -> None:
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
        self._init_key_mod_near_identity()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            bound = 1 / np.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _init_key_mod_near_identity(self) -> None:
        last = self.key_to_mod[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=0.02)
            nn.init.zeros_(last.bias)

    def make_effective_params(
        self,
        key_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mod = self.key_to_mod(key_vec)
        scale_raw, bias_raw = mod.chunk(2, dim=0)
        scale = 1.0 + self.key_strength * torch.tanh(scale_raw)
        effective_weight = self.weight * scale.view(self.out_features, 1)

        if self.bias is None:
            effective_bias = None
        else:
            bias_delta = self.key_strength * torch.tanh(bias_raw)
            effective_bias = self.bias + bias_delta
        return effective_weight, effective_bias

    def forward(self, x: torch.Tensor, key_vec: torch.Tensor) -> torch.Tensor:
        weight, bias = self.make_effective_params(key_vec)
        return F.linear(x, weight, bias)

    def forward_without_key(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ActivationDependentFiLM1d(nn.Module):
    def __init__(
        self,
        features: int,
        key_dim: int,
        hidden_dim: int,
        key_strength: float,
    ) -> None:
        super().__init__()
        self.features = int(features)
        self.key_strength = float(key_strength)
        self.act_proj = nn.Sequential(
            nn.Linear(features, key_dim),
            nn.SiLU(),
            nn.LayerNorm(key_dim),
        )
        self.film = nn.Sequential(
            nn.Linear(key_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, features * 2),
        )
        self._init_near_identity()

    def _init_near_identity(self) -> None:
        last = self.film[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=0.02)
            nn.init.zeros_(last.bias)

    def forward(self, h: torch.Tensor, key_vec: torch.Tensor) -> torch.Tensor:
        orig_shape = h.shape
        if h.dim() > 2:
            h = h.reshape(-1, orig_shape[-1])

        batch_size = h.shape[0]
        act_embed = self.act_proj(h)
        key_embed = key_vec.unsqueeze(0).expand(batch_size, -1)
        joint = torch.cat([act_embed, key_embed, act_embed * key_embed], dim=1)
        gamma_beta = self.film(joint)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        out = h * (1.0 + self.key_strength * torch.tanh(gamma))
        out = out + self.key_strength * torch.tanh(beta)
        return out.reshape(orig_shape)


class KeyedINRLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        key_encoder: CoordinateKeyEncoder,
        key_dim: int,
        key_hidden: int,
        key_strength: float,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "key_encoder", key_encoder)
        self.linear = KeyControlledLinear(
            source.in_features,
            source.out_features,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
            bias=source.bias is not None,
        )
        self.linear.weight.data.copy_(source.weight.data)
        if source.bias is not None:
            self.linear.bias.data.copy_(source.bias.data)

        self.film = (
            ActivationDependentFiLM1d(
                source.out_features,
                key_dim=key_dim,
                hidden_dim=key_hidden,
                key_strength=key_strength,
            )
            if use_film
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key_vec = self.key_encoder()
        out = self.linear(x, key_vec)
        if self.film is not None:
            out = self.film(out, key_vec)
        return out

    def forward_without_inr(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear.forward_without_key(x)


def _replace_linear_modules(
    module: nn.Module,
    key_encoder: CoordinateKeyEncoder,
    key_dim: int,
    key_hidden: int,
    key_strength: float,
    use_film: bool,
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                KeyedINRLinear(
                    child,
                    key_encoder=key_encoder,
                    key_dim=key_dim,
                    key_hidden=key_hidden,
                    key_strength=key_strength,
                    use_film=use_film,
                ),
            )
            replaced += 1
        else:
            replaced += _replace_linear_modules(
                child,
                key_encoder=key_encoder,
                key_dim=key_dim,
                key_hidden=key_hidden,
                key_strength=key_strength,
                use_film=use_film,
            )
    return replaced


def replace_linear_with_inr(
    module: nn.Module,
    coord_dim: int = 2,
    coord_points: int = 128,
    key_dim: int = 16,
    key_hidden: int = 8,
    key_strength: float = 5.0,
    coord_seed: int = 0,
    coord_mode: str = "uniform",
    coord_constant: float = 1.0,
    device: torch.device = torch.device("cpu"),
    use_film: bool = True,
) -> int:
    device = torch.device(device)
    coords = make_coordinates(
        seed=coord_seed,
        mode=coord_mode,
        num_points=coord_points,
        coord_dim=coord_dim,
        device=device,
        constant=coord_constant,
    )
    key_encoder = CoordinateKeyEncoder(coord_dim, coord_points, key_dim, coords).to(device)
    replaced = _replace_linear_modules(
        module,
        key_encoder=key_encoder,
        key_dim=key_dim,
        key_hidden=key_hidden,
        key_strength=key_strength,
        use_film=use_film,
    )
    module.add_module("keyed_inr_key_encoder", key_encoder)

    def set_coordinates(coords: torch.Tensor) -> None:
        key_encoder.set_coordinates(coords)

    module.set_coordinates = set_coordinates
    module.to(device)
    return replaced


def _is_keyed_inr_param_name(name: str) -> bool:
    return (
        "keyed_inr_key_encoder" in name
        or ".key_to_mod." in name
        or ".film." in name
        or ".act_proj." in name
    )


# =========================
# Partitioning
# =========================
def graphs_partition(dataset, num_users, seed=None):
    if seed is not None:
        np.random.seed(seed)
    num_items = len(dataset) // num_users
    all_idxs = np.arange(len(dataset))
    dict_users = {}
    for uid in range(num_users):
        dict_users[uid] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = np.setdiff1d(all_idxs, list(dict_users[uid]))
    dict_users[num_users - 1].update(all_idxs)
    return dict_users


# =========================
# FedAvg and utils
# =========================
def fed_avg(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    w_avg = copy.deepcopy(state_dicts[0])
    for k, value in w_avg.items():
        if "coords" in k or not torch.is_floating_point(value):
            w_avg[k] = value.clone()
            continue
        tensors = [sd[k].float() for sd in state_dicts]
        w_avg[k] = torch.stack(tensors, dim=0).mean(dim=0).to(dtype=value.dtype)
    return w_avg


def adjust_module_prefix(state_dict, add_prefix=False):
    new_state = {}
    for k, v in state_dict.items():
        if add_prefix:
            if not k.startswith("model."):
                nk = "model." + k
            else:
                nk = k
        else:
            if k.startswith("model."):
                nk = k[len("model."):]
            else:
                nk = k
        new_state[nk] = v
    return new_state


def count_inr_params(model: nn.Module) -> int:
    total = 0
    for name, param in model.named_parameters():
        if _is_keyed_inr_param_name(name):
            total += param.numel()
    return total


def count_backbone_params(model: nn.Module) -> int:
    total = 0
    for n, p in model.named_parameters():
        if not _is_keyed_inr_param_name(n):
            total += p.numel()
    return total


# =========================
# Data and loaders
# =========================
def prepare_gears_data(args: FLArgs):
    pdata = PertData(args.data_dir)
    pdata.load(data_name=args.dataset)
    pdata.prepare_split(split="simulation", seed=args.seed)
    pdata.get_dataloader(batch_size=args.batch_size)

    train_graphs = pdata.dataloader["train_loader"].dataset
    train_labels = [g.pert for g in train_graphs]
    subgroup_names = list(pdata.subgroup["test_subgroup"].keys())
    return pdata, train_graphs, train_labels, subgroup_names


def build_sub_dataloader(train_graphs, idxs: List[int], batch_size: int = 32):
    return DataLoader([train_graphs[i] for i in idxs], batch_size=batch_size, shuffle=True, drop_last=True)


# =========================
# Evaluation helpers
# =========================
def evaluate_and_log(gears_global: GEARS, rnd: int, out_dir: str, subgroup_names: List[str],
                     log_overall: List[float], log_de: List[float], log_subgroup: Dict[str, List[float]],
                     target_groups: List[str]):
    if "test_loader" not in gears_global.dataloader:
        print("    └── Do not find test_loader, skip evaluation.")
        return None, None

    test_loader = gears_global.dataloader["test_loader"]
    gears_global.model.eval()
    with torch.no_grad():
        test_res = evaluate(test_loader, gears_global.model, False, gears_global.device)
    test_metrics, test_pert_res = compute_metrics(test_res)
    print(f'    └── Test  Round {rnd+1} Overall  MSE : {test_metrics["mse"]:.4f}')
    print(f'        Test Round {rnd+1} Top-20-DE MSE : {test_metrics["mse_de"]:.4f}')

    print("Start doing subgroup analysis for simulation split...")
    subgroup = gears_global.pdata.subgroup

    # quick average for mse_de per subgroup
    for name in subgroup_names:
        perts = subgroup["test_subgroup"][name]
        vals = [test_pert_res[p]["mse_de"] for p in perts]
        mean_val = np.mean(vals) if len(vals) > 0 else np.nan
        log_subgroup[name].append(mean_val)
        print(f"        {name:<18}: {mean_val:.4f}")

    # deeper analysis
    out = deeper_analysis(gears_global.pdata.adata, test_res)
    out_non_dropout = non_dropout_analysis(gears_global.pdata.adata, test_res)
    metrics = ["pearson_delta"]
    metrics_non_dropout = [
        "frac_opposite_direction_top20_non_dropout",
        "frac_sigma_below_1_non_dropout",
        "mse_top20_de_non_dropout",
    ]
    subgroup_analysis = {n: {m: [] for m in metrics + metrics_non_dropout} for n in subgroup["test_subgroup"].keys()}
    for name, pert_list in subgroup["test_subgroup"].items():
        for pert in pert_list:
            for m in metrics:
                subgroup_analysis[name][m].append(out[pert][m])
            for m in metrics_non_dropout:
                subgroup_analysis[name][m].append(out_non_dropout[pert][m])
    for name, result in subgroup_analysis.items():
        for m in result.keys():
            subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
            print("test_" + name + "_" + m + ": " + str(subgroup_analysis[name][m]))

    # logs
    log_overall.append(test_metrics["mse"])
    log_de.append(test_metrics["mse_de"])

    # save CSV and plots (overwrite each round)
    cur_rounds = np.arange(1, len(log_overall) + 1)

    df_dict = {
        "round": cur_rounds,
        "overall_mse": log_overall,
        "top20de_mse": log_de,
    }
    for n in subgroup_names:
        df_dict[f"{n}_mse_de"] = log_subgroup[n]
    df = pd.DataFrame(df_dict)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "fed_test_metrics.csv"), index=False)

    plt.figure(figsize=(6, 3.5))
    plt.plot(cur_rounds, log_overall, marker="o", label="Overall MSE")
    plt.plot(cur_rounds, log_de, marker="s", label="Top-20-DE MSE")
    plt.xlabel("Global Round")
    plt.ylabel("MSE")
    plt.title("Test MSE vs. Rounds")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_mse_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    for n in subgroup_names:
        plt.plot(cur_rounds, log_subgroup[n], marker=".", label=n)
    plt.xlabel("Global Round")
    plt.ylabel("Top-20-DE MSE")
    plt.title("Sub-group MSE")
    plt.grid(alpha=.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "subgroup_mse_curve.png"), dpi=300)
    plt.close()

    # compute score over target groups
    score_list = []
    for g in target_groups:
        if len(log_subgroup[g]) > 0 and not np.isnan(log_subgroup[g][-1]):
            score_list.append(log_subgroup[g][-1])
    cur_score = np.mean(score_list) if len(score_list) > 0 else np.inf
    print(f'    >>> Avg MSE_DE ({", ".join(target_groups)}): {cur_score:.4f}')

    return cur_score, test_pert_res


# =========================
# High-level FL pipeline
# =========================
def run_federated_demo(args: FLArgs):
    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # prepare data
    pdata, train_graphs, train_labels, subgroup_names = prepare_gears_data(args)
    print("Size of each split:",
          f"train={len(pdata.dataloader['train_loader'].dataset)}, "
          f"val={len(pdata.dataloader['val_loader'].dataset) if 'val_loader' in pdata.dataloader else 'NA'}, "
          f"test={len(pdata.dataloader['test_loader'].dataset) if 'test_loader' in pdata.dataloader else 'NA'}")

    # partition
    dict_clients = graphs_partition(train_graphs, args.n_client, seed=0)
    print("Number of samples per client:", [len(v) for v in dict_clients.values()])

    # init global model
    gears_global = GEARS(pdata, device=device, weight_bias_track=False)
    gears_global.model_initialize(hidden_size=64)
    gears_global.pdata = pdata

    if args.inr:
        replace_linear_with_inr(
            gears_global.model,
            coord_dim=args.coord_dim,
            coord_points=args.coord_points,
            key_dim=args.key_dim,
            key_hidden=args.key_hidden,
            key_strength=args.key_strength,
            coord_seed=args.coord_seed,
            coord_mode=args.coord_mode,
            coord_constant=args.coord_constant,
            device=device,
            use_film=not args.disable_film,
        )

    global_cfg = copy.deepcopy(gears_global.config)

    inr_params = count_inr_params(gears_global.model)
    backbone_params = count_backbone_params(gears_global.model)
    total_params = sum(p.numel() for p in gears_global.model.parameters())
    ratio = inr_params / backbone_params if backbone_params > 0 else float("nan")
    print(f"Keyed-INR parameters: {inr_params}")
    print(f"Backbone (non-INR) parameters: {backbone_params}")
    print(f"Total parameters: {total_params}")
    print(f"Ratio of INR parameters to backbone parameters: {ratio:.4f}")

    # logs
    log_overall, log_de = [], []
    log_subgroup = {n: [] for n in subgroup_names}
    target_groups = ["combo_seen0", "combo_seen1", "combo_seen2", "unseen_single"]

    # out dir
    out_dir = f"{args.out_dir_prefix}_dataset_{args.dataset}_inr_{args.inr}"
    os.makedirs(out_dir, exist_ok=True)

    best_score = np.inf
    best_round = -1

    # FL rounds
    for rnd in range(args.global_round):
        m = max(int(args.frac * args.n_client), 1)
        chosen = np.random.choice(range(args.n_client), m, replace=False)

        local_w = []

        for cid in chosen:
            idxs = list(dict_clients[cid])
            sub_loader = build_sub_dataloader(train_graphs, idxs, batch_size=args.batch_size)

            # shallow copy data holder to reuse AnnData
            pdata_client = copy.copy(pdata)
            pdata_client.dataloader = {
                "train_loader": sub_loader,
                "val_loader": pdata.dataloader["val_loader"],
            }

            # client model
            gears_client = GEARS(pdata_client, device=device, weight_bias_track=False)
            gears_client.model_initialize(**global_cfg)
            gears_client.pdata = pdata_client
            if args.inr:
                replace_linear_with_inr(
                    gears_client.model,
                    coord_dim=args.coord_dim,
                    coord_points=args.coord_points,
                    key_dim=args.key_dim,
                    key_hidden=args.key_hidden,
                    key_strength=args.key_strength,
                    coord_seed=args.coord_seed,
                    coord_mode=args.coord_mode,
                    coord_constant=args.coord_constant,
                    device=device,
                    use_film=not args.disable_film,
                )

            gears_client.model.load_state_dict(copy.deepcopy(gears_global.model.state_dict()))

            # local train
            gears_client.train(epochs=args.local_epoch, lr=args.lr, cid=cid)

            local_w.append(copy.deepcopy(gears_client.model.state_dict()))

        # FedAvg with prefix adjustment
        for i in range(len(local_w)):
            local_w[i] = adjust_module_prefix(local_w[i], add_prefix=False)
        new_glob_w = fed_avg(local_w)
        gears_global.model.load_state_dict(new_glob_w)
        print(f"Round {rnd+1}/{args.global_round} aggregation completed")

        # evaluate + logs/plots
        cur_score, test_pert_res = evaluate_and_log(
            gears_global, rnd, out_dir, subgroup_names, log_overall, log_de, log_subgroup, target_groups
        )

        # track best
        if cur_score is not None and cur_score < best_score:
            best_score = cur_score
            best_round = rnd + 1
            gears_global.save_model(out_dir, gears_global.model.state_dict(), best_score)
            print(f"        ✓ New best!  model saved to {out_dir}")

    print(f"Done! Best round={best_round}, best_score={best_score:.4f}")
    return {
        "out_dir": out_dir,
        "best_round": best_round,
        "best_score": best_score,
        "log_overall": log_overall,
        "log_de": log_de,
        "log_subgroup": log_subgroup,
        "subgroup_names": subgroup_names,
    }

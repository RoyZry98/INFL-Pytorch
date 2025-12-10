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


# =========================
# INR modules
# =========================
class INRLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 inr_input_dim=16,
                 inr_hidden_size=32,
                 inr_output_dim=1,
                 inr_layer_num=3,
                 inr_output_activation=None,
                 inr_output_bias=0.0,
                 device="cpu"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.inr = self._build_inr(
            inr_input_dim, inr_hidden_size, inr_output_dim, inr_layer_num,
            inr_output_activation, inr_output_bias, device
        )
        self.inr_input_dim = inr_input_dim
        self.out_features = out_features
        self.device = device
        self.register_buffer("coords", self._create_coordinates())

    def _build_inr(self, in_dim, hid, out_dim, num_layers, act, bias_val, device):
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hid, hid))
            layers.append(nn.ReLU())
        last = nn.Linear(hid, out_dim)
        if bias_val:
            torch.nn.init.normal_(last.bias, mean=bias_val)
        layers.append(last)
        if act == "relu":
            layers.append(nn.ReLU())
        elif act == "sigmoid":
            layers.append(nn.Sigmoid())
        elif act == "tanh":
            layers.append(nn.Tanh())
        return nn.Sequential(*layers).to(device)

    def _create_coordinates(self):
        idxs = torch.arange(self.out_features, dtype=torch.float32, device=self.device).unsqueeze(1)
        coords = idxs / (self.out_features - 1) * 2 - 1 if self.out_features > 1 else idxs
        pe = self._positional_encoding(coords)
        return pe

    def _positional_encoding(self, coords):
        # norman
        coords_dim = coords.shape[-1]
        L = int(self.inr_input_dim / 2 / coords_dim)
        encoded = torch.zeros(*coords.shape[:-1], coords_dim * 2 * L, device=coords.device)
        for i in range(coords_dim):
            freqs = 2.0 ** torch.linspace(0, L - 1, L, device=coords.device)
            for j in range(L):
                encoded[..., i * L + j] = torch.sin(freqs[j] * coords[..., i])
                encoded[..., i * L + j + L] = torch.cos(freqs[j] * coords[..., i])

        # adamson
        # L = self.inr_input_dim // 2
        # x = coords
        # pe = [torch.sin((2.0 ** i)* np.pi * x) for i in range(L)]
        # pe += [torch.cos((2.0 ** i)* np.pi * x) for i in range(L)]
        # pe = torch.cat(pe,dim=1)
        # encoded = pe

        return encoded

    def forward(self, x):
        # norman 0.6 adamson 0.4
        alpha = 0.6
        feat = self.linear(x)
        delta = self.inr(self.coords).view(1, -1)
        return alpha * feat + (1 - alpha) * delta


def replace_linear_with_inr(module,
                            inr_input_dim=16,
                            inr_hidden_size=32,
                            inr_output_dim=1,
                            inr_layer_num=3,
                            inr_output_activation=None,
                            inr_output_bias=0.0,
                            device="cpu"):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_layer = INRLinear(
                child.in_features, child.out_features,
                bias=(child.bias is not None),
                inr_input_dim=inr_input_dim,
                inr_hidden_size=inr_hidden_size,
                inr_output_dim=inr_output_dim,
                inr_layer_num=inr_layer_num,
                inr_output_activation=inr_output_activation,
                inr_output_bias=inr_output_bias,
                device=device
            )
            new_layer.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.linear.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_inr(
                child, inr_input_dim, inr_hidden_size, inr_output_dim,
                inr_layer_num, inr_output_activation, inr_output_bias, device
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
    for k in w_avg.keys():
        for i in range(1, len(state_dicts)):
            w_avg[k] += state_dicts[i][k]
        w_avg[k] = w_avg[k] / len(state_dicts)
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
    for m in model.modules():
        if isinstance(m, INRLinear):
            total += sum(p.numel() for p in m.inr.parameters())
    return total


def count_backbone_params(model: nn.Module) -> int:
    total = 0
    for n, p in model.named_parameters():
        if ".inr." not in n:
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
        replace_linear_with_inr(gears_global.model, inr_input_dim=16, inr_hidden_size=8, device=device)

    global_cfg = copy.deepcopy(gears_global.config)

    inr_params = count_inr_params(gears_global.model)
    backbone_params = count_backbone_params(gears_global.model)
    total_params = sum(p.numel() for p in gears_global.model.parameters())
    ratio = inr_params / backbone_params if backbone_params > 0 else float("nan")
    print(f"INR-MLP parameters: {inr_params}")
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
                replace_linear_with_inr(gears_client.model, inr_input_dim=16, inr_hidden_size=8, device=device)

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
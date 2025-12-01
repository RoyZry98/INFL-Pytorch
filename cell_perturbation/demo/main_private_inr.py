#!/usr/bin/env python
# coding: utf-8
"""
Federated GEARS – minimal demo
"""
import sys
sys.path.append('../')
import copy, argparse, os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import SGConv
from torch_geometric.data import DataLoader
from gears import PertData, GEARS          # pip install gears-pytorch 或你本地代码
from gears.utils import loss_fct, uncertainty_loss_fct   # 已在文件末导包，可省略
from gears.inference import (evaluate, compute_metrics,
                             deeper_analysis, non_dropout_analysis)
import numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
import math
import random
from copy import deepcopy
warnings.filterwarnings("ignore")

# Decentralized Randomization for Models
def decentralized_randomization(models):
    """Randomizes the order of local model weights for decentralized randomization."""
    for i in range(len(models) - 1):
        j = random.randint(i, len(models) - 1)  # Random index from i to len(models)-1
        models[i], models[j] = models[j], models[i]  # Swap models
    return models

class LoRALinear(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 r=4,  # LoRA rank
                 alpha=1.0,  # Scaling factor
                 bias=True, 
                 device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        # Original linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA components: low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device))

        # LoRA scaling: alpha/r
        self.scaling = alpha / r

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original linear transformation
        output = self.linear(x)

        # LoRA adjustment
        lora_adjustment = torch.matmul(self.lora_B, self.lora_A)  # Low-rank weight adjustment
        lora_adjustment = torch.matmul(x, lora_adjustment.T)  # Apply to input

        # Scale and add LoRA adjustment to output
        return output + self.scaling * lora_adjustment


def replace_linear_with_lora(module, 
                             r=4, 
                             alpha=1.0, 
                             device='cpu'):
    """
    Replace all nn.Linear layers in a given module with LoRALinear layers.
    
    Args:
        module (nn.Module): The neural network module to modify.
        r (int): The rank of the LoRA low-rank matrices.
        alpha (float): The scaling factor for LoRA adjustments.
        device (str): The device to place the new layers on.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace nn.Linear with LoRALinear
            new_layer = LoRALinear(
                child.in_features, 
                child.out_features, 
                r=r, 
                alpha=alpha, 
                bias=(child.bias is not None), 
                device=device
            )

            # Copy the original weights and bias
            new_layer.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.linear.bias.data.copy_(child.bias.data)

            # Replace the layer in the module
            setattr(module, name, new_layer)
        else:
            # Recursively apply to child modules
            replace_linear_with_lora(child, r=r, alpha=alpha, device=device)

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
                 inr_output_bias=0.,
                 device='cpu'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.inr = self._build_inr(inr_input_dim, inr_hidden_size, inr_output_dim, inr_layer_num, inr_output_activation, inr_output_bias, device)
        self.inr_input_dim = inr_input_dim
        self.out_features = out_features
        self.device = device
        self.register_buffer('coords', self._create_coordinates())
    
    def _build_inr(self, in_dim, hid, out_dim, num_layers, act, bias, device):
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(in_dim if i==0 else hid, hid))
            layers.append(nn.ReLU())
        last = nn.Linear(hid, out_dim)
        if bias:
            torch.nn.init.normal_(last.bias, mean=bias)
        layers.append(last)
        if act == 'relu':
            layers.append(nn.ReLU())
        elif act == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif act == 'tanh':
            layers.append(nn.Tanh())
        return nn.Sequential(*layers).to(device)

    def _create_coordinates(self):
        idxs = torch.arange(self.out_features, dtype=torch.float32, device=self.device).unsqueeze(1)
        coords = idxs / (self.out_features-1) * 2 - 1 if self.out_features > 1 else idxs
        pe = self._positional_encoding(coords)
        return pe

    def _positional_encoding(self, coords):
        # 初始化编码张量
        coords_dim = coords.shape[-1]
        L = int(self.inr_input_dim / 2 / coords_dim)
        encoded = torch.zeros(*coords.shape[:-1], coords_dim * 2 * L, device=coords.device)
        # 对每个坐标维度进行位置编码
        for i in range(coords_dim):
            # 计算2^k次幂
            frequencies = 2.0 ** torch.linspace(0, L - 1, L, device=coords.device)
            for j in range(L):
                encoded[..., i * L + j] = torch.sin(frequencies[j] * coords[..., i])
                encoded[..., i * L + j + L] = torch.cos(frequencies[j] * coords[..., i])
        return encoded

    def forward(self, x):
        alpha = 0.3
        feat = self.linear(x)
        delta = self.inr(self.coords).view(1, -1)
        return alpha*feat + (1-alpha)*delta

def replace_linear_with_inr(module, 
                           inr_input_dim=16, 
                           inr_hidden_size=32, 
                           inr_output_dim=1,
                           inr_layer_num=3, 
                           inr_output_activation=None, 
                           inr_output_bias=0.,
                           device='cpu'):
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
            # 拷贝原始权重和bias
            new_layer.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.linear.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_inr(child, inr_input_dim, inr_hidden_size, inr_output_dim, inr_layer_num, inr_output_activation, inr_output_bias, device)

# --------------------------------------------------------------------- #
# 1. 数据切分函数  (IID / Dirichlet-NonIID)
# --------------------------------------------------------------------- #
def graphs_iid(dataset, num_users, seed=None):
    """均匀随机划分"""
    if seed is not None:
        np.random.seed(seed)
    num_items = len(dataset) // num_users
    all_idxs = np.arange(len(dataset))
    dict_users = {}
    for uid in range(num_users):
        dict_users[uid] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = np.setdiff1d(all_idxs, list(dict_users[uid]))
    dict_users[num_users-1].update(all_idxs)   # 把余数给最后一个
    return dict_users


def graphs_noniid_dirichlet(dataset, labels, num_users, alpha=0.5, seed=None):
    """Dirichlet non-IID 划分"""
    if seed is not None:
        np.random.seed(seed)
    labels = np.asarray(labels)
    classes, y = np.unique(labels, return_inverse=True)
    dict_users = {i: set() for i in range(num_users)}
    idxs = np.arange(len(dataset))
    for c in range(len(classes)):
        idxs_c = idxs[y == c]
        np.random.shuffle(idxs_c)
        props = np.random.dirichlet([alpha] * num_users)
        props = (props / props.sum() * len(idxs_c)).astype(int)
        splits = np.split(idxs_c, np.cumsum(props)[:-1])
        for i, sp in enumerate(splits):
            dict_users[i].update(sp)
    return dict_users


# --------------------------------------------------------------------- #
# 2. FedAvg
# --------------------------------------------------------------------- #
def fed_avg(state_dicts):
    w_avg = copy.deepcopy(state_dicts[0])
    for k in w_avg.keys():
        for i in range(1, len(state_dicts)):
            w_avg[k] += state_dicts[i][k]
        w_avg[k] = w_avg[k] / len(state_dicts)
    return w_avg

def count_inr_params(model):
    total = 0
    for m in model.modules():
        if isinstance(m, INRLinear):
            total += sum(p.numel() for p in m.inr.parameters())
    return total

def count_backbone_params(model):
    total = 0
    for n, p in model.named_parameters():
        # 不统计INR-MLP参数
        if ".inr." not in n:
            total += p.numel()
    return total

def adjust_module_prefix(state_dict, add_prefix=False):
    """
    动态调整 state_dict 的键：添加或移除 "model." 前缀

    Args:
        state_dict (dict): 模型的 state_dict。
        add_prefix (bool): 如果为 True，添加 "model." 前缀；如果为 False，移除 "model." 前缀。

    Returns:
        dict: 调整后的 state_dict。
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if add_prefix:
            # 如果需要添加前缀，确保 key 没有 "model." 前缀
            if not key.startswith("model."):
                new_key = "model." + key
            else:
                new_key = key
        else:
            # 如果需要移除前缀，确保 key 有 "model." 前缀
            if key.startswith("model."):
                new_key = key[len("model."):]  # 去掉 "model."
            else:
                new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

# --------------------------------------------------------------------- #
# 3. 命令行参数
# --------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--n_client', type=int, default=3)
    p.add_argument('--iid', action='store_true', help='use IID split')
    p.add_argument('--alpha', type=float, default=0.3, help='Dirichlet α')
    p.add_argument('--global_round', type=int, default=200)
    p.add_argument('--local_epoch', type=int, default=2)
    p.add_argument('--frac', type=float, default=1,
                   help='参与每轮训练客户端比例')
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--inr', action='store_true', help='turn on inr-mlp')
    p.add_argument('--lora', action='store_true', help='turn on lora')
    p.add_argument('--dr', action='store_true', help='turn on dr')

    # ------- DP 相关新参数 -------
    p.add_argument('--dp', action='store_true', help='turn on DP-SGD')
    p.add_argument('--noise', type=float, default=2.36,
                   help='Gaussian noise multiplier σ')
    p.add_argument('--max_grad_norm', type=float, default=1.0,
                   help='per-sample grad clipping L2 norm')
    p.add_argument('--target_delta', type=float, default=1e-5,
                   help='δ in (ε,δ)-DP')
    p.add_argument('--weight_decay', type=float, default=5e-4,
                   help='δ in (ε,δ)-DP')
    return p.parse_args()


# --------------------------------------------------------------------- #
# 4. MAIN
# --------------------------------------------------------------------- #
if __name__ == '__main__':
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')

    best_score  = np.inf      # ★ 新增：记录目前最优平均 mse_de
    best_round  = -1

    # target_groups = ['combo_seen0', 'combo_seen1',
    #                 'combo_seen2', 'unseen_single']    # ★ 你要考察的4组

    # ---------------- 数据准备（同单机用法） -----------------------------
    pdata = PertData('/deltadisk/zhangrongyu/GEARS/demo/data')
    pdata.load(data_path = '/deltadisk/zhangrongyu/GEARS/demo/data/private')
    pdata.prepare_split(split='custom', split_dict_path='custom_split.pkl')# get data split with seed
    pdata.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader

    train_graphs = pdata.dataloader['train_loader'].dataset
    train_labels = [g.pert for g in train_graphs]  # 用 perturbation 作标签

    # =============== 新增：结果缓存容器 =================
    log_overall   = []                      # 每轮 overall MSE
    log_de        = []                      # 每轮 top-20-DE MSE
    # subgroup_names = list(pdata.subgroup['test_subgroup'].keys())
    # log_subgroup  = {n: [] for n in subgroup_names}   # 每轮子集 MSE_DE

    # ------------------- 联邦划分 ---------------------------------------
    if args.iid:
        dict_clients = graphs_iid(train_graphs, args.n_client, seed=0)
    else:
        dict_clients = graphs_noniid_dirichlet(train_graphs, train_labels,
                                               args.n_client, alpha=args.alpha,
                                               seed=0)
    print('每个客户端样本数量：',
          [len(v) for v in dict_clients.values()])

    # ------------------ 初始化全局 GEARS 模型 ---------------------------
    gears_global = GEARS(pdata, device=device, weight_bias_track=False)
    gears_global.model_initialize(hidden_size=64, dp=False)
    if args.inr:
        replace_linear_with_inr(gears_global.model, inr_input_dim=16, inr_hidden_size=8, device=device)
    elif args.lora:
        replace_linear_with_lora(gears_global.model, r=4, alpha=1.0, device=device)
    # 把“静态”配置保存下来，客户端要复用
    global_cfg = copy.deepcopy(gears_global.config)

    inr_params = count_inr_params(gears_global.model)
    backbone_params = count_backbone_params(gears_global.model)
    total_params = sum(p.numel() for p in gears_global.model.parameters())
    ratio = inr_params / backbone_params if backbone_params > 0 else float("nan")

    print(f"INR-MLP参数数: {inr_params}")
    print(f"主干（非INR）参数数: {backbone_params}")
    print(f"全部参数数: {total_params}")
    print(f"INR占主干参数比例: {ratio:.4f}")

    # ------------------ 联邦迭代 ---------------------------------------
    for rnd in range(args.global_round):
        m = max(int(args.frac * args.n_client), 1)
        chosen = np.random.choice(range(args.n_client), m, replace=False)

        # ========= 每轮联邦，遍历被选中的客户端 ==============
        local_w, local_loss = [], []

        for cid in chosen:
            idxs = list(dict_clients[cid])

            # 1) 用本客户端样本生成 DataLoader
            sub_loader = DataLoader([train_graphs[i] for i in idxs],
                                    batch_size=32, shuffle=True, drop_last=True)

            # 2) --- 只做浅拷贝，避免复制 AnnData ----
            import copy
            pdata_client           = copy.copy(pdata)       # 浅拷贝
            pdata_client.dataloader = {
                'train_loader': sub_loader,
                'val_loader'  : pdata.dataloader['val_loader']
            }

            # 3) 初始化客户端 GEARS，并载入全局权重
            gears_client = GEARS(pdata_client, device=device, weight_bias_track=False)
            gears_client.model_initialize(**global_cfg, dp=False)
            if args.inr:
                replace_linear_with_inr(gears_client.model, inr_input_dim=16, inr_hidden_size=8, device=device)
            elif args.lora:
                replace_linear_with_lora(gears_client.model, r=4, alpha=1.0, device=device)
            gears_client.model.load_state_dict(copy.deepcopy(gears_global.model.state_dict()))
         
            # 4) 本地训练（调用 GEARS.train）
            if args.dp:
                gears_client.train_with_dp(epochs=args.local_epoch, lr=args.lr, cid=cid, noise=args.noise)
            else:
                gears_client.train(epochs=args.local_epoch, lr=args.lr, cid=cid)

            # 5) 收集权重
            local_w.append(copy.deepcopy(gears_client.model.state_dict()))
            # GEARS.train 里已经打印 loss，这里就不再额外统计

        # Apply Decentralized Randomization to local models (if enabled)
        if args.dr:
            print("Applying Decentralized Randomization to local models...")
            local_w = decentralized_randomization(local_w)

        # ---------------- FedAvg 聚合 ------------------
        for i in range(len(local_w)):
                local_w[i] = adjust_module_prefix(local_w[i], add_prefix=False)
        new_glob_w = fed_avg(local_w)
        gears_global.model.load_state_dict(new_glob_w)
        print(f'Round {rnd+1}/{args.global_round} 聚合完成')

        # ============================================================
        # ★★★  模型测试  ★★★
        # ============================================================
        if 'test_loader' in gears_global.dataloader:
            test_loader = gears_global.dataloader['test_loader']
            gears_global.model.eval()                # 切到 eval 模式
            with torch.no_grad():
                test_res = evaluate(test_loader,
                                    gears_global.model,
                                    False,           # 你如果打开了 uncertainty 就写 True
                                    device)
            test_metrics, test_pert_res = compute_metrics(test_res)
            print(f'    └── Test  Round {rnd+1} Overall  MSE : {test_metrics["mse"]:.4f}')
            print(f'        Test Round {rnd+1} Top-20-DE MSE : {test_metrics["mse_de"]:.4f}')
        else:
            print('    └── 未检测到 test_loader，跳过测试。')


        # print("Start doing subgroup analysis for simulation split...")
        # subgroup = pdata.subgroup
        # subgroup_analysis = {}
        # for name in subgroup['test_subgroup'].keys():
        #     subgroup_analysis[name] = {}
        #     for m in list(list(test_pert_res.values())[0].keys()):
        #         subgroup_analysis[name][m] = []

        # for name, pert_list in subgroup['test_subgroup'].items():
        #     for pert in pert_list:
        #         for m, res in test_pert_res[pert].items():
        #             subgroup_analysis[name][m].append(res)

        # for name, result in subgroup_analysis.items():
        #     for m in result.keys():
        #         subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])

        #         print('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

        # out = deeper_analysis(pdata.adata, test_res)
        # out_non_dropout = non_dropout_analysis(pdata.adata, test_res)

        # metrics = ['pearson_delta']
        # metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout',
        #                        'frac_sigma_below_1_non_dropout',
        #                        'mse_top20_de_non_dropout']
        
        ## deeper analysis
        # subgroup_analysis = {}
        # for name in subgroup['test_subgroup'].keys():
        #     subgroup_analysis[name] = {}
        #     for m in metrics:
        #         subgroup_analysis[name][m] = []

        #     for m in metrics_non_dropout:
        #         subgroup_analysis[name][m] = []

        # for name, pert_list in subgroup['test_subgroup'].items():
        #     for pert in pert_list:
        #         for m in metrics:
        #             subgroup_analysis[name][m].append(out[pert][m])

        #         for m in metrics_non_dropout:
        #             subgroup_analysis[name][m].append(out_non_dropout[pert][m])

        # for name, result in subgroup_analysis.items():
        #     for m in result.keys():
        #         subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])

        #         print('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))


        # ------------- 1) 记录整体/DE MSE ---------------------------
        log_overall.append(test_metrics['mse'])
        log_de.append(test_metrics['mse_de'])

        # ------------- 2) 记录各 simulation-subgroup ----------------
        # for name in subgroup_names:                         # subgroup_names 在脚本最前面得到
        #     perts = pdata.subgroup['test_subgroup'][name]
        #     vals  = [test_pert_res[p]['mse_de'] for p in perts]
        #     mean_val = np.mean(vals) if len(vals) > 0 else np.nan
        #     log_subgroup[name].append(mean_val)
        #     print(f'        {name:<18}: {mean_val:.4f}')

        # ============================================================
        # ★★★  判断是否刷新最好模型  ★★★
        # ============================================================
        cur_score = test_metrics["mse_de"]

        if cur_score < best_score:
            best_score = cur_score
            best_round = rnd + 1
            save_dir   = f'private_INFL_inr_16_{args.inr}_dp_{args.dp}_lora_{args.lora}_dr_{args.dr}'
            os.makedirs(save_dir, exist_ok=True)
            gears_global.save_model(save_dir, gears_global.model.state_dict(), best_score)          # ★ 保存模型
            print(f'        ✓ New best MSE_DE {cur_score:.6f} model saved to {save_dir}')

        # ============================================================
        #   每一轮都把日志写 CSV，并即时画图（覆盖式保存）
        # ============================================================
        cur_rounds = np.arange(1, len(log_overall) + 1)   # ★ 当前长度 ★

        df_dict = {
            'round'       : cur_rounds,
            'overall_mse' : log_overall,
            'top20de_mse' : log_de,
        }
        # for n in subgroup_names:
        #     df_dict[f'{n}_mse_de'] = log_subgroup[n]

        df = pd.DataFrame(df_dict)
        os.makedirs(f'private_INFL_inr_16_{args.inr}_dp_{args.dp}_lora_{args.lora}_dr_{args.dr}', exist_ok=True)
        df.to_csv(f'private_INFL_inr_16_{args.inr}_dp_{args.dp}_lora_{args.lora}_dr_{args.dr}/fed_test_metrics.csv', index=False)

        # ------- 画总体 MSE 曲线 (覆盖保存) --------------------------
        plt.figure(figsize=(6,3.5))
        plt.plot(cur_rounds, log_overall, marker='o', label='Overall MSE')
        plt.plot(cur_rounds, log_de,      marker='s', label='Top-20-DE MSE')
        plt.xlabel('Global Round'); plt.ylabel('MSE')
        plt.title('Test MSE vs. Rounds')
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(f'private_INFL_inr_16_{args.inr}_dp_{args.dp}_lora_{args.lora}_dr_{args.dr}/test_mse_curve.png', dpi=300)
        plt.close()                               # 释放内存

        # ------- 画子集曲线 (覆盖保存) ------------------------------
        # plt.figure(figsize=(6,3.5))
        # for n in subgroup_names:
        #     plt.plot(cur_rounds, log_subgroup[n], marker='.', label=n)
        # plt.xlabel('Global Round'); plt.ylabel('Top-20-DE MSE')
        # plt.title('Sub-group MSE'); plt.grid(alpha=.3)
        # plt.legend(fontsize=7, ncol=2); plt.tight_layout()
        # plt.savefig('private_INFL_inr_16_{args.inr}_dp_{args.dp}/subgroup_mse_curve.png', dpi=300)
        # plt.close()

    print('Done!')
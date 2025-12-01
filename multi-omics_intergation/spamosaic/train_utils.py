import os, gc
from pathlib import Path, PurePath
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
import h5py
import math
from tqdm import tqdm
import scipy.sparse as sps
import scipy.io as sio
import seaborn as sns
import warnings
# import gzip
from scipy.io import mmread
import random
from os.path import join
import torch

import logging
import torch
from matplotlib import rcParams
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from torch_geometric.utils import negative_sampling

import random
def set_seeds(seed, dt=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(dt)  # ensure reproducibility

def graph_decode(z, edge_index):
    logit = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    return torch.sigmoid(logit)

def graph_recon_crit(z, pos_edge_index):
    # graph reconstruction loss
    pos_loss = -torch.log(graph_decode(z, pos_edge_index)+1e-20)
    pos_loss = pos_loss.mean()

    neg_edge_index = negative_sampling(
        pos_edge_index, z.size(0), num_neg_samples=pos_edge_index.size(1)
    )
    # neg_edge_index = self.subgraph_negative_sampling(pos_edge_index)
    neg_loss = -torch.log(1 - graph_decode(z, neg_edge_index) + 1e-20)
    neg_loss = neg_loss.mean()
    return pos_loss + neg_loss

def train_model(
        mod_model, mod_optims, crit1, crit2, loss_type, w_rec_g, 
        mod_input_feat, mod_graphs_edge, mod_graphs_edge_w, mod_graphs_edge_c, 
        batch_train_meta_numbers, mod_batch_split, 
        T, n_epochs, device
    ):
    bridge_batch_num_ids = [bi for bi,v in batch_train_meta_numbers.items() if v[0]]
    test_batch_num_ids   = [bi for bi,v in batch_train_meta_numbers.items() if not v[0]]

    loss_cl, loss_rec = [], []
    for epoch in tqdm(range(1, n_epochs+1)):
        for k in mod_model.keys(): mod_model[k].train()
        for k in mod_optims.keys(): mod_optims[k].zero_grad() 

        mod_embs, mod_recs = {}, {}
        for k in mod_model.keys():
            z, r = mod_model[k](mod_input_feat[k], mod_graphs_edge[k], mod_graphs_edge_w[k])  # outputs for all spots within each modality 
            mod_embs[k] = list(torch.split(z, mod_batch_split[k]))  
            mod_recs[k] = list(torch.split(r, mod_batch_split[k]))   
        split_input_feat = {k: list(torch.split(v, mod_batch_split[k])) for k,v in mod_input_feat.items()}  # target of reconstruction 

        # feature reconstruction
        l2, l2_step = torch.FloatTensor([0]).to(device), 0  # in case empty test batches
        for bi in test_batch_num_ids:
            ms = batch_train_meta_numbers[bi][1]
            for k in ms:
                # feature reconst
                f_rec_loss = 0 if w_rec_g==1 else crit2(mod_recs[k][bi], split_input_feat[k][bi]) 
                
                if w_rec_g > 0:
                    # graph reconst
                    intra_graph_mask = mod_graphs_edge_c[k] == bi
                    sub_graph = mod_graphs_edge[k][:, intra_graph_mask]
                    start_idx = torch.unique(sub_graph).min()
                    sub_graph_edge_index = sub_graph - start_idx
                    g_rec_loss = graph_recon_crit(mod_embs[k][bi], sub_graph_edge_index)
                else:
                    g_rec_loss = 0.

                l2 += w_rec_g * g_rec_loss + (1-w_rec_g) * f_rec_loss
                l2_step += 1

        # contrastive learning
        l1, l1_step = 0, 0 
        for bi in bridge_batch_num_ids:
            meta = batch_train_meta_numbers[bi]
            # if setting multiple mini-batches, randomly shuffle the spots 
            if meta[3] > 1:                       
                shuffle_ind = torch.randperm(meta[1]).to(device)
                for k in meta[5]: 
                    mod_embs[k][bi] = mod_embs[k][bi][shuffle_ind]

            for i in range(meta[3]):
                feat = [mod_embs[k][bi][i*meta[2]:(i+1)*meta[2]] for k in meta[5]]
                if loss_type=='adapted':
                    feat = torch.cat(feat)
                    l1 += crit1[bi]((feat @ feat.t()) / T)
                else:
                    logit = (feat[0] @ feat[1].T)/T
                    target = torch.arange(meta[2]).to(device)
                    l1 += 0.5 * (crit1[bi](logit, target) + crit1[bi](logit.T, target))
                l1_step += 1
        
        loss = l1/max(1, l1_step) + l2/max(1, l2_step) 

        loss.backward()
        for k in mod_optims.keys(): mod_optims[k].step()

        if l1_step > 0:
            loss_cl.append(l1.item())
        if l2_step > 0:
            loss_rec.append(l2.item())

    return mod_model, loss_cl, loss_rec, loss.item()

def _init_rdp_state(alpha_list):
    return {float(a): 0.0 for a in alpha_list}

def _update_rdp_state(rdp_state, alpha_list, q, noise_mult):
    for alpha in alpha_list:
        rdp_state[float(alpha)] += q ** 2 * alpha / (2 * noise_mult ** 2)

def _get_eps_from_rdp(rdp_state, alpha_list, delta):
    epsilons = [rdp_state[float(alpha)] + math.log(1/delta) / (alpha - 1) for alpha in alpha_list]
    min_idx = int(torch.tensor(epsilons).argmin())
    return float(epsilons[min_idx]), float(alpha_list[min_idx])


def train_model_with_dp(
    mod_model, mod_optims, crit1, crit2, loss_type, w_rec_g, 
    mod_input_feat, mod_graphs_edge, mod_graphs_edge_w, mod_graphs_edge_c, 
    batch_train_meta_numbers, mod_batch_split, 
    T, n_epochs, device,
    dp_epsilon=1.29, dp_clip_norm=1.0, dp_delta=1e-5, batch_size=32,
    verbose=True
):
    alpha_list = torch.tensor([1.25, 1.5, 2, 4, 8, 16, 32, 64], dtype=torch.double)
    rdp_state = _init_rdp_state(alpha_list)
    # dp_noise_scale = 1
    # noise_mult = dp_noise_scale / dp_clip_norm
    dp_epsilon=2.36
    delta=1e-5
    dp_noise_scale = dp_clip_norm * math.sqrt(2 * math.log(1.25 / delta)) / dp_epsilon
    noise_mult     = dp_noise_scale / dp_clip_norm 

    mod_n_samples = {k: mod_input_feat[k].shape[0] for k in mod_input_feat}
    min_n_samples = min(mod_n_samples.values())
    n_steps = int(np.ceil(min_n_samples / batch_size))

    loss_rec_all = []

    for epoch in range(1, n_epochs+1):
        epoch_losses = []
        if verbose:
            pbar = tqdm(mod_model.keys(), desc=f"Epoch {epoch}", disable=not verbose)
        else:
            pbar = mod_model.keys()

        for k in pbar:
            model = mod_model[k]
            optimizer = mod_optims[k]
            model.train()
            x = mod_input_feat[k]
            edge_index = mod_graphs_edge[k]
            edge_weight = mod_graphs_edge_w[k]

            n_samples = x.shape[0]
            indices = torch.randperm(n_samples)
            mini_losses = []

            for step in range(n_steps):
                mini_idx = indices[step*batch_size : (step+1)*batch_size]
                if len(mini_idx) == 0:
                    continue

                z, r = model(x, edge_index, edge_weight)
                loss = crit2(r[mini_idx], x[mini_idx]).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), dp_clip_norm)
            for param in model.parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), dp_clip_norm)
                    noise = torch.normal(0.0, dp_noise_scale, size=param.grad.shape, device=param.grad.device)
                    param.grad += noise
                optimizer.step()

                mini_losses.append(loss.item())

                # DP采样率
                sampling_rate = len(mini_idx) / n_samples
                _update_rdp_state(rdp_state, alpha_list, sampling_rate, noise_mult)

                if verbose and step % 20 == 0:
                    eps, _ = _get_eps_from_rdp(rdp_state, alpha_list, dp_delta)
                    pbar.set_postfix({"loss": f"{mini_losses[-1]:.4f}", "ε": f"{eps:.2f}"})

            # 该模态本epoch的平均loss
            if len(mini_losses) > 0:
                epoch_losses.append(np.mean(mini_losses))

        # 所有模态的平均loss
        if len(epoch_losses) > 0:
            loss_rec_all.append(np.mean(epoch_losses))

    final_eps, best_alpha = _get_eps_from_rdp(rdp_state, alpha_list, dp_delta)
    print(f"DP training done! (ε, δ)=({final_eps:.2f}, {dp_delta}), α*={best_alpha:.2f}")

    return mod_model, None, loss_rec_all, None

def train_model_with_multimodal_dp(
    mod_model, mod_optims, crit1, crit2, loss_type, w_rec_g, 
    mod_input_feat, mod_graphs_edge, mod_graphs_edge_w, mod_graphs_edge_c, 
    batch_train_meta_numbers, mod_batch_split, 
    T, n_epochs, device,
    dp_epsilon=1.29, dp_clip_norm=1.0, dp_delta=1e-5, batch_size=32,
    verbose=True
):
    alpha_list = torch.tensor([1.25, 1.5, 2, 4, 8, 16, 32, 64], dtype=torch.double)
    rdp_state = _init_rdp_state(alpha_list)
    dp_epsilon = 2.36
    delta = 1e-5
    dp_noise_scale = dp_clip_norm * math.sqrt(2 * math.log(1.25 / delta)) / dp_epsilon
    noise_mult = dp_noise_scale / dp_clip_norm 

    mod_n_samples = {k: mod_input_feat[k].shape[0] for k in mod_input_feat}
    min_n_samples = min(mod_n_samples.values())
    n_steps = int(np.ceil(min_n_samples / batch_size))

    loss_rec_all, loss_cl_all = [], []

    bridge_batch_num_ids = [bi for bi, v in batch_train_meta_numbers.items() if v[0]]
    test_batch_num_ids   = [bi for bi, v in batch_train_meta_numbers.items() if not v[0]]

    for epoch in range(1, n_epochs+1):
        epoch_losses = []
        epoch_cl_losses = []
        if verbose:
            pbar = tqdm(mod_model.keys(), desc=f"Epoch {epoch}", disable=not verbose)
        else:
            pbar = mod_model.keys()

        # 1. 先前向收集所有embedding和recon
        for k in mod_model.keys():
            mod_model[k].train()
        mod_embs, mod_recs = {}, {}
        for k in mod_model.keys():
            z, r = mod_model[k](mod_input_feat[k], mod_graphs_edge[k], mod_graphs_edge_w[k])
            mod_embs[k] = list(torch.split(z, mod_batch_split[k]))
            mod_recs[k] = list(torch.split(r, mod_batch_split[k]))
        split_input_feat = {k: list(torch.split(v, mod_batch_split[k])) for k, v in mod_input_feat.items()}

        # 2. 计算重构loss（l2）
        l2, l2_step = torch.FloatTensor([0]).to(device), 0
        for bi in test_batch_num_ids:
            ms = batch_train_meta_numbers[bi][1]
            for k in ms:
                f_rec_loss = 0 if w_rec_g==1 else crit2(mod_recs[k][bi], split_input_feat[k][bi])
                if w_rec_g > 0:
                    intra_graph_mask = mod_graphs_edge_c[k] == bi
                    sub_graph = mod_graphs_edge[k][:, intra_graph_mask]
                    start_idx = torch.unique(sub_graph).min()
                    sub_graph_edge_index = sub_graph - start_idx
                    g_rec_loss = graph_recon_crit(mod_embs[k][bi], sub_graph_edge_index)
                else:
                    g_rec_loss = 0.
                l2 += w_rec_g * g_rec_loss + (1-w_rec_g) * f_rec_loss
                l2_step += 1

        # 3. 计算对比学习loss（l1）
        l1, l1_step = 0, 0
        for bi in bridge_batch_num_ids:
            meta = batch_train_meta_numbers[bi]
            if meta[3] > 1:                       
                shuffle_ind = torch.randperm(meta[1]).to(device)
                for k in meta[5]: 
                    mod_embs[k][bi] = mod_embs[k][bi][shuffle_ind]
            for i in range(meta[3]):
                feat = [mod_embs[k][bi][i*meta[2]:(i+1)*meta[2]] for k in meta[5]]
                if loss_type=='adapted':
                    feat = torch.cat(feat)
                    l1 += crit1[bi]((feat @ feat.t()) / T)
                else:
                    logit = (feat[0] @ feat[1].T)/T
                    target = torch.arange(meta[2]).to(device)
                    l1 += 0.5 * (crit1[bi](logit, target) + crit1[bi](logit.T, target))
                l1_step += 1

        loss = l1 / max(1, l1_step) + l2 / max(1, l2_step)   # 总loss

        # 4. DP的参数更新（加噪声+clip）
        for k in pbar:
            optimizer = mod_optims[k]
            optimizer.zero_grad()
        loss.backward()
        for k in mod_model.keys():
            model = mod_model[k]
            optimizer = mod_optims[k]
            torch.nn.utils.clip_grad_norm_(model.parameters(), dp_clip_norm)
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.normal(0.0, dp_noise_scale, size=param.grad.shape, device=param.grad.device)
                    param.grad += noise
            optimizer.step()
        # 采样率计算
        sampling_rate = batch_size / min_n_samples
        _update_rdp_state(rdp_state, alpha_list, sampling_rate, noise_mult)

        if verbose:
            eps, _ = _get_eps_from_rdp(rdp_state, alpha_list, dp_delta)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "ε": f"{eps:.2f}"})

        epoch_losses.append(l2.item())
        epoch_cl_losses.append(l1.item())

        loss_rec_all.append(np.mean(epoch_losses))
        loss_cl_all.append(np.mean(epoch_cl_losses))

    final_eps, best_alpha = _get_eps_from_rdp(rdp_state, alpha_list, dp_delta)
    print(f"DP training done! (ε, δ)=({final_eps:.2f}, {dp_delta}), α*={best_alpha:.2f}")

    return mod_model, loss_cl_all, loss_rec_all, loss.item()
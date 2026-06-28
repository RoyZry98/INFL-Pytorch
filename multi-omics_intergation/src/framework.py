import os, gc
from pathlib import Path, PurePath
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
from tqdm import tqdm
import scipy.sparse as sps
import warnings
import itertools
from os.path import join
import torch
import pkg_resources

import logging
import torch
from matplotlib import rcParams
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import GAE
from torch_geometric.utils import train_test_split_edges, negative_sampling

from src.train_utils import set_seeds, train_model
import src.utils as utls
import src.build_graph as build_graph
import src.architectures as archs
from src.loss import CL_loss

# cur_dir = os.path.dirname(os.path.abspath(__file__))

def make_coordinates(
        seed,
        mode,
        num_points,
        coord_dim,
        device,
        constant=1.0):
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))

    if mode == 'uniform':
        coords = torch.rand(num_points, coord_dim, generator=g) * 2.0 - 1.0
    elif mode == 'normal':
        coords = torch.randn(num_points, coord_dim, generator=g)
    elif mode == 'ones':
        coords = torch.ones(num_points, coord_dim)
    elif mode == 'zeros':
        coords = torch.zeros(num_points, coord_dim)
    elif mode == 'negones':
        coords = -torch.ones(num_points, coord_dim)
    elif mode == 'constant':
        coords = torch.full((num_points, coord_dim), float(constant))
    else:
        raise ValueError(f'Unknown coordinate mode: {mode}')

    return coords.to(device)


class CoordinateKeyEncoder(nn.Module):
    def __init__(self, coord_dim, num_points, key_dim, coords):
        super().__init__()

        if coords.shape != (num_points, coord_dim):
            raise ValueError(
                f'coords shape mismatch: expected {(num_points, coord_dim)}, got {tuple(coords.shape)}'
            )

        self.coord_dim = int(coord_dim)
        self.num_points = int(num_points)
        self.key_dim = int(key_dim)

        self.register_buffer('coords', coords.detach().clone())

        self.net = nn.Sequential(
            nn.Linear(coord_dim, key_dim),
            nn.SiLU(),
            nn.Linear(key_dim, key_dim),
            nn.SiLU(),
            nn.LayerNorm(key_dim),
        )

    def set_coordinates(self, coords):
        if coords.shape != self.coords.shape:
            raise ValueError(
                f'coords shape mismatch: expected {tuple(self.coords.shape)}, got {tuple(coords.shape)}'
            )

        with torch.no_grad():
            self.coords.copy_(
                coords.to(device=self.coords.device, dtype=self.coords.dtype)
            )

    def forward(self):
        return self.net(self.coords).mean(dim=0)


class KeyControlledLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            key_dim,
            hidden_dim,
            key_strength,
            bias=True):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.key_strength = float(key_strength)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.key_to_mod = nn.Sequential(
            nn.Linear(key_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features * 2),
        )

        self.reset_parameters()
        self._init_key_mod_near_identity()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if self.bias is not None:
            bound = 1 / np.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _init_key_mod_near_identity(self):
        last = self.key_to_mod[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=0.02)
            nn.init.zeros_(last.bias)

    def make_effective_params(self, key_vec):
        mod = self.key_to_mod(key_vec)
        scale_raw, bias_raw = mod.chunk(2, dim=0)

        scale = 1.0 + self.key_strength * torch.tanh(scale_raw)
        scale = scale.view(self.out_features, 1)

        effective_weight = self.weight * scale

        if self.bias is None:
            effective_bias = None
        else:
            bias_delta = self.key_strength * torch.tanh(bias_raw)
            effective_bias = self.bias + bias_delta

        return effective_weight, effective_bias

    def forward(self, x, key_vec):
        w, b = self.make_effective_params(key_vec)
        return F.linear(x, w, b)

    def forward_without_key(self, x):
        return F.linear(x, self.weight, self.bias)


class ActivationDependentFiLM1d(nn.Module):
    def __init__(self, features, key_dim, hidden_dim, key_strength):
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

    def _init_near_identity(self):
        last = self.film[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=0.02)
            nn.init.zeros_(last.bias)

    def forward(self, h, key_vec):
        batch_size = h.shape[0]

        act_embed = self.act_proj(h)
        key_embed = key_vec.unsqueeze(0).expand(batch_size, -1)

        joint = torch.cat(
            [
                act_embed,
                key_embed,
                act_embed * key_embed,
            ],
            dim=1,
        )

        gamma_beta = self.film(joint)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)

        return h * (1.0 + self.key_strength * gamma) + self.key_strength * beta


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

        coord_dim = 2
        coord_points = 128
        key_dim = int(inr_input_dim)
        key_hidden = int(inr_hidden_size)
        key_strength = 1.0

        coords = make_coordinates(
            seed=0,
            mode='uniform',
            num_points=coord_points,
            coord_dim=coord_dim,
            device=torch.device(device),
        )

        self.key_encoder = CoordinateKeyEncoder(
            coord_dim=coord_dim,
            num_points=coord_points,
            key_dim=key_dim,
            coords=coords,
        )
        self.linear = KeyControlledLinear(
            in_features,
            out_features,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
            bias=bias,
        )
        self.film = ActivationDependentFiLM1d(
            out_features,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )

    @property
    def coords(self):
        return self.key_encoder.coords

    def set_coordinates(self, coords):
        self.key_encoder.set_coordinates(coords)

    def forward(self, x):
        key_vec = self.key_encoder()
        feat = self.linear(x, key_vec)
        return self.film(feat, key_vec)

    def forward_without_inr(self, x):
        return self.linear.forward_without_key(x)

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
            new_layer.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.linear.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_inr(child, inr_input_dim, inr_hidden_size, inr_output_dim, inr_layer_num, inr_output_activation, inr_output_bias, device)

class SpaMosaic(object):
    def __init__(
        self, 
        modBatch_dict={},  # dict={'rna':[batch1, None, ...], 'adt':[batch1, batch2, ...], ...}
        input_key='dimred_bc',
        batch_key='batch', 
        radius_cutoff=2000, intra_knn=10, inter_knn=10, w_g=0.8, 
        log_dir=None,
        seed=1234, num_workers=6,
        device='cuda:0'
    ):  
        self.device = torch.device(device) if ('cuda' in device) and torch.cuda.is_available() else torch.device('cpu')
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        set_seeds(seed)
        self.radius_cutoff = radius_cutoff
        self.w_g = w_g
        self.intra_knn = intra_knn
        self.inter_knn = inter_knn
        self.seed = seed
        self.num_workers = num_workers

        self.input_key = input_key
        self.mod_list = np.array(list(modBatch_dict.keys()))
        self.n_mods = len(self.mod_list)
        self.n_batches = len(modBatch_dict[self.mod_list[0]])
        self.batch_key = batch_key
        self.barc2batch = utls.get_barc2batch(modBatch_dict)

        # check if there is empty batch
        self.batch_contained_mod_ids = utls.check_batch_empty(modBatch_dict)

        # check if this dataset can be integrated
        self.check_integrity()

        # prepare spot-spot for each modality
        self.prepare_graphs(modBatch_dict)

    def check_integrity(self):
        mod_graph = np.zeros((self.n_mods, self.n_mods))
        for bi in range(self.n_batches):
            modIds_in_bi = self.batch_contained_mod_ids[bi]
            mod_pairs = np.array(list(itertools.product(modIds_in_bi, modIds_in_bi)))
            mod_graph[mod_pairs[:, 0], mod_pairs[:, 1]] = 1
        n_cs, labels = connected_components(mod_graph, directed=False, return_labels=True)
        if n_cs > 1:
            for ci in np.unique(labels):
                ni_msk = labels == ci
                # print(f'conn {ci}:', self.mod_list[ni_msk])
            raise RuntimeError('Dataset not connected, cannot be integrated')

    def prepare_graphs(self, modBatch_dict):
        # determine which batches as bridges, which not. 
        mod_mask = np.zeros((self.n_mods, self.n_batches))
        for bi in range(self.n_batches):
            for ki,k in enumerate(self.mod_list):
                if modBatch_dict[k][bi] is not None:
                    mod_mask[ki][bi] = 1

        self.bridge_batch_num_ids = np.where(mod_mask.sum(axis=0) >= 2)[0]
        self.non_bridge_batch_num_ids = np.where(mod_mask.sum(axis=0) < 2)[0]

        ### prepare meta parameter for training
        self.mod_batch_split = {
            key: [modBatch_dict[key][bi].n_obs if modBatch_dict[key][bi] is not None else 0 for bi in range(self.n_batches)]
            for key in self.mod_list
        }

        # mapping batch id to their modality set
        # e.g., 1 -> ['rna', 'adt'], 2 -> ['atac']
        self.bridge_batch_num_ids2mod = {
            bi: [key for key in self.mod_list if modBatch_dict[key][bi] is not None]
            for bi in self.bridge_batch_num_ids
        }
        self.non_bridge_batch_num_ids2mod = {
            bi: [key for key in self.mod_list if modBatch_dict[key][bi] is not None]
            for bi in self.non_bridge_batch_num_ids
        }

        bridge_ads = {
            key: [modBatch_dict[key][aid] for aid in self.bridge_batch_num_ids if modBatch_dict[key][aid] is not None]
            for key in self.mod_list
        }
        test_ads = {
            key: [modBatch_dict[key][aid] for aid in self.non_bridge_batch_num_ids if modBatch_dict[key][aid] is not None]
            for key in self.mod_list
        }

        # spatial-neighbor graph
        for key in self.mod_list:
            ads = [ad for ad in modBatch_dict[key] if ad is not None]
            build_graph.build_intra_graph(ads, rad_cutoff=self.radius_cutoff, knn=self.intra_knn)
        
        # expression-neighbor graph
        mod_mnn_set = {}
        for key in self.mod_list:
            mod_mnn_set[key] = build_graph.build_mnn_graph(bridge_ads[key], test_ads[key], self.input_key, self.inter_knn, self.seed)
            # print('Number of mnn pairs for {}:{}'.format(key, len(mod_mnn_set[key])))

        ### prepare inputs
        mod_graphs, mod_input_ads, mod_graph_idx2barc = {}, {}, {}
        for k in self.mod_list:
            mod_input_ads[k], mod_graph_idx2barc[k] = build_graph.mergeGraph([ad for ad in modBatch_dict[k] if ad is not None], 
                                                                            mod_mnn_set[k], inter_weight=self.w_g, use_rep=self.input_key)
        
        mod_graphs_edge_arr = {k:mod_input_ads[k].uns['edgeList'] for k in self.mod_list}
        mod_graphs['edge'] = {
            k:torch.LongTensor(np.array([mod_graphs_edge_arr[k][0], mod_graphs_edge_arr[k][1]])).to(self.device) 
            for k in self.mod_list
        }
        mod_graphs['edgeW'] = {k:torch.FloatTensor(mod_input_ads[k].uns['edgeW']).to(self.device) for k in self.mod_list}  # edge weights
        mod_graphs['edgeC'] = {}
        for k in self.mod_list:
            barc1 = utls.dict_map(mod_graph_idx2barc[k], mod_graphs_edge_arr[k][0])
            barc2 = utls.dict_map(mod_graph_idx2barc[k], mod_graphs_edge_arr[k][1])
            b1, b2 = np.array(utls.dict_map(self.barc2batch, barc1)), np.array(utls.dict_map(self.barc2batch, barc2))
            mod_graphs['edgeC'][k] = torch.LongTensor(np.where(b1 == b2, b1, -1)).to(self.device)

        mod_graphs['attribute'] = {
            k:torch.FloatTensor(mod_input_ads[k].obsm[self.input_key]).to(self.device)
            for k in self.mod_list
        }

        mod_graphs['attribute_dim'] = {k:mod_graphs['attribute'][k].shape[1] for k in self.mod_list}

        self.mod_graphs = mod_graphs

    def prepare_net(self, net, inr=False, inr_hidden_size=16):
        # config = utls.load_config(f'{cur_dir}/configs/{net}.yaml')
        config_path = pkg_resources.resource_filename('src', f'configs/{net}.yaml')
        config = utls.load_config(config_path)

        mod_model = {}
        for k in self.mod_list:
            encoder = archs.wlgcn.WLGCN(self.mod_graphs['attribute_dim'][k], config.model.out_dim, K=config.model.n_layer, dec_l=config.model.n_dec_l, 
                                        hidden_size=config.model.hid_dim, dropout=config.model.dropout, slope=config.model.slope)
            if inr:
                replace_linear_with_inr(encoder, inr_input_dim=16, inr_hidden_size=inr_hidden_size, inr_output_dim=1, inr_layer_num=3, inr_output_activation=None, inr_output_bias=0., device=self.device)
            mod_model[k] = encoder.to(self.device)

        return mod_model

    def train(self, net, lr, use_mini_thr=8000, mini_batch_size=1024, loss_type='adapted', T=0.01, bias=0, n_epochs=100, w_rec_g=0.):
        # set model architectures
        mod_model = net
        # set optimizer
        mod_optims = {k:torch.optim.Adam(mod_model[k].parameters(), lr=lr, weight_decay=5e-4) for k in self.mod_list}

        # for each batch, the meaning of parameter:
        # bridge_batch_meta_numbers = [{if as bridge}, {number of spots}, {size of mini-batches}, {number of measured modalites}, {measured modalities}]
        # test_batch_meta_numbers   = [{if as bridge}, {measured modalities}]
        batch_train_meta_numbers = {}
        for bi, ms in self.bridge_batch_num_ids2mod.items():
            n_cell = self.mod_batch_split[ms[0]][bi]
            n_loss_batch = n_cell if n_cell <= use_mini_thr else mini_batch_size  # determine mini-batch size for each batch
            n_batch = n_cell // n_loss_batch                                      # size of mini-batches
            batch_train_meta_numbers[bi] = (True, n_cell, n_loss_batch, n_batch, len(ms), ms)
        for bi, ms in self.non_bridge_batch_num_ids2mod.items():
            batch_train_meta_numbers[bi] = (False, ms)
        # print(batch_train_meta_numbers)

        # detetermine contrastive learning loss 
        if loss_type=='adapted':  # use our proposed contrastive learning loss, which can handle alignment of three or more modalities
            crit1 = {
                bi:CL_loss(batch_train_meta_numbers[bi][2], rep=batch_train_meta_numbers[bi][4], bias=bias).to(self.device)
                for bi in self.bridge_batch_num_ids
            }
        else:  
            crit1 = nn.CrossEntropyLoss().to(self.device)  # canonical implementation for CL loss, can only handle 2-modal alignment

        # feature reconstruction loss
        crit2 = nn.MSELoss().to(self.device)  

        # mod_input_feat, mod_graphs_edge, mod_graphs_edge_w
        mod_model, loss_cl, loss_rec, loss = train_model(
            mod_model, mod_optims, crit1, crit2, loss_type, w_rec_g,
            self.mod_graphs['attribute'], self.mod_graphs['edge'], self.mod_graphs['edgeW'], self.mod_graphs['edgeC'],
            batch_train_meta_numbers, self.mod_batch_split,
            T, n_epochs, self.device
        )

        self.mod_model = mod_model
        self.loss_cl  = loss_cl
        self.loss_rec = loss_rec
        self.loss = loss
        print(f"Loss: {loss}")
        # print(f"Loss_cl: {loss_cl}")
        # print(f"Loss_rec: {loss_rec}")
        return mod_model, loss_cl, loss_rec, loss
    
    

    def infer_emb(self, modBatch_dict, emb_key='emb', final_latent_key='merged_emb', cat=False): 
        for k in self.mod_list:
            self.mod_model[k].eval()
            z, _ = self.mod_model[k](self.mod_graphs['attribute'][k], self.mod_graphs['edge'][k], self.mod_graphs['edgeW'][k])
            z_split = torch.split(z, self.mod_batch_split[k])
            for bi in range(self.n_batches):
                if modBatch_dict[k][bi] is not None:
                    modBatch_dict[k][bi].obsm[emb_key] = z_split[bi].detach().cpu().numpy()

        # merge embs from measured modality
        ad_finals = []
        for bi in range(self.n_batches):
            embs = []
            for m in self.mod_list:
                if modBatch_dict[m][bi] is not None:
                    embs.append(modBatch_dict[m][bi].obsm[emb_key])
                    ad_tmp = modBatch_dict[m][bi]
            if cat:
                emb = np.hstack(emb)
            else:
                emb = np.mean(embs, axis=0)
            ad = sc.AnnData(np.zeros((emb.shape[0], 2)), obs=ad_tmp.obs.copy(), obsm={'spatial':ad_tmp.obsm['spatial']})
            ad.obsm[final_latent_key] = emb
            ad_finals.append(ad)
        return ad_finals

    def impute(self, modBatch_dict, emb_key='emb', layer_key='counts', imp_knn=10):
        aligned_pool = {
            k: np.vstack([ad.obsm[emb_key] for ad in modBatch_dict[k] if ad is not None])
            for k in modBatch_dict.keys()
        }

        target_pool = {
            k: np.vstack([ad.layers[layer_key].A if sps.issparse(ad.layers[layer_key]) else ad.layers[layer_key]
                        for ad in modBatch_dict[k] if ad is not None])
            for k in modBatch_dict.keys()
        }
        
        imputed_batchDict = {
            k: [None]*self.n_batches
            for k in modBatch_dict.keys()
        }
        for bi in range(self.n_batches):
            bi_measued_mod_names = [_ for _ in modBatch_dict.keys() if modBatch_dict[_][bi] is not None]
            bi_missing_mod_names = list(set(modBatch_dict.keys()) - set(bi_measued_mod_names))
            for k_q in bi_missing_mod_names:
                # print(f'impute {k_q}-{layer_key} for batch-{bi+1}')
                # visit all measured mods to impute missing k 
                imps = []
                for k_v in bi_measued_mod_names:
                    knn_ind = utls.nn_approx(modBatch_dict[k_v][bi].obsm[emb_key], aligned_pool[k_q], knn=imp_knn)
                    p_q = target_pool[k_q][knn_ind.ravel()].reshape(*(knn_ind.shape), target_pool[k_q].shape[1])
                    imps.append(np.mean(p_q, axis=1))
                imp = np.mean(imps, axis=0)
                imputed_batchDict[k_q][bi] = imp

        return imputed_batchDict

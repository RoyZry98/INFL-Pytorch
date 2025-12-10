# pipeline_utils.py
import os
import copy
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

from spamosaic.framework import SpaMosaic
from spamosaic.utils import nn_approx, plot_basis, clustering, split_adata_ob, get_umap
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess

# -----------------------------Horizontal----------------------------------- #
# =========================
# Setup Environment
# =========================
def setup_env(r_home=None, r_user=None):
    if r_home is not None:
        os.environ['R_HOME'] = r_home
    if r_user is not None:
        os.environ['R_USER'] = r_user
    # Set CUBLAS_WORKSPACE_CONFIG only when CuBLAS determinism is needed
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


# =========================
# Configuration
# =========================
@dataclass
class FLConfig:
    data_dir: str = '/deltadisk/zhangrongyu/SpaMosaic/data/integration/Human_tonsil'
    device: str = 'cuda:0'
    num_clients: int = 3
    num_rounds: int = 50
    local_epochs: int = 2
    learning_rate: float = 1e-3
    seed: int = 1234

    # Model/Graph Construction
    intra_knn: int = 2
    inter_knn: int = 2
    w_g: float = 0.8
    batch_key: str = 'src'
    input_key: str = 'dimred_bc'
    net: str = 'wlgcn'

    # Feature Selection/Preprocessing
    n_hvg: int = 5000
    favor: str = 'scanpy'
    batch_corr: bool = True

    # Algorithm Switch and Hyperparameters
    inr: bool = False
    inr_hidden_size: int = 8
    lora: bool = False  # If you need to pass to prepare_net, you can add parameters yourself
    dp: bool = False
    max_grad_norm: float = 1.0
    noise_std: float = 0.5
    dr: bool = False

    # Output Directory
    fig_dir: str = './figure'
    emb_dir: str = './tonsil_output_embeddings'
    log_dir: str = './tonsil_logs'
    run_name: Optional[str] = None  # If not empty, used to name file prefix


# =========================
# Utility Functions
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(module: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters of a nn.Module
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def get_param_report(global_model: SpaMosaic) -> Dict[str, Any]:
    """
    Return the number of parameters of each modality submodel in SpaMosaic,
    and try to identify/summarize the number of parameters and ratio of INR-related modules.
    Convention:
      - global_model.mod_model is a dict-like, key is modality, like 'rna'
      - Each submodel is a nn.Module, possibly containing submodules named 'inr', 'mlp_inr', 'implicit', etc.
    """
    report = {}
    total_params = 0
    total_inr_params = 0

    # You can expand the rules for identifying INR submodules here
    inr_names = {'inr', 'mlp_inr', 'implicit', 'sir', 'inr_net', 'implicit_mlp'}

    for mod_name in global_model.mod_list:
        submodel = global_model.mod_model[mod_name]
        mod_total = count_parameters(submodel)

        # Enumerate submodules, aggregate parameters of "potential INR"
        mod_inr = 0
        for name, child in submodel.named_modules():
            base = name.split('.')[-1]
            if base.lower() in inr_names:
                mod_inr += count_parameters(child)

        # If no hit above aliases, but prepare_net(inr=True) is only enabled for this modality,
        # You can also fall back to the heuristic of "layer name contains inr/implicit":
        if mod_inr == 0:
            for name, child in submodel.named_modules():
                lname = name.lower()
                if ('inr' in lname) or ('implicit' in lname):
                    mod_inr += count_parameters(child)

        report[mod_name] = {
            'total_params': mod_total,
            'inr_params': mod_inr,
            'inr_ratio': (mod_inr / mod_total) if mod_total > 0 else float('nan'),
        }
        total_params += mod_total
        total_inr_params += mod_inr

    report['__overall__'] = {
        'total_params': total_params,
        'inr_params': total_inr_params,
        'inr_ratio': (total_inr_params / total_params) if total_params > 0 else float('nan'),
    }
    return report

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def decentralized_randomization(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Randomizes the order of local model weights for decentralized randomization."""
    models = copy.deepcopy(models)
    for i in range(len(models) - 1):
        j = random.randint(i, len(models) - 1)
        models[i], models[j] = models[j], models[i]
    return models


def fed_avg(weight_list: List[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    weight_list: List of per-client weights
      each element: {module_name: state_dict}
    return: averaged weights with the same structure
    """
    assert len(weight_list) > 0, "No local weights to aggregate."
    w_avg = copy.deepcopy(weight_list[0])

    for module_name in w_avg.keys():
        for param_name in w_avg[module_name].keys():
            for i in range(1, len(weight_list)):
                w_avg[module_name][param_name] += weight_list[i][module_name][param_name]
            w_avg[module_name][param_name] = torch.div(w_avg[module_name][param_name], len(weight_list))
    return w_avg


# =========================
# Data Loading and Preprocessing
# =========================
def load_client_adatas(data_dir: str) -> Dict[str, sc.AnnData]:
    """
    Load three slices according to the current human tonsil data organization structure.
    You can also extend it to any number and path here.
    """
    ad1_rna = sc.read_h5ad(os.path.join(data_dir, 'slice1/s1_adata_rna.h5ad'))
    ad2_rna = sc.read_h5ad(os.path.join(data_dir, 'slice2/s2_adata_rna.h5ad'))
    ad3_rna = sc.read_h5ad(os.path.join(data_dir, 'slice3/s3_adata_rna.h5ad'))
    return {'s1': ad1_rna, 's2': ad2_rna, 's3': ad3_rna}


def preprocess_rna(adatas: List[sc.AnnData],
                   batch_corr: bool,
                   favor: str,
                   n_hvg: int,
                   batch_key: str,
                   input_key: str):
    RNA_preprocess(adatas, batch_corr=batch_corr, favor=favor, n_hvg=n_hvg,
                   batch_key=batch_key, key=input_key)


# =========================
# Model Preparation
# =========================
def build_global_model(input_dict: Dict[str, List[sc.AnnData]],
                       cfg: FLConfig) -> SpaMosaic:
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model = SpaMosaic(
        modBatch_dict=input_dict,
        input_key=cfg.input_key,
        batch_key=cfg.batch_key,
        intra_knn=cfg.intra_knn,
        inter_knn=cfg.inter_knn,
        w_g=cfg.w_g,
        seed=cfg.seed,
        device=str(device)
    )
    model.prepare_graphs(input_dict)
    model.mod_model = model.prepare_net(net=cfg.net, inr=cfg.inr, inr_hidden_size=cfg.inr_hidden_size)
    return model


def init_global_weights(model: SpaMosaic) -> Dict[str, Dict[str, torch.Tensor]]:
    return {k: copy.deepcopy(model.mod_model[k].state_dict()) for k in model.mod_list}


def build_local_model(client_data: List[sc.AnnData], cfg: FLConfig) -> SpaMosaic:
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    local_model = SpaMosaic(
        modBatch_dict={'rna': client_data},
        input_key=cfg.input_key,
        batch_key=cfg.batch_key,
        intra_knn=cfg.intra_knn,
        inter_knn=cfg.inter_knn,
        w_g=cfg.w_g,
        seed=cfg.seed,
        device=str(device)
    )
    local_model.prepare_graphs({'rna': client_data})
    local_model.mod_model = local_model.prepare_net(net=cfg.net, inr=cfg.inr, inr_hidden_size=cfg.inr_hidden_size)
    return local_model


# =========================
# Training and Federated Loop
# =========================
def one_client_train(client_data: List[sc.AnnData],
                     global_weights: Dict[str, Dict[str, torch.Tensor]],
                     cfg: FLConfig) -> Tuple[Dict[str, Dict[str, torch.Tensor]], float]:
    """
    Return: The weight dictionary of the client after training and the loss of the last epoch
    """
    local_model = build_local_model(client_data, cfg)

    # Load global weights
    for k in local_model.mod_list:
        local_model.mod_model[k].load_state_dict(global_weights[k])

    # Train
    if cfg.dp:
        local_model.train(net=copy.deepcopy(local_model.mod_model),
                          lr=cfg.learning_rate,
                          n_epochs=cfg.local_epochs,
                          w_rec_g=0.,
                          dp=True,
                          max_grad_norm=cfg.max_grad_norm,
                          noise_std=cfg.noise_std)
    else:
        local_model.train(net=copy.deepcopy(local_model.mod_model),
                          lr=cfg.learning_rate,
                          n_epochs=cfg.local_epochs,
                          w_rec_g=0.)

    trained_weights = {k: copy.deepcopy(local_model.mod_model[k].state_dict())
                       for k in local_model.mod_list}
    last_loss = float(local_model.loss_rec[-1]) if len(local_model.loss_rec) > 0 else np.nan
    return trained_weights, last_loss


def get_clients_data(ad_dict: Dict[str, sc.AnnData]) -> List[List[sc.AnnData]]:
    """
    According to the client division scheme in your original script:
    - client 1: [ad1, ad2]
    - client 2: [ad3, ad2]
    - client 3: [ad1, ad3]
    """
    ad1, ad2, ad3 = ad_dict['s1'], ad_dict['s2'], ad_dict['s3']
    return [
        [ad1, ad2],
        [ad3, ad2],
        [ad1, ad3],
    ]


def run_federated_spamosaic(cfg: FLConfig) -> Dict[str, Any]:
    """
    Complete federated training + best model weights recording + inference and saving.
    Return a result dictionary containing:
      - best_loss
      - best_weights
      - loss_history
      - merged_embeddings_path
      - loss_csv_path
      - loss_curve_path
      - config (dict)
    """
    set_seed(cfg.seed)
    ensure_dir(cfg.fig_dir)
    ensure_dir(cfg.emb_dir)
    ensure_dir(cfg.log_dir)

    # 1) Load data
    ad_map = load_client_adatas(cfg.data_dir)
    input_dict = {'rna': [ad_map['s1'], ad_map['s2'], ad_map['s3']]}

    # 2) Preprocess
    preprocess_rna(
        input_dict['rna'],
        batch_corr=cfg.batch_corr,
        favor=cfg.favor,
        n_hvg=cfg.n_hvg,
        batch_key=cfg.batch_key,
        input_key=cfg.input_key
    )

    # 3) Global model
    global_model = build_global_model(input_dict, cfg)
    global_weights = init_global_weights(global_model)

    print("-------------------------------- Global Model Parameter Report --------------------------------")
    param_report = get_param_report(global_model)
    print("Parameter report:")
    for k, v in param_report.items():
        if k == '__overall__':
            print(f"[Overall] total={v['total_params']:,}, inr={v['inr_params']:,}, ratio={v['inr_ratio']:.4f}")
        else:
            print(f"[{k}] total={v['total_params']:,}, inr={v['inr_params']:,}, ratio={v['inr_ratio']:.4f}")
            
    # 4) Federated loop
    best_loss = float('inf')
    best_weights = None
    loss_history: List[float] = []

    clients_data = get_clients_data(ad_map)

    for round_idx in range(cfg.num_rounds):
        print(f"--- Round {round_idx + 1}/{cfg.num_rounds} ---")
        local_weights: List[Dict[str, Dict[str, torch.Tensor]]] = []
        local_losses: List[float] = []

        for client_idx, client_data in enumerate(clients_data):
            print(f"Training on client {client_idx + 1}...")
            trained_w, last_loss = one_client_train(client_data, global_weights, cfg)
            local_weights.append(trained_w)
            local_losses.append(last_loss)

        # Decentralized randomization (optional)
        if cfg.dr:
            print("Applying Decentralized Randomization to local models...")
            local_weights = decentralized_randomization(local_weights)

        # Aggregate
        global_weights = fed_avg(local_weights)

        # Update global model
        for k in global_model.mod_list:
            global_model.mod_model[k].load_state_dict(global_weights[k])

        # Record and select the best
        avg_loss = float(np.mean(local_losses)) if len(local_losses) > 0 else np.nan
        print(f"Average loss for round {round_idx + 1}: {avg_loss:.6f}")
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = copy.deepcopy(global_weights)
            print(f"New best model found with loss: {best_loss:.6f}")

    # 5) Save training curve
    run_name = cfg.run_name or f"inr_{cfg.inr}_dp_{cfg.dp}_lora_{cfg.lora}_dr_{cfg.dr}"
    loss_curve_path = os.path.join(cfg.fig_dir, f"fed_loss_curve_{run_name}.png")
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Federated Learning Loss Curve')
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    # 6) Overwrite to global model and infer
    if best_weights is not None:
        for k in global_model.mod_list:
            global_model.mod_model[k].load_state_dict(best_weights[k])
        print(f"Loaded best model with loss: {best_loss:.6f}")

    ad_embs = global_model.infer_emb({'rna': [ad_map['s1'], ad_map['s2'], ad_map['s3']]},
                                     emb_key='emb',
                                     final_latent_key='merged_emb')
    ad_mosaic = sc.concat(ad_embs)

    emb_path = os.path.join(cfg.emb_dir, f"tonsil_merged_embeddings_{run_name}.h5ad")
    ensure_dir(os.path.dirname(emb_path))
    ad_mosaic.write_h5ad(emb_path)
    print(f"Saved merged embeddings to {emb_path}")

    # 7) Save training log
    loss_csv_path = os.path.join(cfg.log_dir, f"tonsil_loss_history_{run_name}.csv")
    ensure_dir(os.path.dirname(loss_csv_path))
    loss_history_df = pd.DataFrame({'round': list(range(1, len(loss_history) + 1)), 'loss': loss_history})
    loss_history_df.to_csv(loss_csv_path, index=False)
    print(f"Saved loss history to {loss_csv_path}")

    return {
        'best_loss': best_loss,
        'best_weights': best_weights,
        'loss_history': loss_history,
        'merged_embeddings_path': emb_path,
        'loss_csv_path': loss_csv_path,
        'loss_curve_path': loss_curve_path,
        'config': asdict(cfg),
    }


# =========================
# Convenient entry (for Notebook usage)
# =========================
def run_and_save(
    data_dir: str = '/deltadisk/zhangrongyu/SpaMosaic/data/integration/Human_tonsil',
    device: str = 'cuda:0',
    num_rounds: int = 50,
    local_epochs: int = 2,
    learning_rate: float = 1e-3,
    inr: bool = False,
    inr_hidden_size: int = 8,
    dp: bool = False,
    max_grad_norm: float = 1.0,
    noise_std: float = 0.5,
    dr: bool = False,
    lora: bool = False,
    run_name: Optional[str] = None,
    seed: int = 1234,
) -> Dict[str, Any]:
    cfg = FLConfig(
        data_dir=data_dir,
        device=device,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        inr=inr,
        inr_hidden_size=inr_hidden_size,
        dp=dp,
        max_grad_norm=max_grad_norm,
        noise_std=noise_std,
        dr=dr,
        lora=lora,
        run_name=run_name,
        seed=seed,
    )
    return run_federated_spamosaic(cfg)






# -----------------------------Mosaic----------------------------------- #
# =========================
# Parameter statistics tool (generic)
# =========================
def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())

def get_inr_param_report(spamosaic_model: SpaMosaic) -> Dict[str, Any]:
    """
    Count the number of trainable parameters of each modality submodel, and the number and ratio of INR parameters identified by Heuristic.
    约定：
      - spamosaic_model.mod_model: dict-like {mod_name: nn.Module}
      - Identify INR by submodule name containing 'inr' or 'implicit' (you can also customize it according to the actual naming)
    """
    report: Dict[str, Any] = {}
    total_params, total_inr = 0, 0
    inr_alias = {'inr', 'mlp_inr', 'implicit', 'sir', 'inr_net', 'implicit_mlp'}

    for mod_name in spamosaic_model.mod_list:
        submodel = spamosaic_model.mod_model[mod_name]
        mod_total = count_parameters(submodel, trainable_only=True)

        mod_inr = 0
        # First match precisely by aliases
        for name, child in submodel.named_modules():
            base = name.split('.')[-1].lower()
            if base in inr_alias:
                mod_inr += count_parameters(child, trainable_only=True)
        # If not hit, use inclusion relationship as fallback
        if mod_inr == 0:
            for name, child in submodel.named_modules():
                lname = name.lower()
                if ('inr' in lname) or ('implicit' in lname):
                    mod_inr += count_parameters(child, trainable_only=True)

        report[mod_name] = {
            'total_params': int(mod_total),
            'inr_params': int(mod_inr),
            'inr_ratio': (float(mod_inr) / float(mod_total)) if mod_total > 0 else float('nan'),
        }
        total_params += mod_total
        total_inr += mod_inr

    report['__overall__'] = {
        'total_params': int(total_params),
        'inr_params': int(total_inr),
        'inr_ratio': (float(total_inr) / float(total_params)) if total_params > 0 else float('nan'),
    }
    return report


# =========================
# Multi-omics (RNA+ADT) federated training configuration
# =========================
@dataclass
class FLConfigMM:
    # Data and device
    data_dir: str = '/deltadisk/zhangrongyu/SpaMosaic/data/integration/Human_tonsil'
    device: str = 'cuda:0'
    seed: int = 1234

    # Clients and training
    num_clients: int = 3
    num_rounds: int = 50
    local_epochs: int = 2
    learning_rate: float = 1e-3

    # Graph and model
    intra_knn: int = 2
    inter_knn: int = 2
    w_g: float = 0.8
    batch_key: str = 'src'
    input_key: str = 'dimred_bc'
    net: str = 'wlgcn'

    # Preprocessing
    n_hvg: int = 5000
    favor: str = 'scanpy'
    batch_corr: bool = True

    # Algorithm switch
    inr: bool = False
    inr_hidden_size: int = 8
    lora: bool = False
    dp: bool = False
    max_grad_norm: float = 1.0
    noise_std: float = 0.5
    dr: bool = False

    # Output
    fig_dir: str = './figure'
    emb_dir: str = './tonsil_output_embeddings'
    log_dir: str = './tonsil_logs'
    run_name: Optional[str] = None  # File name suffix; if empty, generate by switch combination


# =========================
# Multi-omics data loading and preprocessing
# =========================
def load_mm_adatas(data_dir: str) -> Dict[str, Dict[str, sc.AnnData]]:
    """
    Return:
    {
      'rna': {'s1': ad1_rna, 's2': ad2_rna, 's3': ad3_rna},
      'adt': {'s1': ad1_adt, 's2': ad2_adt, 's3': ad3_adt}
    }
    """
    ad1_rna = sc.read_h5ad(os.path.join(data_dir, 'slice1/s1_adata_rna.h5ad'))
    ad2_rna = sc.read_h5ad(os.path.join(data_dir, 'slice2/s2_adata_rna.h5ad'))
    ad3_rna = sc.read_h5ad(os.path.join(data_dir, 'slice3/s3_adata_rna.h5ad'))

    ad1_adt = sc.read_h5ad(os.path.join(data_dir, 'slice1/s1_adata_adt.h5ad'))
    ad2_adt = sc.read_h5ad(os.path.join(data_dir, 'slice2/s2_adata_adt.h5ad'))
    ad3_adt = sc.read_h5ad(os.path.join(data_dir, 'slice3/s3_adata_adt.h5ad'))

    return {
        'rna': {'s1': ad1_rna, 's2': ad2_rna, 's3': ad3_rna},
        'adt': {'s1': ad1_adt, 's2': ad2_adt, 's3': ad3_adt},
    }

def preprocess_mm(input_dict: Dict[str, List[sc.AnnData]], cfg: FLConfigMM):
    RNA_preprocess(input_dict['rna'], batch_corr=cfg.batch_corr, favor=cfg.favor,
                   n_hvg=cfg.n_hvg, batch_key=cfg.batch_key, key=cfg.input_key)
    ADT_preprocess(input_dict['adt'], batch_corr=cfg.batch_corr,
                   batch_key=cfg.batch_key, key=cfg.input_key)


# =========================
# Multi-omics model building
# =========================
def build_global_model_mm(input_dict: Dict[str, List[sc.AnnData]], cfg: FLConfigMM) -> SpaMosaic:
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model = SpaMosaic(
        modBatch_dict=input_dict,
        input_key=cfg.input_key,
        batch_key=cfg.batch_key,
        intra_knn=cfg.intra_knn,
        inter_knn=cfg.inter_knn,
        w_g=cfg.w_g,
        seed=cfg.seed,
        device=str(device)
    )
    model.prepare_graphs(input_dict)
    model.mod_model = model.prepare_net(net=cfg.net, inr=cfg.inr, inr_hidden_size=cfg.inr_hidden_size)
    return model

def init_global_weights_mm(model: SpaMosaic) -> Dict[str, Dict[str, torch.Tensor]]:
    return {k: copy.deepcopy(model.mod_model[k].state_dict()) for k in model.mod_list}


def build_local_model_mm(client_input_dict: Dict[str, List[Optional[sc.AnnData]]], cfg: FLConfigMM) -> SpaMosaic:
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    local_model = SpaMosaic(
        modBatch_dict=client_input_dict,
        input_key=cfg.input_key,
        batch_key=cfg.batch_key,
        intra_knn=cfg.intra_knn,
        inter_knn=cfg.inter_knn,
        w_g=cfg.w_g,
        seed=cfg.seed,
        device=str(device)
    )
    local_model.prepare_graphs(client_input_dict)
    local_model.mod_model = local_model.prepare_net(net=cfg.net, inr=cfg.inr, inr_hidden_size=cfg.inr_hidden_size)
    return local_model


# =========================
# Client division (follow your script logic)
# =========================
def get_clients_mm(ad_rna: Dict[str, sc.AnnData], ad_adt: Dict[str, sc.AnnData]) -> List[Dict[str, List[Optional[sc.AnnData]]]]:
    """
    与你脚本保持一致：
      - Client 1: rna=[s1, s2, None], adt=[s1, None, s3]
      - Client 2: rna=[s1, s2, None], adt=[None, s2, s3]
      - Client 3: rna=[None, s2, s3], adt=[s1, None, s3]
    """
    s1r, s2r, s3r = ad_rna['s1'], ad_rna['s2'], ad_rna['s3']
    s1a, s2a, s3a = ad_adt['s1'], ad_adt['s2'], ad_adt['s3']

    return [
        {'rna': [s1r, s2r, None], 'adt': [s1a, None, s3a]},
        {'rna': [s1r, s2r, None], 'adt': [None, s2a, s3a]},
        {'rna': [None, s2r, s3r], 'adt': [s1a, None, s3a]},
    ]


# =========================
# Federated averaging and randomization
# =========================
def fed_avg_mm(weight_list: List[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
    assert len(weight_list) > 0, "No local weights to aggregate."
    w_avg = copy.deepcopy(weight_list[0])
    for module_name in w_avg.keys():
        for param_name in w_avg[module_name].keys():
            for i in range(1, len(weight_list)):
                w_avg[module_name][param_name] += weight_list[i][module_name][param_name]
            w_avg[module_name][param_name] = torch.div(w_avg[module_name][param_name], len(weight_list))
    return w_avg

def decentralized_randomization_mm(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    models = copy.deepcopy(models)
    for i in range(len(models) - 1):
        j = random.randint(i, len(models) - 1)
        models[i], models[j] = models[j], models[i]
    return models


# =========================
# Multi-omics federated training main流程
# =========================
def run_federated_spamosaic_mm(cfg: FLConfigMM) -> Dict[str, Any]:
    set_seed(cfg.seed)
    ensure_dir(cfg.fig_dir)
    ensure_dir(cfg.emb_dir)
    ensure_dir(cfg.log_dir)

    # Optional environment variable setting (when consistent with original script, can be set outside)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    # 1) Load and preprocess
    ad_map = load_mm_adatas(cfg.data_dir)
    input_dict = {
        'rna': [ad_map['rna']['s1'], ad_map['rna']['s2'], ad_map['rna']['s3']],
        'adt': [ad_map['adt']['s1'], ad_map['adt']['s2'], ad_map['adt']['s3']],
    }
    preprocess_mm(input_dict, cfg)

    # 2) Global model
    global_model = build_global_model_mm(input_dict, cfg)
    global_weights = init_global_weights_mm(global_model)

    # 3) Federated loop
    best_loss = float('inf')
    best_weights = None
    loss_history: List[float] = []

    clients = get_clients_mm(ad_map['rna'], ad_map['adt'])

    for round_idx in range(cfg.num_rounds):
        print(f"--- Round {round_idx + 1}/{cfg.num_rounds} ---")
        local_weights: List[Dict[str, Dict[str, torch.Tensor]]] = []
        local_losses: List[float] = []

        for client_idx, client_input in enumerate(clients):
            print(f"Training on client {client_idx + 1}...")
            local_model = build_local_model_mm(client_input, cfg)

            # Sync weights
            for k in local_model.mod_list:
                local_model.mod_model[k].load_state_dict(global_weights[k])

            # Train (keep consistent with your script: use local_model.loss as the last loss)
            if cfg.dp:
                local_model.train(net=copy.deepcopy(local_model.mod_model),
                                  lr=cfg.learning_rate, T=0.01,
                                  n_epochs=cfg.local_epochs, w_rec_g=0.,
                                  dp=True, max_grad_norm=cfg.max_grad_norm, noise_std=cfg.noise_std)
            else:
                local_model.train(net=copy.deepcopy(local_model.mod_model),
                                  lr=cfg.learning_rate, T=0.01,
                                  n_epochs=cfg.local_epochs, w_rec_g=0.)

            local_weights.append({k: copy.deepcopy(local_model.mod_model[k].state_dict())
                                  for k in local_model.mod_list})
            # Your script uses local_model.loss (not loss_rec[-1])
            last_loss = float(local_model.loss) if hasattr(local_model, 'loss') else \
                        (float(local_model.loss_rec[-1]) if hasattr(local_model, 'loss_rec') and len(local_model.loss_rec) > 0 else np.nan)
            local_losses.append(last_loss)

        if cfg.dr:
            print("Applying Decentralized Randomization to local models...")
            local_weights = decentralized_randomization_mm(local_weights)

        global_weights = fed_avg_mm(local_weights)

        for k in global_model.mod_list:
            global_model.mod_model[k].load_state_dict(global_weights[k])

        avg_loss = float(np.mean(local_losses)) if len(local_losses) > 0 else np.nan
        print(f"Average loss for round {round_idx + 1}: {avg_loss}")
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = copy.deepcopy(global_weights)
            print(f"New best model found with loss: {best_loss}")

    # 4) Save curve
    run_name = cfg.run_name or f"inr_{cfg.inr}_dp_{cfg.dp}_lora_{cfg.lora}_dr_{cfg.dr}"
    loss_curve_path = os.path.join(cfg.fig_dir, f"fed_loss_curve_{run_name}.png")
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Federated Learning Loss Curve (RNA+ADT)')
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    # 5) Overwrite to best weights and infer
    if best_weights is not None:
        for k in global_model.mod_list:
            global_model.mod_model[k].load_state_dict(best_weights[k])
        print(f"Loaded best model with loss: {best_loss}")

    ad_embs = global_model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
    ad_mosaic = sc.concat(ad_embs)
    emb_path = os.path.join(cfg.emb_dir, f"merged_embeddings_{run_name}.h5ad")
    ensure_dir(os.path.dirname(emb_path))
    ad_mosaic.write_h5ad(emb_path)
    print(f"Saved merged embeddings to {emb_path}")

    # 6) Save loss history
    loss_csv_path = os.path.join(cfg.log_dir, f"loss_history_{run_name}.csv")
    ensure_dir(os.path.dirname(loss_csv_path))
    loss_history_df = pd.DataFrame({'round': list(range(1, len(loss_history) + 1)), 'loss': loss_history})
    loss_history_df.to_csv(loss_csv_path, index=False)
    print(f"Saved loss history to {loss_csv_path}")

    # 7) Statistics and save parameter report
    param_report = get_inr_param_report(global_model)
    import json
    param_json_path = os.path.join(cfg.log_dir, f"param_report_{run_name}.json")
    with open(param_json_path, 'w') as f:
        json.dump(param_report, f, indent=2)
    print(f"Saved parameter report to {param_json_path}")

    return {
        'best_loss': best_loss,
        'best_weights': best_weights,  # Note: the dictionary contains tensors, which is generally not recommended to persist long-term
        'loss_history': loss_history,
        'merged_embeddings_path': emb_path,
        'loss_csv_path': loss_csv_path,
        'loss_curve_path': loss_curve_path,
        'param_report_path': param_json_path,
        'config': asdict(cfg),
    }


# =========================
# Convenient entry (multi-omics, for Notebook usage)
# =========================
def run_and_save_mm(
    data_dir: str = '/deltadisk/zhangrongyu/SpaMosaic/data/integration/Human_tonsil',
    device: str = 'cuda:0',
    num_rounds: int = 50,
    local_epochs: int = 2,
    learning_rate: float = 1e-3,
    inr: bool = False,
    inr_hidden_size: int = 8,
    dp: bool = False,
    max_grad_norm: float = 1.0,
    noise_std: float = 0.5,
    dr: bool = False,
    lora: bool = False,
    run_name: Optional[str] = None,
    seed: int = 1234,
) -> Dict[str, Any]:
    cfg = FLConfigMM(
        data_dir=data_dir,
        device=device,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        inr=inr,
        inr_hidden_size=inr_hidden_size,
        dp=dp,
        max_grad_norm=max_grad_norm,
        noise_std=noise_std,
        dr=dr,
        lora=lora,
        run_name=run_name,
        seed=seed,
    )
    return run_federated_spamosaic_mm(cfg)
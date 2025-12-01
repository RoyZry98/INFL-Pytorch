import os
import scanpy as sc
from os.path import join
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse  # 用于解析命令行参数
import pandas as pd  # 确保已导入 pandas
import random
import sys
sys.path.insert(0, '../../..')

from spamosaic.framework import SpaMosaic
from spamosaic.utils import nn_approx, plot_basis, clustering, split_adata_ob, get_umap
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess

os.environ['R_HOME'] = '/deltadisk/miniconda3/envs/SpaMosaic/lib/R'
os.environ['R_USER'] = '/deltadisk/miniconda3/envs/SpaMosaic/lib/python3.8/site-packages/rpy2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for CuBLAS operation and you have CUDA >= 10.2

# Decentralized Randomization for Models
def decentralized_randomization(models):
    """Randomizes the order of local model weights for decentralized randomization."""
    for i in range(len(models) - 1):
        j = random.randint(i, len(models) - 1)  # Random index from i to len(models)-1
        models[i], models[j] = models[j], models[i]  # Swap models
    return models

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for param_key in w_avg[k].keys():  # 遍历每个参数
            for i in range(1, len(w)):
                w_avg[k][param_key] += w[i][k][param_key]  # 对张量求和
            w_avg[k][param_key] = torch.div(w_avg[k][param_key], len(w))  # 对张量求平均
    return w_avg

def main(args):
    # Define global settings
    data_dir = '/deltadisk/zhangrongyu/SpaMosaic/data/integration/Human_tonsil'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_clients = 3
    num_rounds = 50
    local_epochs = 2
    learning_rate = 0.001

    # Load client datasets
    ad1_rna = sc.read_h5ad(join(data_dir, 'slice1/s1_adata_rna.h5ad'))
    ad1_adt = sc.read_h5ad(join(data_dir, 'slice1/s1_adata_adt.h5ad'))
    ad2_rna = sc.read_h5ad(join(data_dir, 'slice2/s2_adata_rna.h5ad'))
    ad2_adt = sc.read_h5ad(join(data_dir, 'slice2/s2_adata_adt.h5ad'))
    ad3_adt = sc.read_h5ad(join(data_dir, 'slice3/s3_adata_adt.h5ad'))
    ad3_rna = sc.read_h5ad(join(data_dir, 'slice3/s3_adata_rna.h5ad'))

    input_dict = {
        'rna': [ad1_rna, ad2_rna, ad3_rna],
        'adt': [ad1_adt, ad2_adt, ad3_adt]
    }

    input_key = 'dimred_bc'

    # Preprocess data
    RNA_preprocess(input_dict['rna'], batch_corr=True, favor='scanpy', n_hvg=5000, batch_key='src', key=input_key)
    ADT_preprocess(input_dict['adt'], batch_corr=True, batch_key='src', key=input_key)
    
    # Initialize global SpaMosaic model
    global_model = SpaMosaic(
        modBatch_dict=input_dict,
        input_key=input_key,
        batch_key='src',
        intra_knn=2,
        inter_knn=2,
        w_g=0.8,
        seed=1234,
        device='cuda:0'
    )

    # Prepare graphs and network (ensure correct initialization order)
    global_model.prepare_graphs(input_dict)

    # Call prepare_net and manually assign the returned mod_model to global_model
    global_model.mod_model = global_model.prepare_net(net='wlgcn', inr=args.inr, inr_hidden_size=args.inr_hidden_size)  # Assign mod_model here

    # Initialize global weights
    global_weights = {k: copy.deepcopy(global_model.mod_model[k].state_dict()) for k in global_model.mod_list}

    # Initialize variables to track the best model
    best_loss = float('inf')  # 初始化为正无穷
    best_weights = None       # 保存最低 loss 对应的 global_weights

    # Training
    loss_history = []
    for round_num in range(num_rounds):
        print(f"--- Round {round_num + 1} ---")
        local_weights = []
        local_losses = []

        # Distribute global weights to clients and train locally
        for client_idx in range(num_clients):
            print(f"Training on client {client_idx + 1}...")

            # Define data for each client based on your requirements
            if client_idx == 0:  # Client 1
                client_input_dict = {
                    'rna': [ad1_rna, ad2_rna, None],
                    'adt': [ad1_adt, None, ad3_adt]
                }
            elif client_idx == 1:  # Client 2
                client_input_dict = {
                    'rna': [ad1_rna, ad2_rna, None],
                    'adt': [None, ad2_adt, ad3_adt]
                }
            elif client_idx == 2:  # Client 3
                client_input_dict = {
                    'rna': [None, ad2_rna, ad3_rna],
                    'adt': [ad1_adt, None, ad3_adt]
                }

            # Create a local SpaMosaic instance for the client
            local_model = SpaMosaic(
                modBatch_dict=client_input_dict,  # Single client's data
                input_key=input_key,
                batch_key='src', intra_knn=2, inter_knn=2, w_g=0.8, 
                seed=1234, device='cuda:0'
            )
            local_model.prepare_graphs(client_input_dict)  # Prepare graphs for the client
            local_model.mod_model = local_model.prepare_net(net='wlgcn', inr=args.inr, inr_hidden_size=args.inr_hidden_size)  # Assign mod_model here

            # Load global weights into local model
            for k in local_model.mod_list:
                local_model.mod_model[k].load_state_dict(global_weights[k])

            # Train the local model
            if args.dp:
                local_model.train(net=copy.deepcopy(local_model.mod_model), lr=learning_rate, T=0.01, n_epochs=local_epochs, w_rec_g=0., dp=True, max_grad_norm=args.max_grad_norm, noise_std=args.noise_std)
            else:
                local_model.train(net=copy.deepcopy(local_model.mod_model), lr=learning_rate, T=0.01, n_epochs=local_epochs, w_rec_g=0.)
            local_weights.append({k: copy.deepcopy(local_model.mod_model[k].state_dict()) for k in local_model.mod_list})
            local_losses.append(local_model.loss)  # Record the last loss

        # Apply Decentralized Randomization to local models (if enabled)
        if args.dr:
            print("Applying Decentralized Randomization to local models...")
            local_weights = decentralized_randomization(local_weights)

        # Federated averaging
        global_weights = FedAvg(local_weights)

        # Update global model with aggregated weights
        for k in global_model.mod_list:
            global_model.mod_model[k].load_state_dict(global_weights[k])

        # Log average loss
        avg_loss = np.mean(local_losses)
        print(f"Average loss for round {round_num + 1}: {avg_loss}")
        loss_history.append(avg_loss)

        # Check if current loss is the best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = copy.deepcopy(global_weights)  # Save the best weights
            print(f"New best model found with loss: {best_loss}")

    # Plot loss curve
    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Federated Learning Loss Curve')
    plt.savefig(f'./figure/fed_loss_curve_{args.inr}.png')
    
    # Load the best weights into the global model
    if best_weights is not None:
        for k in global_model.mod_list:
            global_model.mod_model[k].load_state_dict(best_weights[k])
        print(f"Loaded best model with loss: {best_loss}")

    ad_embs = global_model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
    ad_mosaic = sc.concat(ad_embs)
    # 保存合并后的 AnnData 对象
    ad_mosaic.write_h5ad(f'./tonsil_output_embeddings/merged_embeddings_inr_{args.inr}_dp_{args.dp}_lora_{args.lora}_dr_{args.dr}.h5ad')
    print(f"Saved merged embeddings to ./tonsil_output_embeddings/merged_embeddings_inr_{args.inr}_dp_{args.dp}_lora_{args.lora}_dr_{args.dr}.h5ad")

    # Save loss history to CSV
    loss_history_df = pd.DataFrame({'round': range(1, len(loss_history) + 1), 'loss': loss_history})
    loss_history_df.to_csv(f'./tonsil_logs/loss_history_inr_{args.inr}_dp_{args.dp}_lora_{args.lora}_dr_{args.dr}.csv', index=False)
    print(f"Saved loss history to ./tonsil_logs/loss_history_inr_{args.inr}_dp_{args.dp}_lora_{args.lora}_dr_{args.dr}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpaMosaic with optional INR layers.")
    parser.add_argument('--inr', action='store_true', help="Enable INR layers in the model.")
    parser.add_argument('--inr_hidden_size', type=int, default=8, help="Hidden size for the INR layers.")
    parser.add_argument('--lora', action='store_true', help="Enable LoRA layers in the model.")
    parser.add_argument('--dp', action='store_true', help="Enable DP in the model.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for DP.")
    parser.add_argument('--noise_std', type=float, default=0.5, help="Noise standard deviation for DP.")
    parser.add_argument('--dr', action='store_true', help="Enable decentralized randomization.")
    args = parser.parse_args()

    main(args)
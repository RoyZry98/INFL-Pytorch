import os
import scanpy as sc
from os.path import join
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse  # 用于解析命令行参数
import pandas as pd  # 确保已导入 pandas

import sys
sys.path.insert(0, '../../..')

from spamosaic.framework import SpaMosaic
from spamosaic.utils import nn_approx, plot_basis, clustering, split_adata_ob, get_umap
from spamosaic.preprocessing import RNA_preprocess
import spamosaic.utils as utls

os.environ['R_HOME'] = '/deltadisk/miniconda3/envs/SpaMosaic/lib/R'
os.environ['R_USER'] = '/deltadisk/miniconda3/envs/SpaMosaic/lib/python3.8/site-packages/rpy2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for CuBLAS operation and you have CUDA >= 10.2

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
    data_dir = '/deltadisk/zhangrongyu/SpaMosaic/data/integration/Human_lymph_node'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_clients = 3
    num_rounds = 1
    local_epochs = 2
    learning_rate = 0.01

    # Load client datasets
    ad1_rna = sc.read_h5ad(join(data_dir, 'slice1/s1_adata_rna.h5ad'))
    ad2_rna = sc.read_h5ad(join(data_dir, 'slice2/s2_adata_rna.h5ad'))
    ad3_rna = sc.read_h5ad(join(data_dir, 'slice3/s3_adata_rna.h5ad'))

    input_dict = {
        'rna': [ad1_rna, ad2_rna, ad3_rna],
    }

    input_key = 'dimred_bc'

    # Preprocess data
    RNA_preprocess(input_dict['rna'], batch_corr=True, favor='scanpy', n_hvg=5000, batch_key='src', key=input_key)

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
                client_data = [ad1_rna, ad2_rna]
            elif client_idx == 1:  # Client 2
                client_data = [ad3_rna, ad2_rna]
            elif client_idx == 2:  # Client 3
                client_data = [ad1_rna, ad3_rna]

            # Create a local SpaMosaic instance for the client
            local_model = SpaMosaic(
                modBatch_dict={'rna': client_data},  # Single client's data
                input_key=input_key,
                batch_key='src', intra_knn=2, inter_knn=2, w_g=0.8, 
                seed=1234, device='cuda:0'
            )
            local_model.prepare_graphs({'rna': client_data})  # Prepare graphs for the client
            local_model.mod_model = local_model.prepare_net(net='wlgcn', inr=args.inr, inr_hidden_size=args.inr_hidden_size)  # Assign mod_model here

            # Load global weights into local model
            for k in local_model.mod_list:
                local_model.mod_model[k].load_state_dict(global_weights[k])

            # Train the local model
            local_model.train(net=copy.deepcopy(local_model.mod_model), lr=learning_rate, n_epochs=local_epochs, w_rec_g=0.)
            local_weights.append({k: copy.deepcopy(local_model.mod_model[k].state_dict()) for k in local_model.mod_list})
            local_losses.append(local_model.loss_rec[-1])  # Record the last loss
        
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
    ad_mosaic.write_h5ad(f'./merged_embeddings_{args.inr}.h5ad')
    print(f"Saved merged embeddings to ./output_embeddings/merged_embeddings_inr_{args.inr}.h5ad")

    # Save loss history to CSV
    loss_history_df = pd.DataFrame({'round': range(1, len(loss_history) + 1), 'loss': loss_history})
    loss_history_df.to_csv(f'./loss_history_inr_{args.inr}.csv', index=False)
    print(f"Saved loss history to ./loss_history_inr_{args.inr}.csv")

    ad_mosaic = get_umap(ad_mosaic, use_reps=['merged_emb'])
    utls.plot_basis(ad_mosaic, basis='merged_emb_umap', color=['src'])
    utls.clustering(ad_mosaic, n_cluster=7, used_obsm='merged_emb', algo='mclust', key='mclust')
    utls.split_adata_ob(ad_embs, ad_mosaic, 'obs', 'mclust')
    utls.plot_basis(ad_mosaic, basis='merged_emb_umap', color=['src', 'mclust'])
    for ad in ad_embs:
        utls.plot_basis(ad, 'spatial', 'mclust', s=70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpaMosaic with optional INR layers.")
    parser.add_argument('--inr', action='store_true', help="Enable INR layers in the model.")
    parser.add_argument('--inr_hidden_size', type=int, default=8, help="Hidden size for the INR layers.")
    args = parser.parse_args()

    main(args)
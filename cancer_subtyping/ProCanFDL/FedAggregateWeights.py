import os
import shap
import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from FedModel import FedProtNet


class WeightsAggregation:
    def __init__(self, model_paths, global_model=None, mu=0.1, use_dr=False):
        self.models = [torch.load(model_path) for model_path in model_paths]
        self.model_dict = {}
        self.mu = mu  # Proximal term coefficient
        self.global_weights = None
        self.global_model = global_model
        self.use_dr = use_dr

    def decentralized_randomization(self, models):
        """Randomizes the order of local model weights for decentralized randomization."""
        for i in range(len(models) - 1):
            j = random.randint(i, len(models) - 1)  # Random index from i to len(models)-1
            models[i], models[j] = models[j], models[i]  # Swap models
        return models

    def fed_avg(self):
        # Apply decentralized randomization if enabled
        models_to_use = self.models.copy()
        if self.use_dr:
            models_to_use = self.decentralized_randomization(models_to_use)
            print(f"Applied decentralized randomization to {len(models_to_use)} models")
        
        self.global_weights = {key: torch.zeros_like(value) for key, value in models_to_use[0].items()}
        num_models = len(models_to_use)

        for model_weights in models_to_use:
            for key in self.global_weights.keys():
                # Only average parameters that are float tensors (weights and biases)
                # Skip integer tensors (like buffer indices, counts, etc.)
                if self.global_weights[key].dtype.is_floating_point:
                    self.global_weights[key] += model_weights[key] / num_models
                else:
                    # For non-floating point tensors, just copy the first model's values
                    if self.global_weights[key].sum() == 0:  # Only copy if not already set
                        self.global_weights[key] = model_weights[key].clone()

    def fed_prox(self):
        # Apply decentralized randomization if enabled
        models_to_use = self.models.copy()
        if self.use_dr:
            models_to_use = self.decentralized_randomization(models_to_use)
            print(f"Applied decentralized randomization to {len(models_to_use)} models")
        
        self.global_weights = {key: torch.zeros_like(value) for key, value in self.global_model.items()}
        num_models = len(models_to_use)

        for model_weights in models_to_use:
            for key in self.global_weights.keys():
                # Only apply FedProx to parameters that are float tensors (weights and biases)
                # Skip integer tensors (like buffer indices, counts, etc.)
                if self.global_weights[key].dtype.is_floating_point:
                    # FedProx: Incorporating the proximal term
                    fed_prox_update = model_weights[key] + self.mu * (model_weights[key] - self.global_model[key])
                    self.global_weights[key] += fed_prox_update / num_models
                else:
                    # For non-floating point tensors, just copy the first model's values
                    if self.global_weights[key].sum() == 0:  # Only copy if not already set
                        self.global_weights[key] = model_weights[key].clone()

    def save_model(self, save_path):
        torch.save(self.global_weights, save_path)


if __name__ == "__main__":
    weight_agg = WeightsAggregation(
        model_paths=[f"../models/Fed/local_model_Broad_Cancer_Type_{i}.pt" for i in range(10)])
    weight_agg.fed_avg()
    weight_agg.save_model("../models/Fed/Broad_Cancer_Type_fedavg.pt")

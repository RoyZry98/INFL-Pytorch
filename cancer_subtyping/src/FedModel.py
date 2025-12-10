# Description: This file contains the implementation of the FedProtNet model.
import torch
import torch.nn as nn
import torch.nn.functional as F


class FedProtNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(FedProtNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Initial embedding layer
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)

        # Residual block 1
        self.resblock1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.resblock1_bn = nn.BatchNorm1d(self.hidden_dim)

        # Residual block 2
        self.resblock2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.resblock2_bn = nn.BatchNorm1d(self.hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc2 = nn.Linear(self.hidden_dim // 2, self.num_classes)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # Input embedding
        x = self.embedding(x)

        # Residual block 1
        residual = x
        x = self.resblock1(x)
        x = self.resblock1_bn(x + residual)  # Add residual connection

        # Residual block 2
        residual = x
        x = self.resblock2(x)
        x = self.resblock2_bn(x + residual)  # Add residual connection

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
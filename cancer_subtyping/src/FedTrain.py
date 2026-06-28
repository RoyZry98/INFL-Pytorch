import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
import warnings
import math
from typing import Optional, Tuple
from FedModel import FedProtNet, FedProtNet_DP
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except Exception:
    PrivacyEngine = None
    OPACUS_AVAILABLE = False


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
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device)

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

            # Copy the original weights and bias, ensuring device match
            new_layer.linear.weight.data.copy_(child.weight.data.to(device))
            if child.bias is not None:
                new_layer.linear.bias.data.copy_(child.bias.data.to(device))

            # Replace the layer in the module
            setattr(module, name, new_layer)
        else:
            # Recursively apply to child modules
            replace_linear_with_lora(child, r=r, alpha=alpha, device=device)

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
        batch_size = h.shape[0]
        act_embed = self.act_proj(h)
        key_embed = key_vec.unsqueeze(0).expand(batch_size, -1)
        joint = torch.cat([act_embed, key_embed, act_embed * key_embed], dim=1)
        gamma_beta = self.film(joint)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        return h * (1.0 + self.key_strength * gamma) + self.key_strength * beta


class FedProtNet_KeyedINR(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        dropout=0.5,
        coord_dim=2,
        coord_points=128,
        key_dim=16,
        key_hidden=8,
        key_strength=20.0,
        key_only_fc2_bias=False,
        coords=None,
        device="cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.key_only_fc2_bias = bool(key_only_fc2_bias)
        self.key_strength = float(key_strength)

        device = torch.device(device)
        if coords is None:
            coords = make_coordinates(
                seed=0,
                mode="uniform",
                num_points=coord_points,
                coord_dim=coord_dim,
                device=device,
            )

        self.key_encoder = CoordinateKeyEncoder(
            coord_dim=coord_dim,
            num_points=coord_points,
            key_dim=key_dim,
            coords=coords,
        )

        self.embedding = KeyControlledLinear(
            input_dim,
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.embedding_film = ActivationDependentFiLM1d(
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )

        self.rb1_linear1 = KeyControlledLinear(
            hidden_dim,
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.rb1_film1 = ActivationDependentFiLM1d(
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.rb1_bn1 = nn.BatchNorm1d(hidden_dim)

        self.rb1_linear2 = KeyControlledLinear(
            hidden_dim,
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.rb1_film2 = ActivationDependentFiLM1d(
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.resblock1_bn = nn.BatchNorm1d(hidden_dim)

        self.rb2_linear1 = KeyControlledLinear(
            hidden_dim,
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.rb2_film1 = ActivationDependentFiLM1d(
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.rb2_bn1 = nn.BatchNorm1d(hidden_dim)

        self.rb2_linear2 = KeyControlledLinear(
            hidden_dim,
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.rb2_film2 = ActivationDependentFiLM1d(
            hidden_dim,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.resblock2_bn = nn.BatchNorm1d(hidden_dim)

        self.fc1 = KeyControlledLinear(
            hidden_dim,
            hidden_dim // 2,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )
        self.fc1_film = ActivationDependentFiLM1d(
            hidden_dim // 2,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
        )

        self.fc2 = KeyControlledLinear(
            hidden_dim // 2,
            num_classes,
            key_dim=key_dim,
            hidden_dim=key_hidden,
            key_strength=key_strength,
            bias=not self.key_only_fc2_bias,
        )

        if self.key_only_fc2_bias:
            self.fc2_key_bias = nn.Sequential(
                nn.Linear(key_dim, key_hidden),
                nn.SiLU(),
                nn.LayerNorm(key_hidden),
                nn.Linear(key_hidden, num_classes),
            )
            nn.init.zeros_(self.fc2_key_bias[-1].weight)
            nn.init.zeros_(self.fc2_key_bias[-1].bias)
        else:
            self.fc2_key_bias = None

        self.dropout = nn.Dropout(self.dropout_rate)

    @property
    def coords(self) -> torch.Tensor:
        return self.key_encoder.coords

    def set_coordinates(self, coords: torch.Tensor) -> None:
        self.key_encoder.set_coordinates(coords)

    def forward(self, x):
        key_vec = self.key_encoder()

        x = self.embedding(x, key_vec)
        x = self.embedding_film(x, key_vec)

        residual = x
        x = self.rb1_linear1(x, key_vec)
        x = self.rb1_film1(x, key_vec)
        x = self.rb1_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.rb1_linear2(x, key_vec)
        x = self.rb1_film2(x, key_vec)
        x = self.resblock1_bn(x + residual)

        residual = x
        x = self.rb2_linear1(x, key_vec)
        x = self.rb2_film1(x, key_vec)
        x = self.rb2_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.rb2_linear2(x, key_vec)
        x = self.rb2_film2(x, key_vec)
        x = self.resblock2_bn(x + residual)

        x = self.fc1(x, key_vec)
        x = self.fc1_film(x, key_vec)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x, key_vec)
        if self.fc2_key_bias is not None:
            x = x + self.key_strength * torch.tanh(self.fc2_key_bias(key_vec)).unsqueeze(0)

        return x

    def forward_without_inr(self, x):
        x = self.embedding.forward_without_key(x)

        residual = x
        x = self.rb1_linear1.forward_without_key(x)
        x = self.rb1_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.rb1_linear2.forward_without_key(x)
        x = self.resblock1_bn(x + residual)

        residual = x
        x = self.rb2_linear1.forward_without_key(x)
        x = self.rb2_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.rb2_linear2.forward_without_key(x)
        x = self.resblock2_bn(x + residual)

        x = self.fc1.forward_without_key(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2.forward_without_key(x)

        return x


def _is_keyed_inr_param(name: str) -> bool:
    return (
        "key_encoder" in name
        or "key_to_mod" in name
        or "film" in name
        or "act_proj" in name
        or "fc2_key_bias" in name
    )


class TrainFedProtNet:
    """
    Training and evaluation class for FedProtNet model with optional INR integration.
    """

    def __init__(
        self, train_dataset, test_dataset, hypers, load_model=None, save_path=None, use_inr=None
    ):
        """
        Initialize the training class.

        Args:
            train_dataset: Training dataset (ProtDataset)
            test_dataset: Test dataset (ProtDataset)
            hypers: Dictionary of hyperparameters
            load_model: Path to load pre-trained model (optional)
            save_path: Path to save model (optional)
            use_inr: Whether to use INR layers in the model (default: False)
            inr_config: Dictionary with INR configuration (optional)
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.hypers = hypers
        self.save_path = save_path
        self.use_inr = use_inr
        self.inr_config = {}

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.use_inr:
            coords = make_coordinates(
                seed=0,
                mode="uniform",
                num_points=128,
                coord_dim=2,
                device=self.device,
                constant=1.0,
            )
            self.model = FedProtNet_KeyedINR(
                input_dim=train_dataset.feature_dim,
                hidden_dim=hypers["hidden_dim"],
                num_classes=train_dataset.num_classes,
                dropout=hypers["dropout"],
                coord_dim=2,
                coord_points=128,
                key_dim=16,
                key_hidden=8,
                key_strength=20.0,
                key_only_fc2_bias=False,
                coords=coords,
                device=self.device,
            ).to(self.device)
        else:
            self.model = FedProtNet(
                input_dim=train_dataset.feature_dim,
                hidden_dim=hypers["hidden_dim"],
                num_classes=train_dataset.num_classes,
                dropout=hypers["dropout"],
            ).to(self.device)

        # Load pre-trained weights if specified
        if load_model is not None:
            self.load_model(load_model)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=hypers["lr"],
            weight_decay=hypers["weight_decay"],
        )
        
        # Check if INR parameters are included in optimizer
        self._check_inr_in_optimizer()

        # Training parameters
        self.batch_size = hypers["batch_size"]
        self.epochs = hypers["epochs"]

        # Store predictions for confusion matrix
        self.predictions = None

    def _check_inr_in_optimizer(self):
        """
        Check if INR parameters are included in the optimizer and print detailed information.
        """
        print("=" * 80)
        print("INR OPTIMIZER CHECK")
        print("=" * 80)
        
        # Get all model parameters
        model_params = list(self.model.parameters())
        optimizer_params = []
        for group in self.optimizer.param_groups:
            optimizer_params.extend(group['params'])
        
        # Count parameters by type
        total_model_params = 0
        inr_params = 0
        linear_params = 0
        other_params = 0

        
        for name, param in self.model.named_parameters():
            total_model_params += param.numel()
            
            if _is_keyed_inr_param(name):
                inr_params += param.numel()
            elif 'linear' in name and not _is_keyed_inr_param(name):
                linear_params += param.numel()
            else:
                other_params += param.numel()
        
        # Check if all model parameters are in optimizer
        model_param_ids = {id(p) for p in model_params}
        optimizer_param_ids = {id(p) for p in optimizer_params}
        
        
        if model_param_ids == optimizer_param_ids:
            print("✓ SUCCESS: All model parameters are included in optimizer")
        else:
            print("✗ ERROR: Some parameters are missing from optimizer")
            missing = model_param_ids - optimizer_param_ids
            extra = optimizer_param_ids - model_param_ids
            if missing:
                print(f"  Missing parameters: {len(missing)}")
            if extra:
                print(f"  Extra parameters: {len(extra)}")
        
        # Check specifically for INR parameters
        inr_in_optimizer = 0
        for name, param in self.model.named_parameters():
            if _is_keyed_inr_param(name) and id(param) in optimizer_param_ids:
                inr_in_optimizer += param.numel()
        
        print(f"\nINR Parameter Check:")
        print("-" * 50)
        if inr_params > 0:
            if inr_in_optimizer == inr_params:
                print(f"✓ SUCCESS: All INR parameters ({inr_in_optimizer:,}) are in optimizer")
            else:
                print(f"✗ WARNING: Only {inr_in_optimizer:,} out of {inr_params:,} INR parameters are in optimizer")
        else:
            print("ℹ INFO: No INR parameters found in model")
        
        print("=" * 80)

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_data)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Calculate AUC (one-vs-rest for multiclass)
        try:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="weighted"
            )
        except:
            auc = 0.0

        # Store predictions for confusion matrix
        self.predictions = all_probs

        return accuracy, f1, auc, all_preds, all_labels, all_probs

    def run_train_val(self):
        """
        Run full training and validation

        Returns:
            dict: Dictionary containing training results
        """
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        best_f1 = 0
        best_auc = 0
        best_epoch = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            test_acc, test_f1, test_auc, _, _, _ = self.evaluate(test_loader)

            # Save best model
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_auc = test_auc
                best_epoch = epoch
                if self.save_path is not None:
                    self.save_model()

            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Loss: {train_loss:.4f}, "
                    f"Test Acc: {test_acc:.4f}, "
                    f"Test F1: {test_f1:.4f}, "
                    f"Test AUC: {test_auc:.4f}"
                )

        # Load best model
        if self.save_path is not None:
            self.load_model(self.save_path)

        # Final evaluation
        test_acc, test_f1, test_auc, _, _, _ = self.evaluate(test_loader)

        return {
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "best_f1": best_f1,
            "best_auc": best_auc,
            "best_epoch": best_epoch,
        }

    def predict(self):
        """
        Run prediction on test set (used for federated evaluation)

        Returns:
            dict: Dictionary containing test results
        """
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        test_acc, test_f1, test_auc, _, _, _ = self.evaluate(test_loader)

        return {"test_acc": test_acc, "test_f1": test_f1, "test_auc": test_auc}

    def get_pred_for_cm(self):
        """
        Get predictions for confusion matrix

        Returns:
            numpy array: Prediction probabilities
        """
        if self.predictions is None:
            test_loader = DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False
            )
            _, _, _, _, _, _ = self.evaluate(test_loader)

        return self.predictions

    def predict_custom(self, data):
        """
        Make predictions on custom data

        Args:
            data: numpy array or torch tensor of shape (n_samples, n_features)

        Returns:
            tuple: (predicted_classes, probabilities)
        """
        self.model.eval()

        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        data = data.to(self.device)

        with torch.no_grad():
            outputs = self.model(data)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy(), probs.cpu().numpy()

    def save_model(self, path=None):
        """Save model weights"""
        save_path = path if path is not None else self.save_path
        if save_path is not None:
            # Always save the unwrapped model state dict (without _module. prefix)
            state_dict = self.model.state_dict()
            
            # If model is wrapped (has _module. prefix), remove it before saving
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_module.'):
                    new_key = key[8:]  # Remove '_module.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            torch.save(new_state_dict, save_path)

    def load_model(self, path):
        """Load model weights"""
        state_dict = torch.load(path, map_location=self.device)
        
        # Check if current model is wrapped (has _module attribute or is GradSampleModule)
        is_wrapped = hasattr(self.model, '_module') or type(self.model).__name__ == 'GradSampleModule'
        
        # Check if loaded state_dict has _module. prefix
        has_module_prefix = any(key.startswith('_module.') for key in state_dict.keys())
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if is_wrapped and not has_module_prefix:
                # Model is wrapped but state_dict doesn't have prefix -> add prefix
                new_key = '_module.' + key
                new_state_dict[new_key] = value
            elif not is_wrapped and has_module_prefix:
                # Model is not wrapped but state_dict has prefix -> remove prefix
                new_key = key[8:]  # Remove '_module.' prefix
                new_state_dict[new_key] = value
            else:
                # Prefix matches, no change needed
                new_state_dict[key] = value
        
        self.model.load_state_dict(new_state_dict)


class TrainFedProtNet_DP:
    """
    Training and evaluation class for FedProtNet model with differential privacy (no INR).
    """

    def __init__(
        self, train_dataset, test_dataset, hypers, load_model=None, save_path=None, use_lora=None
    ):
        """
        Initialize the DP training class.

        Args:
            train_dataset: Training dataset (ProtDataset)
            test_dataset: Test dataset (ProtDataset)
            hypers: Dictionary of hyperparameters
            load_model: Path to load pre-trained model (optional)
            save_path: Path to save model (optional)
            use_lora: Whether to use LoRA layers (optional)
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.hypers = hypers
        self.save_path = save_path
        self.use_lora = use_lora

        # DP config
        self.dp_noise_multiplier = 3.757 # dp
        # self.dp_noise_multiplier = 2.053 # ppml
        self.dp_max_grad_norm = 1.0
        self.dp_target_epsilon = hypers.get("dp_target_epsilon")  # optional
        self.dp_target_delta = hypers.get("dp_target_delta", 1e-5)
        self.dp_secure_rng = bool(hypers.get("dp_secure_rng", False))

        # LoRA config
        self.lora_rank = 4
        self.lora_alpha = 1.0

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize DP model
        self.model = FedProtNet_DP(
            input_dim=train_dataset.feature_dim,
            hidden_dim=hypers["hidden_dim"],
            num_classes=train_dataset.num_classes,
            dropout=hypers["dropout"],
        ).to(self.device)

        # Replace linear layers with LoRA if use_lora is True
        if self.use_lora:
            replace_linear_with_lora(
                self.model,
                r=self.lora_rank,
                alpha=self.lora_alpha,
                device=self.device
            )
            # Ensure the entire model is on the correct device after LoRA replacement
            self.model = self.model.to(self.device)

        # Load pre-trained weights if specified
        if load_model is not None:
            self.load_model(load_model)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=hypers["lr"],
            weight_decay=hypers["weight_decay"],
        )

        # Training parameters
        self.batch_size = hypers["batch_size"]
        self.epochs = hypers["epochs"]

        # Store predictions for confusion matrix
        self.predictions = None

        # Initialize DP
        self.privacy_engine = None
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is not installed. Please install it with: pip install opacus")
        
        self.privacy_engine = PrivacyEngine()
        
        # Suppress the backward hook warning from Opacus
        # This warning is expected when using PrivacyEngine and doesn't affect training
        warnings.filterwarnings('ignore', message='Full backward hook is firing when gradients are computed with respect to module outputs')

    def _check_lora_parameters(self):
        """Check LoRA parameters and print debug information"""
        print("=" * 80)
        print("LoRA PARAMETER CHECK")
        print("=" * 80)
        
        lora_layers = 0
        total_params = 0
        lora_params = 0
        frozen_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_layers += 1
                print(f"LoRA Layer: {name}")
                print(f"  - Input features: {module.in_features}")
                print(f"  - Output features: {module.out_features}")
                print(f"  - Rank: {module.rank}")
                print(f"  - Alpha: {module.alpha}")
                print(f"  - Scaling: {module.scaling}")
                
                # Check parameter counts
                for param_name, param in module.named_parameters():
                    total_params += param.numel()
                    if param.requires_grad:
                        lora_params += param.numel()
                        print(f"  - Trainable: {param_name} - {param.numel()} params")
                    else:
                        frozen_params += param.numel()
                        print(f"  - Frozen: {param_name} - {param.numel()} params")
                
                # Check weight statistics
                print(f"  - LoRA A weight stats: mean={module.lora_A.weight.mean().item():.6f}, std={module.lora_A.weight.std().item():.6f}")
                print(f"  - LoRA B weight stats: mean={module.lora_B.weight.mean().item():.6f}, std={module.lora_B.weight.std().item():.6f}")
                print(f"  - Original linear weight stats: mean={module.linear.weight.mean().item():.6f}, std={module.linear.weight.std().item():.6f}")
                print()
        
        print(f"Total LoRA layers: {lora_layers}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {lora_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print("=" * 80)
        
        # Check optimizer parameters
        print("\nOptimizer Parameter Groups:")
        for i, param_group in enumerate(self.optimizer.param_groups):
            print(f"  Group {i}: lr={param_group['lr']}, weight_decay={param_group.get('weight_decay', 0)}")
            group_params = sum(p.numel() for p in param_group['params'])
            print(f"    Parameters: {group_params:,}")
        print("=" * 80)
        
        # Test forward pass with dummy data
        if lora_layers > 0:
            print("Testing LoRA forward pass...")
            dummy_input = torch.randn(1, self.train_dataset.feature_dim).to(self.device)
            try:
                with torch.no_grad():
                    output = self.model(dummy_input)
                    print(f"✓ Forward pass successful. Output shape: {output.shape}")
                    print(f"✓ Output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
            except Exception as e:
                print(f"✗ Forward pass failed: {e}")
            print("=" * 80)

    def train_epoch(self, dataloader):
        """Train for one epoch with DP"""
        self.model.train()
        total_loss = 0

        for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_data)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Calculate AUC (one-vs-rest for multiclass)
        try:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="weighted"
            )
        except:
            auc = 0.0

        # Store predictions for confusion matrix
        self.predictions = all_probs

        return accuracy, f1, auc, all_preds, all_labels, all_probs

    def run_train_val(self):
        """
        Run full training and validation with DP

        Returns:
            dict: Dictionary containing training results
        """
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Attach PrivacyEngine to model/optimizer/dataloader
        try:
            self.model, self.optimizer, train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                noise_multiplier=self.dp_noise_multiplier,
                max_grad_norm=self.dp_max_grad_norm,
            )
            print(f"[DP] Enabled: noise_multiplier={self.dp_noise_multiplier}, max_grad_norm={self.dp_max_grad_norm}, secure_rng={self.dp_secure_rng}")
        except Exception as e:
            raise RuntimeError(f"Failed to enable DP via Opacus: {e}")

        best_f1 = 0
        best_epoch = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            test_acc, test_f1, test_auc, _, _, _ = self.evaluate(test_loader)

            # Report privacy budget
            # try:
            #     eps = self.privacy_engine.get_epsilon(delta=self.dp_target_delta)
            #     print(f"[DP] Epoch {epoch+1}: epsilon={eps:.2f}, delta={self.dp_target_delta}")
            # except Exception as e:
            #     print(f"[DP] Failed to compute epsilon: {e}")

            # Save best model
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_epoch = epoch
                if self.save_path is not None:
                    self.save_model()

            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Loss: {train_loss:.4f}, "
                    f"Test Acc: {test_acc:.4f}, "
                    f"Test F1: {test_f1:.4f}, "
                    f"Test AUC: {test_auc:.4f}"
                )

        # Load best model
        if self.save_path is not None:
            self.load_model(self.save_path)

        # Final evaluation
        test_acc, test_f1, test_auc, _, _, _ = self.evaluate(test_loader)

        return {
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "best_epoch": best_epoch,
        }

    def predict(self):
        """
        Run prediction on test set (used for federated evaluation)

        Returns:
            dict: Dictionary containing test results
        """
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        test_acc, test_f1, test_auc, _, _, _ = self.evaluate(test_loader)

        return {"test_acc": test_acc, "test_f1": test_f1, "test_auc": test_auc}

    def get_pred_for_cm(self):
        """
        Get predictions for confusion matrix

        Returns:
            numpy array: Prediction probabilities
        """
        if self.predictions is None:
            test_loader = DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False
            )
            _, _, _, _, _, _ = self.evaluate(test_loader)

        return self.predictions

    def predict_custom(self, data):
        """
        Make predictions on custom data

        Args:
            data: numpy array or torch tensor of shape (n_samples, n_features)

        Returns:
            tuple: (predicted_classes, probabilities)
        """
        self.model.eval()

        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        data = data.to(self.device)

        with torch.no_grad():
            outputs = self.model(data)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy(), probs.cpu().numpy()

    def save_model(self, path=None):
        """Save model weights"""
        save_path = path if path is not None else self.save_path
        if save_path is not None:
            # Always save the unwrapped model state dict (without _module. prefix)
            state_dict = self.model.state_dict()
            
            # If model is wrapped (has _module. prefix), remove it before saving
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_module.'):
                    new_key = key[8:]  # Remove '_module.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            torch.save(new_state_dict, save_path)

    def load_model(self, path):
        """Load model weights"""
        state_dict = torch.load(path, map_location=self.device)
        
        # Check if current model is wrapped (has _module attribute or is GradSampleModule)
        is_wrapped = hasattr(self.model, '_module') or type(self.model).__name__ == 'GradSampleModule'
        
        # Check if loaded state_dict has _module. prefix
        has_module_prefix = any(key.startswith('_module.') for key in state_dict.keys())
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if is_wrapped and not has_module_prefix:
                # Model is wrapped but state_dict doesn't have prefix -> add prefix
                new_key = '_module.' + key
                new_state_dict[new_key] = value
            elif not is_wrapped and has_module_prefix:
                # Model is not wrapped but state_dict has prefix -> remove prefix
                new_key = key[8:]  # Remove '_module.' prefix
                new_state_dict[new_key] = value
            else:
                # Prefix matches, no change needed
                new_state_dict[key] = value
        
        self.model.load_state_dict(new_state_dict)

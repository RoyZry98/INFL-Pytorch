import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
import warnings
import math
from FedModel import FedProtNet
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except Exception:
    PrivacyEngine = None
    OPACUS_AVAILABLE = False


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
        # Create coordinates and register as buffer on the correct device
        coords = self._create_coordinates()
        self.register_buffer('coords', coords.to(device))
        
        # Move the entire module to the specified device
        self.to(device)
    
    def _build_inr(self, in_dim, hid, out_dim, num_layers, act, bias, device):
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hid, hid))
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
        coords = idxs / (self.out_features - 1) * 2 - 1 if self.out_features > 1 else idxs
        pe = self._positional_encoding(coords)
        return pe.to(self.device)

    def _positional_encoding(self, coords):
        L = self.inr_input_dim // 2
        x = coords
        pe = [torch.sin((2.0 ** i) * np.pi * x) for i in range(L)]
        pe += [torch.cos((2.0 ** i) * np.pi * x) for i in range(L)]
        pe = torch.cat(pe, dim=1)
        return pe.to(self.device)

    def forward(self, x):
        feat = self.linear(x)
        delta = self.inr(self.coords).view(1, -1)
        return feat + delta


def replace_linear_with_inr(module, 
                           inr_input_dim=16, 
                           inr_hidden_size=32, 
                           inr_output_dim=1,
                           inr_layer_num=3, 
                           inr_output_activation=None, 
                           inr_output_bias=0.,
                           device='cpu'):
    """
    Replace all nn.Linear layers in a given module with INRLinear layers.
    
    Args:
        module (nn.Module): The neural network module to modify.
        inr_input_dim (int): Positional encoding input dimension.
        inr_hidden_size (int): Hidden layer size for INR.
        inr_output_dim (int): Output dimension for INR.
        inr_layer_num (int): Number of layers in the INR network.
        inr_output_activation (str or None): Activation function for INR output.
        inr_output_bias (float): Bias for INR output initialization.
        device (str): Device for the INR layers.
    """
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
            # Copy original weights and biases, ensuring device match
            new_layer.linear.weight.data.copy_(child.weight.data.to(device))
            if child.bias is not None:
                new_layer.linear.bias.data.copy_(child.bias.data.to(device))
            setattr(module, name, new_layer)
        else:
            # Recursively traverse child modules
            replace_linear_with_inr(child, inr_input_dim, inr_hidden_size, inr_output_dim, inr_layer_num, inr_output_activation, inr_output_bias, device)


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

        # Initialize model
        self.model = FedProtNet(
            input_dim=train_dataset.feature_dim,
            hidden_dim=hypers["hidden_dim"],
            num_classes=train_dataset.num_classes,
            dropout=hypers["dropout"],
        ).to(self.device)

        # Replace linear layers with INRLinear if use_inr is True
        if self.use_inr:
            replace_linear_with_inr(
                self.model,
                inr_input_dim=16,
                inr_hidden_size=16,
                inr_output_dim=1,
                inr_layer_num=3,
                inr_output_activation=None,
                inr_output_bias=0.0,
                device=self.device
            )

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
            
            if 'inr' in name:
                inr_params += param.numel()
            elif 'linear' in name and 'inr' not in name:
                linear_params += param.numel()
            else:
                other_params += param.numel()
        
        # Check if all model parameters are in optimizer
        model_param_ids = {id(p) for p in model_params}
        optimizer_param_ids = {id(p) for p in optimizer_params}
        
        
        # Check specifically for INR parameters
        inr_in_optimizer = 0
        for name, param in self.model.named_parameters():
            if 'inr' in name and id(param) in optimizer_param_ids:
                inr_in_optimizer += param.numel()
        

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

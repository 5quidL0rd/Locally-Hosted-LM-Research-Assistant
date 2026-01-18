# agents/nn_builder_agent.py - STREAMLINED NEURAL NETWORK BUILDER
"""
A streamlined, intuitive neural network builder supporting:
- MLP, LSTM, CNN, and Transformer architectures
- Automatic task detection (classification/regression)
- Easy model saving/loading with registry
- Rich visualizations of training progress and predictions
- Seamless integration with Kaggle datasets

Usage:
    builder = NeuralNetworkBuilder(memory)

    # Quick train
    results = builder.quick_train("path/to/data.csv")

    # Custom experiment
    results = builder.run_experiment({
        "dataset": {"path": "data.csv", "target": "price"},
        "models": ["mlp", "lstm", "transformer"],
        "epochs": 100
    })

    # Load and use saved model
    model = builder.load_model("experiment_20240115_mlp")
    predictions = builder.predict(model, new_data)
"""

import json
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union
from dataclasses import dataclass, asdict
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.model_selection import train_test_split

# Optional Optuna import for hyperparameter tuning
try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[NNBuilder] Optuna not installed. Hyperparameter tuning disabled. Install with: pip install optuna")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a neural network model"""
    model_type: str
    input_size: int
    output_size: int
    hidden_sizes: List[int] = None
    dropout: float = 0.2
    num_layers: int = 2
    num_heads: int = 4  # For Transformer
    seq_length: int = 20
    task_type: str = "regression"

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64]


@dataclass
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 15
    min_delta: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "plateau"  # "plateau", "cosine", "none"


@dataclass
class ExperimentResult:
    """Results from a training experiment"""
    model_name: str
    model_type: str
    task_type: str
    final_metrics: Dict[str, float]
    best_epoch: int
    training_time: float
    model_path: str
    config: Dict[str, Any]


# ============================================================================
# DATASET HANDLING
# ============================================================================

class NNDataset(Dataset):
    """Flexible dataset supporting both tabular and sequential data"""

    def __init__(self, X: np.ndarray, y: np.ndarray, task_type: str = "regression"):
        self.X = torch.FloatTensor(X)
        if task_type == "classification":
            self.y = torch.LongTensor(y)
        else:
            self.y = torch.FloatTensor(y)
        self.task_type = task_type

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataProcessor:
    """Handles all data preprocessing"""

    def __init__(self):
        self.scaler_X = None
        self.scaler_y = None
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = None
        self.task_type = None
        self.num_classes = 1

    def fit_transform(self, df: pd.DataFrame, target_column: str = None,
                      task_type: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessors and transform data"""

        # Auto-detect target column
        if target_column is None:
            candidates = ['target', 'label', 'class', 'y', 'Close', 'price', 'Price']
            target_column = next((c for c in candidates if c in df.columns), df.columns[-1])

        self.target_name = target_column

        # Handle datetime columns
        date_cols = []
        for col in df.columns:
            if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]':
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass

        # Sort by date if available
        if date_cols:
            df = df.sort_values(date_cols[0]).reset_index(drop=True)

        # Extract target
        y = df[target_column].values

        # Drop target and date columns from features
        X_df = df.drop(columns=[target_column] + date_cols, errors='ignore')

        # Encode categorical features
        cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            self.label_encoders[col] = le

        # Handle missing values
        X_df = X_df.fillna(X_df.median(numeric_only=True))

        self.feature_names = X_df.columns.tolist()

        # Detect task type
        if task_type is None:
            # Filter out NaN values for analysis
            y_clean = y[~pd.isna(y)]
            unique_vals = len(np.unique(y_clean))

            # Check if target is numeric
            try:
                y_numeric = pd.to_numeric(y_clean, errors='raise')
                # Check if it looks like classification (few unique integer values)
                if unique_vals <= 20 and np.all(y_numeric == y_numeric.astype(int)):
                    task_type = "classification"
                else:
                    task_type = "regression"
            except (ValueError, TypeError):
                # Non-numeric target = classification
                task_type = "classification"

        self.task_type = task_type

        # Encode target for classification
        if task_type == "classification":
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            self.label_encoders['__target__'] = le_target
            self.num_classes = len(le_target.classes_)
        else:
            self.num_classes = 1
            # Scale target for regression
            y = y.astype(float)
            self.scaler_y = StandardScaler()
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Scale features
        self.scaler_X = StandardScaler()
        X = self.scaler_X.fit_transform(X_df.values)

        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessors"""
        X_df = df[self.feature_names].copy()

        for col, le in self.label_encoders.items():
            if col != '__target__' and col in X_df.columns:
                X_df[col] = le.transform(X_df[col].astype(str))

        X_df = X_df.fillna(X_df.median(numeric_only=True))
        return self.scaler_X.transform(X_df.values)

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Convert predictions back to original scale"""
        if self.task_type == "regression" and self.scaler_y is not None:
            return self.scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
        elif self.task_type == "classification" and '__target__' in self.label_encoders:
            return self.label_encoders['__target__'].inverse_transform(y.astype(int))
        return y

    def make_sequences(self, X: np.ndarray, y: np.ndarray,
                       seq_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for temporal models (LSTM, CNN, Transformer)"""
        if len(X) <= seq_length:
            raise ValueError(f"Dataset too small ({len(X)}) for sequence length {seq_length}")

        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:i + seq_length])
            ys.append(y[i + seq_length])

        return np.array(Xs), np.array(ys)

    def save(self, path: str):
        """Save preprocessor state"""
        state = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'task_type': self.task_type,
            'num_classes': self.num_classes
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'DataProcessor':
        """Load preprocessor state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        processor = cls()
        for k, v in state.items():
            setattr(processor, k, v)
        return processor


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class FlexibleMLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        layers = []
        prev_size = config.input_size

        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, config.output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Handle sequential input by flattening
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.network(x)


class FlexibleLSTM(nn.Module):
    """LSTM for sequential/time-series data"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_sizes[0],
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_out_size = config.hidden_sizes[0] * 2  # bidirectional

        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_size // 2, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_sizes[0], config.output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.fc(context)


class FlexibleCNN(nn.Module):
    """1D CNN for sequential data"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(config.input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        return self.fc(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).transpose(0, 1)  # back to (batch, seq_len, d_model)


class FlexibleTransformer(nn.Module):
    """Transformer encoder for sequential data"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        d_model = config.hidden_sizes[0]

        # Project input to d_model dimensions
        self.input_projection = nn.Linear(config.input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, config.seq_length, config.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model // 2, config.output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Use last timestep or average pool
        x = x[:, -1, :]  # (batch, d_model)
        return self.fc(x)


# ============================================================================
# MODEL REGISTRY - EASY SAVE/LOAD
# ============================================================================

class ModelRegistry:
    """Registry for saving, loading, and managing trained models"""

    def __init__(self, base_dir: str = "nn_models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load existing registry or create new one"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": {}}

    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def save_model(self, model: nn.Module, config: ModelConfig,
                   processor: DataProcessor, metrics: Dict,
                   name: str = None) -> str:
        """Save a trained model with all necessary metadata"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = name or f"{config.model_type}_{timestamp}"
        model_dir = self.base_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Save model weights
        torch.save(model.state_dict(), model_dir / "weights.pt")

        # Save config
        with open(model_dir / "config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2)

        # Save preprocessor
        processor.save(str(model_dir / "processor.pkl"))

        # Save metrics
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Update registry
        self.registry["models"][model_name] = {
            "path": str(model_dir),
            "model_type": config.model_type,
            "task_type": config.task_type,
            "metrics": metrics,
            "created": timestamp,
            "input_size": config.input_size,
            "output_size": config.output_size
        }
        self._save_registry()

        print(f"[Registry] Saved model: {model_name}")
        return model_name

    def load_model(self, name: str) -> Tuple[nn.Module, ModelConfig, DataProcessor]:
        """Load a model by name"""

        if name not in self.registry["models"]:
            available = list(self.registry["models"].keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")

        model_dir = Path(self.registry["models"][name]["path"])

        # Load config
        with open(model_dir / "config.json", 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)

        # Load preprocessor
        processor = DataProcessor.load(str(model_dir / "processor.pkl"))

        # Create and load model
        model = self._create_model(config)
        model.load_state_dict(torch.load(model_dir / "weights.pt", weights_only=True))
        model.eval()

        return model, config, processor

    def _create_model(self, config: ModelConfig) -> nn.Module:
        """Create model instance from config"""
        model_classes = {
            "mlp": FlexibleMLP,
            "lstm": FlexibleLSTM,
            "cnn": FlexibleCNN,
            "transformer": FlexibleTransformer
        }

        if config.model_type not in model_classes:
            raise ValueError(f"Unknown model type: {config.model_type}")

        return model_classes[config.model_type](config)

    def list_models(self) -> List[Dict]:
        """List all saved models"""
        return [
            {"name": name, **info}
            for name, info in self.registry["models"].items()
        ]

    def delete_model(self, name: str):
        """Delete a saved model"""
        import shutil

        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found")

        model_dir = Path(self.registry["models"][name]["path"])
        if model_dir.exists():
            shutil.rmtree(model_dir)

        del self.registry["models"][name]
        self._save_registry()
        print(f"[Registry] Deleted model: {name}")


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Handles model training with early stopping and metrics"""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, config: TrainingConfig,
              task_type: str = "regression") -> Dict:
        """Train model with early stopping"""

        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)

        # Learning rate scheduler
        if config.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        elif config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs
            )
        else:
            scheduler = None

        # Loss function
        if task_type == "classification":
            criterion = nn.CrossEntropyLoss()
            main_metric = "accuracy"
        else:
            criterion = nn.MSELoss()
            main_metric = "r2"

        best_val_loss = float('inf')
        best_metrics = {}
        best_epoch = 0
        patience_counter = 0
        best_state = None

        start_time = time.time()

        for epoch in range(config.epochs):
            # Training phase
            model.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = model(X_batch)

                if task_type == "regression":
                    output = output.squeeze()

                loss = criterion(output, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(loss.item())

            # Validation phase
            model.eval()
            val_losses = []
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    output = model(X_batch)
                    if task_type == "regression":
                        output = output.squeeze(-1)  # Only squeeze last dim to preserve batch

                    # Ensure output has at least 1 dimension for extend
                    if output.dim() == 0:
                        output = output.unsqueeze(0)
                    if y_batch.dim() == 0:
                        y_batch = y_batch.unsqueeze(0)

                    loss = criterion(output, y_batch)
                    val_losses.append(loss.item())

                    # Convert to list for extend
                    pred_np = output.cpu().numpy()
                    target_np = y_batch.cpu().numpy()
                    if pred_np.ndim == 0:
                        all_preds.append(pred_np.item())
                    else:
                        all_preds.extend(pred_np.tolist())
                    if target_np.ndim == 0:
                        all_targets.append(target_np.item())
                    else:
                        all_targets.extend(target_np.tolist())

            # Calculate metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            metrics = self._compute_metrics(np.array(all_targets), np.array(all_preds), task_type)

            # Update history
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_metrics"].append(metrics)

            # Learning rate scheduling
            if scheduler:
                if config.scheduler == "plateau":
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # Early stopping check
            if avg_val_loss < best_val_loss - config.min_delta:
                best_val_loss = avg_val_loss
                best_metrics = metrics
                best_epoch = epoch
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # Progress logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | "
                      f"Val {main_metric}: {metrics[main_metric]:.4f}")

            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state:
            model.load_state_dict(best_state)

        training_time = time.time() - start_time

        return {
            "best_metrics": best_metrics,
            "best_epoch": best_epoch + 1,
            "training_time": training_time,
            "final_val_loss": best_val_loss,
            "history": self.history
        }

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         task_type: str) -> Dict[str, float]:
        """Compute appropriate metrics"""
        metrics = {}

        if task_type == "regression":
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        else:
            if len(y_pred.shape) > 1:
                y_pred = np.argmax(y_pred, axis=1)

            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1

        return metrics


# ============================================================================
# VISUALIZATIONS
# ============================================================================

class Visualizer:
    """Creates training visualizations"""

    def __init__(self, output_dir: str = "nn_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def plot_training_history(self, histories: Dict[str, Dict], task_type: str,
                              save_path: str = None) -> str:
        """Plot training curves for all models"""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Neural Network Training Results', fontsize=14, fontweight='bold')

        colors = plt.cm.Set2(np.linspace(0, 1, len(histories)))

        # Plot 1: Loss curves
        ax = axes[0, 0]
        for (name, hist), color in zip(histories.items(), colors):
            ax.plot(hist["train_loss"], label=f'{name} (train)', color=color, alpha=0.7)
            ax.plot(hist["val_loss"], label=f'{name} (val)', color=color, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Main metric
        ax = axes[0, 1]
        metric_name = 'accuracy' if task_type == "classification" else 'r2'

        for (name, hist), color in zip(histories.items(), colors):
            metric_vals = [m[metric_name] for m in hist["val_metrics"]]
            ax.plot(metric_vals, label=name, color=color, linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f'Validation {metric_name.upper()} Over Time')
        ax.legend(loc='lower right' if task_type == "regression" else 'lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 3: Final metrics comparison
        ax = axes[1, 0]
        model_names = list(histories.keys())
        final_metrics = {name: hist["val_metrics"][-1] for name, hist in histories.items()}
        metric_keys = list(final_metrics[model_names[0]].keys())

        x = np.arange(len(metric_keys))
        width = 0.8 / len(model_names)

        for i, (name, metrics) in enumerate(final_metrics.items()):
            values = [metrics[k] for k in metric_keys]
            ax.bar(x + i * width, values, width, label=name, alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Final Metrics Comparison')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels([k.upper() for k in metric_keys], rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Summary table
        ax = axes[1, 1]
        ax.axis('off')

        # Create summary text
        summary = "MODEL PERFORMANCE SUMMARY\n" + "=" * 35 + "\n\n"

        best_model = None
        best_score = -float('inf')

        for name, metrics in final_metrics.items():
            score = metrics[metric_name]
            summary += f"{name.upper()}:\n"
            for k, v in metrics.items():
                summary += f"  {k:12s}: {v:.4f}\n"
            summary += "\n"

            if score > best_score:
                best_score = score
                best_model = name

        summary += "-" * 35 + f"\nWINNER: {best_model.upper()} ({metric_name}: {best_score:.4f})"

        ax.text(0.1, 0.5, summary, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.output_dir / f"training_results_{timestamp}.png")

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[Visualizer] Saved training plot: {save_path}")
        return save_path

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                         task_type: str, model_name: str = "Model",
                         save_path: str = None) -> str:
        """Plot predictions vs actuals"""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{model_name} Predictions', fontsize=12, fontweight='bold')

        if task_type == "regression":
            # Scatter plot: predicted vs actual
            ax = axes[0]
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')

            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Predicted vs Actual')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Residual distribution
            ax = axes[1]
            residuals = y_pred - y_true
            ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', label='Zero Error')
            ax.set_xlabel('Residual (Predicted - Actual)')
            ax.set_ylabel('Frequency')
            ax.set_title('Residual Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:  # Classification
            # Confusion matrix
            ax = axes[0]
            if len(y_pred.shape) > 1:
                y_pred = np.argmax(y_pred, axis=1)

            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')

            # Class distribution
            ax = axes[1]
            unique, counts = np.unique(y_pred, return_counts=True)
            ax.bar(unique, counts, alpha=0.7)
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Prediction Distribution')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.output_dir / f"predictions_{model_name}_{timestamp}.png")

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[Visualizer] Saved predictions plot: {save_path}")
        return save_path


# ============================================================================
# HYPERPARAMETER TUNER
# ============================================================================

class HyperparameterTuner:
    """
    Automated hyperparameter optimization using Optuna.

    Supports tuning for all model types (MLP, LSTM, CNN, Transformer).
    """

    def __init__(self, device: torch.device = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params = None
        self.study = None

    def tune(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             model_type: str, task_type: str,
             n_trials: int = 50,
             timeout: int = None,
             n_jobs: int = 1) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_type: One of 'mlp', 'lstm', 'cnn', 'transformer'
            task_type: 'classification' or 'regression'
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds (optional)
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary with best parameters and study results
        """
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING - {model_type.upper()}")
        print(f"{'='*60}")
        print(f"Trials: {n_trials}")
        print(f"Task: {task_type}")
        print(f"Device: {self.device}")

        is_sequential = model_type in ['lstm', 'cnn', 'transformer']
        input_size = X_train.shape[-1]
        num_classes = len(np.unique(y_train)) if task_type == 'classification' else 1

        # Create objective function
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters based on model type
            params = self._suggest_params(trial, model_type, input_size)

            try:
                # Create model config
                config = ModelConfig(
                    model_type=model_type,
                    input_size=input_size,
                    output_size=num_classes,
                    hidden_sizes=params['hidden_sizes'],
                    dropout=params['dropout'],
                    num_layers=params['num_layers'],
                    num_heads=params.get('num_heads', 4),
                    seq_length=params.get('seq_length', 20),
                    task_type=task_type
                )

                # Create training config
                train_config = TrainingConfig(
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    learning_rate=params['learning_rate'],
                    patience=10,
                    weight_decay=params['weight_decay'],
                    scheduler=params['scheduler']
                )

                # Prepare data
                if is_sequential:
                    seq_len = params.get('seq_length', 20)
                    if len(X_train) <= seq_len:
                        return float('inf')

                    X_tr_seq = self._make_sequences(X_train, seq_len)
                    y_tr_seq = y_train[seq_len:]
                    X_va_seq = self._make_sequences(X_val, seq_len)
                    y_va_seq = y_val[seq_len:]

                    train_ds = NNDataset(X_tr_seq, y_tr_seq, task_type)
                    val_ds = NNDataset(X_va_seq, y_va_seq, task_type)
                else:
                    train_ds = NNDataset(X_train, y_train, task_type)
                    val_ds = NNDataset(X_val, y_val, task_type)

                train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)

                # Create and train model
                model = self._create_model(config)
                trainer = Trainer(self.device)
                trainer.history = {"train_loss": [], "val_loss": [], "val_metrics": []}

                result = trainer.train(model, train_loader, val_loader, train_config, task_type)

                # Return metric to optimize (minimize loss / maximize accuracy)
                if task_type == 'classification':
                    return 1 - result['best_metrics']['accuracy']  # Minimize 1-accuracy
                else:
                    return -result['best_metrics']['r2']  # Minimize negative R2

            except Exception as e:
                print(f"  [Trial {trial.number}] Failed: {e}")
                return float('inf')

        # Create and run study
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            direction='minimize',
            study_name=f'{model_type}_tuning',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        # Get best parameters
        self.best_params = self.study.best_params
        best_value = self.study.best_value

        # Convert back to actual metric
        if task_type == 'classification':
            best_metric = 1 - best_value
            metric_name = 'accuracy'
        else:
            best_metric = -best_value
            metric_name = 'r2'

        print(f"\n{'='*60}")
        print("TUNING COMPLETE")
        print(f"{'='*60}")
        print(f"Best {metric_name}: {best_metric:.4f}")
        print(f"\nBest Parameters:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")

        return {
            'best_params': self.best_params,
            'best_metric': best_metric,
            'metric_name': metric_name,
            'n_trials': n_trials,
            'completed_trials': len(self.study.trials),
            'study': self.study
        }

    def _suggest_params(self, trial: Trial, model_type: str, input_size: int) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial"""

        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'epochs': trial.suggest_int('epochs', 30, 150),
            'scheduler': trial.suggest_categorical('scheduler', ['plateau', 'cosine', 'none']),
        }

        if model_type == 'mlp':
            n_layers = trial.suggest_int('n_hidden_layers', 1, 4)
            hidden_sizes = []
            for i in range(n_layers):
                size = trial.suggest_categorical(f'hidden_size_{i}', [32, 64, 128, 256, 512])
                hidden_sizes.append(size)
            params['hidden_sizes'] = hidden_sizes
            params['num_layers'] = n_layers

        elif model_type == 'lstm':
            params['hidden_sizes'] = [trial.suggest_categorical('lstm_hidden', [32, 64, 128, 256])]
            params['num_layers'] = trial.suggest_int('lstm_layers', 1, 3)
            params['seq_length'] = trial.suggest_int('seq_length', 10, 50)

        elif model_type == 'cnn':
            params['hidden_sizes'] = [64]  # CNN has fixed architecture
            params['num_layers'] = 3
            params['seq_length'] = trial.suggest_int('seq_length', 10, 50)

        elif model_type == 'transformer':
            d_model = trial.suggest_categorical('d_model', [32, 64, 128])
            params['hidden_sizes'] = [d_model]
            params['num_layers'] = trial.suggest_int('transformer_layers', 1, 4)
            params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
            params['seq_length'] = trial.suggest_int('seq_length', 10, 50)

        return params

    def _make_sequences(self, X: np.ndarray, seq_length: int) -> np.ndarray:
        """Create sequences for temporal models"""
        Xs = []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:i + seq_length])
        return np.array(Xs)

    def _create_model(self, config: ModelConfig) -> nn.Module:
        """Create model instance from config"""
        model_classes = {
            "mlp": FlexibleMLP,
            "lstm": FlexibleLSTM,
            "cnn": FlexibleCNN,
            "transformer": FlexibleTransformer
        }
        return model_classes[config.model_type](config)

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        if self.study is None:
            return None

        trials_data = []
        for trial in self.study.trials:
            data = {
                'trial': trial.number,
                'value': trial.value,
                'state': trial.state.name,
            }
            data.update(trial.params)
            trials_data.append(data)

        return pd.DataFrame(trials_data)

    def plot_optimization_history(self, output_path: str = None) -> str:
        """Plot optimization history"""
        if self.study is None:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Optimization history
        ax = axes[0]
        trials = [t for t in self.study.trials if t.value is not None and t.value != float('inf')]
        values = [t.value for t in trials]
        best_values = np.minimum.accumulate(values)

        ax.plot(values, 'o-', alpha=0.5, label='Trial Value')
        ax.plot(best_values, 'r-', linewidth=2, label='Best Value')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Parameter importance (if enough trials)
        ax = axes[1]
        if len(trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(self.study)
                params = list(importance.keys())[:10]
                values = [importance[p] for p in params]

                ax.barh(params, values, color='steelblue')
                ax.set_xlabel('Importance')
                ax.set_title('Parameter Importance')
            except:
                ax.text(0.5, 0.5, 'Not enough trials\nfor importance analysis',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Parameter Importance')
        else:
            ax.text(0.5, 0.5, 'Not enough trials\nfor importance analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter Importance')

        plt.tight_layout()

        if output_path is None:
            output_path = f"tuning_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Tuner] Saved optimization history: {output_path}")
        return output_path


# ============================================================================
# MAIN NEURAL NETWORK BUILDER
# ============================================================================

class NeuralNetworkBuilder:
    """
    Streamlined neural network builder for training and evaluation.

    Supports:
    - MLP, LSTM, CNN, Transformer architectures
    - Automatic task detection (classification/regression)
    - Easy model saving/loading
    - Rich visualizations
    """

    def __init__(self, memory_palace=None, output_dir: str = "nn_experiments"):
        self.memory = memory_palace
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.registry = ModelRegistry(str(self.output_dir / "models"))
        self.visualizer = Visualizer(str(self.output_dir))

        print(f"[NNBuilder] Initialized | Device: {self.device}")

    # ========================================================================
    # QUICK TRAINING API
    # ========================================================================

    def quick_train(self, dataset_path: str, target: str = None,
                    models: List[str] = None, epochs: int = 50) -> Dict:
        """
        Quick training with sensible defaults.

        Args:
            dataset_path: Path to CSV file
            target: Target column name (auto-detected if None)
            models: List of models to train (default: ["mlp", "lstm"])
            epochs: Number of training epochs

        Returns:
            Dictionary with results, winner, and visualization path
        """
        spec = {
            "dataset": {"path": dataset_path},
            "models": models or ["mlp", "lstm"],
            "epochs": epochs
        }

        if target:
            spec["dataset"]["target_column"] = target

        return self.run_experiment(spec)

    # ========================================================================
    # FULL EXPERIMENT API
    # ========================================================================

    def run_experiment(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete neural network experiment.

        Spec format:
        {
            "dataset": {
                "path": "path/to/data.csv",
                "target_column": "target_name"  # optional
            },
            "models": ["mlp", "lstm", "cnn", "transformer"],  # optional
            "epochs": 100,  # optional
            "batch_size": 32,  # optional
            "sequence_length": 20,  # optional, for LSTM/CNN/Transformer
            "learning_rate": 0.001,  # optional
            "save_models": True  # optional
        }
        """

        print("\n" + "=" * 70)
        print("NEURAL NETWORK EXPERIMENT")
        print("=" * 70)

        # Load and preprocess data
        dataset_path = spec["dataset"]["path"]
        target_column = spec["dataset"].get("target_column")

        print(f"\n[1/4] Loading dataset: {dataset_path}")

        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            return {"error": f"Failed to load dataset: {e}"}

        processor = DataProcessor()
        X, y = processor.fit_transform(df, target_column)

        task_type = processor.task_type
        num_features = X.shape[1]
        num_classes = processor.num_classes

        print(f"    Task Type: {task_type}")
        print(f"    Features: {num_features}")
        print(f"    Classes/Output: {num_classes}")
        print(f"    Samples: {len(X)}")
        print(f"    Target: {processor.target_name}")

        # Train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=(task_type == "classification")
        )

        # Configuration
        models_to_train = spec.get("models", ["mlp", "lstm"])
        epochs = spec.get("epochs", 100)
        batch_size = spec.get("batch_size", 32)
        seq_length = spec.get("sequence_length", 20)
        learning_rate = spec.get("learning_rate", 1e-3)
        save_models = spec.get("save_models", True)

        training_config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        print(f"\n[2/4] Training Configuration:")
        print(f"    Models: {', '.join(m.upper() for m in models_to_train)}")
        print(f"    Epochs: {epochs}")
        print(f"    Batch Size: {batch_size}")
        print(f"    Learning Rate: {learning_rate}")

        # Train each model
        results = {}
        histories = {}
        trained_models = {}

        print(f"\n[3/4] Training Models...")

        for model_type in models_to_train:
            print(f"\n{'─' * 50}")
            print(f"Training {model_type.upper()}")
            print(f"{'─' * 50}")

            try:
                # Prepare data for model type
                is_sequential = model_type in ["lstm", "cnn", "transformer"]

                if is_sequential:
                    if len(X_train) <= seq_length:
                        print(f"[Skip] Dataset too small for {model_type}")
                        continue

                    X_tr_seq, y_tr_seq = processor.make_sequences(X_train, y_train, seq_length)
                    X_va_seq, y_va_seq = processor.make_sequences(X_val, y_val, seq_length)

                    train_ds = NNDataset(X_tr_seq, y_tr_seq, task_type)
                    val_ds = NNDataset(X_va_seq, y_va_seq, task_type)

                    input_size = num_features
                else:
                    train_ds = NNDataset(X_train, y_train, task_type)
                    val_ds = NNDataset(X_val, y_val, task_type)
                    input_size = num_features

                # Create data loaders
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

                # Create model config
                model_config = ModelConfig(
                    model_type=model_type,
                    input_size=input_size,
                    output_size=num_classes,
                    hidden_sizes=[128, 64] if model_type == "mlp" else [64],
                    dropout=0.2,
                    num_layers=2,
                    num_heads=4,
                    seq_length=seq_length,
                    task_type=task_type
                )

                # Create model
                model = self._create_model(model_config)

                # Train
                trainer = Trainer(self.device)
                train_result = trainer.train(
                    model, train_loader, val_loader,
                    training_config, task_type
                )

                results[model_type] = train_result["best_metrics"]
                histories[model_type] = train_result["history"]
                trained_models[model_type] = (model, model_config, processor)

                print(f"\nBest {model_type.upper()} Metrics (epoch {train_result['best_epoch']}):")
                for k, v in train_result["best_metrics"].items():
                    print(f"  {k:12s}: {v:.4f}")

            except Exception as e:
                print(f"[Error] Failed to train {model_type}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not results:
            return {"error": "No models trained successfully"}

        # Determine winner
        metric_key = 'accuracy' if task_type == "classification" else 'r2'
        winner = max(results.items(), key=lambda x: x[1][metric_key])

        print(f"\n{'=' * 50}")
        print(f"WINNER: {winner[0].upper()} ({metric_key}: {winner[1][metric_key]:.4f})")
        print(f"{'=' * 50}")

        # Save models
        saved_models = {}
        if save_models:
            print(f"\n[4/4] Saving Models...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for model_type, (model, config, proc) in trained_models.items():
                model_name = f"{timestamp}_{model_type}"
                self.registry.save_model(model, config, proc, results[model_type], model_name)
                saved_models[model_type] = model_name

        # Create visualizations
        viz_path = self.visualizer.plot_training_history(histories, task_type)

        # Store in memory if available
        if self.memory:
            self.memory.add_node(
                f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                {
                    "type": "nn_experiment",
                    "dataset": dataset_path,
                    "winner": winner[0],
                    "score": winner[1][metric_key],
                    "task_type": task_type
                }
            )

        return {
            "task_type": task_type,
            "results": results,
            "winner": winner[0],
            "winner_score": winner[1][metric_key],
            "visualization": viz_path,
            "saved_models": saved_models,
            "histories": histories
        }

    def _create_model(self, config: ModelConfig) -> nn.Module:
        """Create model instance from config"""
        model_classes = {
            "mlp": FlexibleMLP,
            "lstm": FlexibleLSTM,
            "cnn": FlexibleCNN,
            "transformer": FlexibleTransformer
        }

        if config.model_type not in model_classes:
            raise ValueError(f"Unknown model type: {config.model_type}")

        return model_classes[config.model_type](config)

    # ========================================================================
    # MODEL LOADING & PREDICTION API
    # ========================================================================

    def load_model(self, name: str) -> Tuple[nn.Module, ModelConfig, DataProcessor]:
        """Load a saved model by name"""
        return self.registry.load_model(name)

    def list_models(self) -> List[Dict]:
        """List all saved models"""
        models = self.registry.list_models()

        if not models:
            print("[NNBuilder] No saved models found.")
            return []

        print("\nSaved Models:")
        print("-" * 70)
        for m in models:
            print(f"  {m['name']:30s} | {m['model_type']:12s} | {m['task_type']:15s}")
        print("-" * 70)

        return models

    def predict(self, model_name: str, data: Union[str, pd.DataFrame, np.ndarray],
                return_original_scale: bool = True) -> np.ndarray:
        """
        Make predictions with a saved model.

        Args:
            model_name: Name of saved model
            data: Path to CSV, DataFrame, or numpy array
            return_original_scale: Whether to inverse transform predictions

        Returns:
            Predictions array
        """
        model, config, processor = self.load_model(model_name)
        model = model.to(self.device)
        model.eval()

        # Load data if path
        if isinstance(data, str):
            data = pd.read_csv(data)

        # Transform data
        if isinstance(data, pd.DataFrame):
            X = processor.transform(data)
        else:
            X = data

        # Create sequences if needed
        is_sequential = config.model_type in ["lstm", "cnn", "transformer"]
        if is_sequential:
            # For prediction, use last seq_length samples
            X = X[-config.seq_length:].reshape(1, config.seq_length, -1)

        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = model(X_tensor).cpu().numpy()

        # Handle output
        if config.task_type == "classification":
            if len(predictions.shape) > 1:
                predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions.squeeze()

        # Inverse transform if requested
        if return_original_scale:
            predictions = processor.inverse_transform_y(predictions)

        return predictions

    def delete_model(self, name: str):
        """Delete a saved model"""
        self.registry.delete_model(name)

    # ========================================================================
    # HYPERPARAMETER TUNING API
    # ========================================================================

    def tune_hyperparameters(self, dataset_path: str, model_type: str,
                             target: str = None, n_trials: int = 50,
                             timeout: int = None) -> Dict[str, Any]:
        """
        Automatically tune hyperparameters for a model type.

        Args:
            dataset_path: Path to CSV file
            model_type: One of 'mlp', 'lstm', 'cnn', 'transformer'
            target: Target column name (auto-detected if None)
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds (optional)

        Returns:
            Dictionary with best parameters, metrics, and trained model
        """
        if not OPTUNA_AVAILABLE:
            return {"error": "Optuna not installed. Install with: pip install optuna"}

        print(f"\n{'='*70}")
        print("HYPERPARAMETER TUNING")
        print(f"{'='*70}")

        # Load and preprocess data
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            return {"error": f"Failed to load dataset: {e}"}

        processor = DataProcessor()
        X, y = processor.fit_transform(df, target)

        task_type = processor.task_type
        print(f"[Tuning] Dataset: {dataset_path}")
        print(f"[Tuning] Task Type: {task_type}")
        print(f"[Tuning] Model: {model_type.upper()}")
        print(f"[Tuning] Trials: {n_trials}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42,
            shuffle=(task_type == "classification")
        )

        # Run tuning
        tuner = HyperparameterTuner(self.device)
        tuning_result = tuner.tune(
            X_train, y_train, X_val, y_val,
            model_type=model_type,
            task_type=task_type,
            n_trials=n_trials,
            timeout=timeout
        )

        # Save tuning visualization
        viz_path = tuner.plot_optimization_history(
            str(self.output_dir / f"tuning_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        )
        tuning_result['visualization'] = viz_path

        # Train final model with best params
        print("\n[Tuning] Training final model with best parameters...")
        best_params = tuning_result['best_params']

        # Build spec from best params
        spec = {
            "dataset": {"path": dataset_path},
            "models": [model_type],
            "epochs": best_params.get('epochs', 100),
            "batch_size": best_params.get('batch_size', 32),
            "learning_rate": best_params.get('learning_rate', 1e-3),
            "sequence_length": best_params.get('seq_length', 20),
            "save_models": True
        }

        if target:
            spec["dataset"]["target_column"] = target

        # Run final training
        final_result = self.run_experiment(spec)

        tuning_result['final_training'] = final_result
        tuning_result['saved_model'] = final_result.get('saved_models', {}).get(model_type)

        return tuning_result


# ============================================================================
# CONVENIENCE FUNCTION FOR ORCHESTRATOR
# ============================================================================

def create_nn_builder(memory_palace=None, output_dir: str = "nn_experiments") -> NeuralNetworkBuilder:
    """Factory function to create NeuralNetworkBuilder"""
    return NeuralNetworkBuilder(memory_palace, output_dir)

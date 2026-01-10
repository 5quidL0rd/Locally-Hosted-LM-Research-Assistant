# agents/nn_builder_agent.py

import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# ============================
# DATASET
# ============================

class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================
# NN BUILDER
# ============================

class NeuralNetworkBuilder:

    def __init__(self, memory_palace, output_dir="nn_experiments"):
        self.memory = memory_palace
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NNBuilder] Using device: {self.device}")

    # ============================
    # DATA LOADING
    # ============================

    def load_dataset(self, dataset_path, target_column="Close"):
        # Handle directory paths
        if os.path.isdir(dataset_path):
            csvs = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
            if not csvs:
                return None, "No CSV found"
            dataset_path = os.path.join(dataset_path, csvs[0])

        df = pd.read_csv(dataset_path)

        # Sort by time if possible
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

        # Use log returns (stable regression target)
        df["log_return"] = np.log(df[target_column] / df[target_column].shift(1))
        df = df.dropna()

        y = df["log_return"].values

        X = df.drop(columns=["log_return", target_column, "Date"], errors="ignore")

        # Encode categoricals
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        scaler = StandardScaler()
        X = scaler.fit_transform(X.values)

        return {
            "X": X,
            "y": y,
            "num_features": X.shape[1],
            "scaler": scaler
        }, None

    # ============================
    # MODELS
    # ============================

    def create_mlp(self, input_size):
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def create_lstm(self, input_size):
        class LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, 64, batch_first=True)
                self.fc = nn.Linear(64, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        return LSTM()

    # ============================
    # SEQUENCING
    # ============================

    def make_sequences(self, X, y, seq_len=20):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)

    # ============================
    # TRAINING
    # ============================

    def train_model(self, model, train_ds, val_ds, epochs=50):
        loader_tr = DataLoader(train_ds, batch_size=32, shuffle=False)
        loader_va = DataLoader(val_ds, batch_size=32)

        model.to(self.device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        best = float("inf")
        patience = 10
        wait = 0

        history = {"val_metric": []}

        for epoch in range(epochs):
            model.train()
            for X, y in loader_tr:
                X, y = X.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = loss_fn(model(X).squeeze(), y)
                loss.backward()
                opt.step()

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for X, y in loader_va:
                    X, y = X.to(self.device), y.to(self.device)
                    p = model(X).squeeze()
                    preds.extend(p.cpu().numpy())
                    trues.extend(y.cpu().numpy())

            r2 = r2_score(trues, preds)
            history["val_metric"].append(r2)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Val R²: {r2:.4f}")

            if loss.item() < best:
                best = loss.item()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"[NNBuilder] Early stopping at epoch {epoch+1}")
                    break

        return history

    # ============================
    # RUN EXPERIMENT
    # ============================

    def run_experiment(self, spec):
        data_info, err = self.load_dataset(spec["dataset"]["path"])
        if err:
            return {"error": err}

        X, y = data_info["X"], data_info["y"]

        split = int(len(X) * 0.8)

        # ===== MLP =====
        X_tr, X_va = X[:split], X[split:]
        y_tr, y_va = y[:split], y[split:]

        mlp = self.create_mlp(X.shape[1])
        hist_mlp = self.train_model(
            mlp,
            TensorDataset(X_tr, y_tr),
            TensorDataset(X_va, y_va)
        )
        mlp_score = hist_mlp["val_metric"][-1]

        # ===== LSTM =====
        X_seq, y_seq = self.make_sequences(X, y)
        split = int(len(X_seq) * 0.8)

        lstm = self.create_lstm(X.shape[1])
        hist_lstm = self.train_model(
            lstm,
            TensorDataset(X_seq[:split], y_seq[:split]),
            TensorDataset(X_seq[split:], y_seq[split:])
        )
        lstm_score = hist_lstm["val_metric"][-1]

        winner = "lstm" if lstm_score > mlp_score else "mlp"

        print(f"[NNBuilder] MLP final score: {mlp_score:.4f}")
        print(f"[NNBuilder] LSTM final score: {lstm_score:.4f}")
        print(f"Winner: {winner.upper()}")

        return {
            "mlp_score": mlp_score,
            "lstm_score": lstm_score,
            "winner": winner
        }

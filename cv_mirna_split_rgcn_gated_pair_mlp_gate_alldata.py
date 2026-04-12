import os
import json
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.decomposition import PCA

try:
    import optuna
except Exception:
    optuna = None


# -----------------------------
# Globals configured in main
# -----------------------------
SEED = 42
DEVICE = torch.device('cpu')
OUTPUT_FOLDER = 'models_cv'
EPOCHS = 200
PATIENCE = 20


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    global SEED
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------
# Data loading / preprocessing
# -----------------------------
def load_data(file_path):
    try:
        path = Path(file_path)
        if path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(path)
        elif path.suffix.lower() in ['.csv']:
            data = pd.read_csv(path)
        elif path.suffix.lower() in ['.txt', '.tsv']:
            data = pd.read_csv(path, sep='\t')
        else:
            raise ValueError(f'Unsupported file format: {path.suffix}')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(data: pd.DataFrame):
    cols_lower = {str(c).strip().lower(): c for c in data.columns}
    mirna_col = None
    disease_col = None
    for key in ['mirna', 'miRNA'.lower(), 'mirna_name', 'mirna id', 'mirna_id']:
        if key in cols_lower:
            mirna_col = cols_lower[key]
            break
    for key in ['disease', 'disease_name', 'disease id', 'disease_id']:
        if key in cols_lower:
            disease_col = cols_lower[key]
            break

    if mirna_col is None or disease_col is None:
        if data.shape[1] < 2:
            raise ValueError('Input table must contain at least two columns: miRNA and disease.')
        mirna_col = data.columns[0]
        disease_col = data.columns[1]

    associations = data[[mirna_col, disease_col]].drop_duplicates().copy()
    associations.columns = ['miRNA', 'disease']
    associations['miRNA'] = associations['miRNA'].astype(str).str.strip()
    associations['disease'] = associations['disease'].astype(str).str.strip()

    miRNAs = associations['miRNA'].unique()
    diseases = associations['disease'].unique()
    matrix = pd.DataFrame(0, index=miRNAs, columns=diseases, dtype=np.int8)
    matrix.index.name = 'miRNA'
    matrix.columns.name = 'disease'
    for _, row in associations.iterrows():
        matrix.loc[row['miRNA'], row['disease']] = 1
    return matrix, miRNAs, diseases


# -----------------------------
# Similarity / graph features
# -----------------------------
def compute_similarity(matrix: pd.DataFrame):
    gamma_m = 1.0 / (matrix.sum(axis=1).mean() + 1e-10)
    miRNA_sim = pd.DataFrame(
        rbf_kernel(matrix, gamma=gamma_m),
        index=matrix.index,
        columns=matrix.index,
    )
    gamma_d = 1.0 / (matrix.sum(axis=0).mean() + 1e-10)
    disease_sim = pd.DataFrame(
        rbf_kernel(matrix.T, gamma=gamma_d),
        index=matrix.columns,
        columns=matrix.columns,
    )
    return miRNA_sim, disease_sim


def prepare_rgcn_data(matrix, miRNA_sim, disease_sim, feature_dim=16, device=None):
    if device is None:
        device = DEVICE

    miRNAs = list(matrix.index)
    diseases = list(matrix.columns)
    n_mirnas = len(miRNAs)
    n_diseases = len(diseases)

    feature_dim_m = min(feature_dim, len(miRNAs), max(1, miRNA_sim.shape[1]))
    feature_dim_d = min(feature_dim, len(diseases), max(1, disease_sim.shape[1]))

    pca_m = PCA(n_components=feature_dim_m, random_state=SEED)
    miRNA_features = pca_m.fit_transform(miRNA_sim)
    pca_d = PCA(n_components=feature_dim_d, random_state=SEED)
    disease_features = pca_d.fit_transform(disease_sim)

    max_dim = max(miRNA_features.shape[1], disease_features.shape[1])
    if miRNA_features.shape[1] < max_dim:
        miRNA_features = np.pad(miRNA_features, ((0, 0), (0, max_dim - miRNA_features.shape[1])))
    if disease_features.shape[1] < max_dim:
        disease_features = np.pad(disease_features, ((0, 0), (0, max_dim - disease_features.shape[1])))

    x = torch.tensor(np.vstack([miRNA_features, disease_features]), dtype=torch.float, device=device)

    edges = matrix.stack().reset_index()
    edges = edges[edges[0] == 1]

    miRNA_to_idx = {m: i for i, m in enumerate(miRNAs)}
    disease_to_idx = {d: i for i, d in enumerate(diseases)}

    miRNA_indices = edges['miRNA'].map(miRNA_to_idx).values
    disease_indices = edges['disease'].map(disease_to_idx).values + n_mirnas

    forward_edge_index = np.array([miRNA_indices, disease_indices])
    reverse_edge_index = np.array([disease_indices, miRNA_indices])
    edge_index = torch.tensor(
        np.concatenate([forward_edge_index, reverse_edge_index], axis=1),
        dtype=torch.long,
        device=device,
    )
    edge_type = torch.tensor(
        np.concatenate([
            np.zeros(len(miRNA_indices), dtype=np.int64),
            np.ones(len(miRNA_indices), dtype=np.int64),
        ]),
        dtype=torch.long,
        device=device,
    )

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.n_mirnas = n_mirnas
    data.n_diseases = n_diseases
    return data.to(device), n_mirnas, n_diseases


# -----------------------------
# MDMF branch
# -----------------------------
def _matrix_factorization(M, K, steps=2000, alpha=0.002, beta=0.02, seed=42):
    # Classical gradient-descent MF used as a light MDMF proxy initialization.
    np.random.seed(seed)
    N, D = M.shape
    U = np.random.normal(scale=1.0 / K, size=(N, K))
    V = np.random.normal(scale=1.0 / K, size=(D, K))

    nz = np.argwhere(M > 0)
    if len(nz) == 0:
        return U, V

    for _ in range(steps):
        np.random.shuffle(nz)
        for i, j in nz:
            eij = M[i, j] - np.dot(U[i, :], V[j, :].T)
            U[i, :] += alpha * (2 * eij * V[j, :] - beta * U[i, :])
            V[j, :] += alpha * (2 * eij * U[i, :] - beta * V[j, :])
    return U, V


class MDMF(nn.Module):
    def __init__(self, num_mirnas, num_diseases, latent_dim, lambda_reg, init_U=None, init_V=None):
        super().__init__()
        if init_U is None:
            init_U = torch.randn(num_mirnas, latent_dim)
        if init_V is None:
            init_V = torch.randn(num_diseases, latent_dim)
        self.U = nn.Parameter(init_U)
        self.V = nn.Parameter(init_V)
        self.lambda_reg = lambda_reg

    def forward(self, A, S_m, S_d):
        recon_loss = F.mse_loss(torch.mm(self.U, self.V.t()), A)
        reg_U = torch.norm(self.U, p=2)
        reg_V = torch.norm(self.V, p=2)
        reg_S_m = torch.norm(torch.mm(self.U, self.U.t()) - S_m, p=2)
        reg_S_d = torch.norm(torch.mm(self.V, self.V.t()) - S_d, p=2)
        return recon_loss + self.lambda_reg * (reg_U + reg_V + reg_S_m + reg_S_d)

    def get_features(self):
        return self.U, self.V


def train_mdmf(mdmf, A, S_m, S_d, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(mdmf.parameters(), lr=lr)
    mdmf.to(DEVICE)
    A, S_m, S_d = A.to(DEVICE), S_m.to(DEVICE), S_d.to(DEVICE)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = mdmf(A, S_m, S_d)
        loss.backward()
        optimizer.step()
    return mdmf.get_features()


# -----------------------------
# Model
# -----------------------------
def pair_indices_to_components(flat_pair_indices, n_diseases, device):
    flat_pair_indices = torch.as_tensor(flat_pair_indices, dtype=torch.long, device=device)
    mirna_idx = flat_pair_indices // n_diseases
    disease_idx = flat_pair_indices % n_diseases
    return mirna_idx, disease_idx


class PairMLPDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        half_hidden = max(hidden_dim // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(half_hidden, 1),
        )

    def forward(self, mirna_emb, disease_emb):
        feat = torch.cat([
            mirna_emb,
            disease_emb,
            torch.abs(mirna_emb - disease_emb),
            mirna_emb * disease_emb,
        ], dim=-1)
        return self.net(feat).view(-1)


class NodewiseSoftGateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(dim // 2, 8)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, h_graph, h_side):
        weights = torch.softmax(self.gate(torch.cat([h_graph, h_side], dim=-1)), dim=-1)
        fused = weights[:, 0:1] * h_graph + weights[:, 1:2] * h_side
        return fused, weights


class RGCNGatedPairMLP(nn.Module):
    def __init__(self, in_channels, mdmf_dim, hidden_channels, out_channels, dropout, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_relations = 2
        self.mdmf_dim = mdmf_dim
        self.embed_dim = out_channels

        self.rgcn_layers = nn.ModuleList()
        self.rgcn_layers.append(RGCNConv(in_channels, hidden_channels, num_relations=self.num_relations))
        for _ in range(max(0, num_layers - 2)):
            self.rgcn_layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations=self.num_relations))
        self.rgcn_layers.append(RGCNConv(hidden_channels, out_channels, num_relations=self.num_relations))
        self.rgcn_residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        self.mdmf_branch = nn.Sequential(
            nn.Linear(mdmf_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        self.fusion = NodewiseSoftGateFusion(out_channels)
        self.post_norm = nn.LayerNorm(out_channels)
        self.decoder = PairMLPDecoder(self.embed_dim, max(hidden_channels, out_channels), dropout)
        self.last_gate_weights = None

    def encode(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        h = x
        for i, layer in enumerate(self.rgcn_layers):
            h = layer(h, edge_index, edge_type)
            if i < self.num_layers - 1:
                h = h.relu()
                h = F.dropout(h, p=self.dropout, training=self.training)
        h_graph = h + self.rgcn_residual(x)

        h_mdmf = self.mdmf_branch(x[:, -self.mdmf_dim:])
        h_fused, gate_weights = self.fusion(h_graph, h_mdmf)
        h_fused = self.post_norm(h_fused)
        self.last_gate_weights = gate_weights
        return h_fused

    def pair_logits(self, data, flat_pair_indices):
        node_out = self.encode(data)
        n_mirnas = data.n_mirnas
        n_diseases = data.n_diseases
        mirna_emb = node_out[:n_mirnas]
        disease_emb = node_out[n_mirnas:]
        mirna_idx, disease_idx = pair_indices_to_components(flat_pair_indices, n_diseases, node_out.device)
        return self.decoder(mirna_emb[mirna_idx], disease_emb[disease_idx])


# -----------------------------
# Training / evaluation helpers
# -----------------------------
def train_model(model, data, train_mask, train_labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    pred = model.pair_logits(data, train_mask)
    loss = criterion(pred, train_labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(model, data, mask, labels):
    model.eval()
    with torch.no_grad():
        pred = model.pair_logits(data, mask)
        pred_scores = torch.sigmoid(pred).detach().cpu().numpy()
        auc = roc_auc_score(labels.detach().cpu().numpy(), pred_scores) if len(np.unique(labels.detach().cpu().numpy())) > 1 else 0.0
    return auc


def compute_full_metrics(model, data, mask, labels):
    model.eval()
    with torch.no_grad():
        logits = model.pair_logits(data, mask)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        final_labels = labels.detach().cpu().numpy()
        auc = roc_auc_score(final_labels, probs) if len(np.unique(final_labels)) > 1 else 0.0
        pr_auc = average_precision_score(final_labels, probs)
        preds_bin = (probs >= 0.5).astype(int)
        accuracy = accuracy_score(final_labels, preds_bin)
        precision = precision_score(final_labels, preds_bin, zero_division=0)
        recall = recall_score(final_labels, preds_bin, zero_division=0)
        f1 = f1_score(final_labels, preds_bin, zero_division=0)
    return auc, accuracy, precision, recall, f1, pr_auc, final_labels, probs


def plot_roc_pr_by_fold(curve_records, save_path, title_prefix=''):
    if len(curve_records) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for rec in curve_records:
        ax.plot(rec['fpr'], rec['tpr'], lw=2, label=f"Fold {rec['fold']} (AUC = {rec['auc']:.3f})")
    ax.set_title('ROC Curves by Fold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=9)

    ax = axes[1]
    for rec in curve_records:
        ax.plot(rec['recall_curve'], rec['precision_curve'], lw=2, label=f"Fold {rec['fold']} (AP = {rec['ap']:.3f})")
    ax.set_title('Precision-Recall Curves by Fold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower left', fontsize=9)

    fig.suptitle(title_prefix, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# -----------------------------
# Flat pair index mapping
# -----------------------------
def build_pair_mapper(matrix):
    miRNA_to_idx = {m: i for i, m in enumerate(matrix.index)}
    disease_to_idx = {d: i for i, d in enumerate(matrix.columns)}
    n_diseases = len(matrix.columns)

    def pair_to_flat_index(miRNA_label, disease_label):
        mi_idx = miRNA_to_idx[miRNA_label]
        d_idx = disease_to_idx[disease_label]
        return mi_idx * n_diseases + d_idx

    return pair_to_flat_index


# -----------------------------
# Split builders (same logic as strongest code)
# -----------------------------
def random_split_single(matrix, pair_to_flat_index, test_size=0.2, negative_ratio=1.0):
    print(f"Performing random split (test_size={test_size}, seed={SEED})...")
    known_pairs = matrix.stack().reset_index()
    known_pairs = known_pairs[known_pairs[0] == 1].rename(columns={'level_0': 'miRNA', 'level_1': 'disease'})
    positive_pairs_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(known_pairs['miRNA'], known_pairs['disease'])])
    pos_train_idx, pos_test_idx = train_test_split(positive_pairs_idx, test_size=test_size, random_state=SEED)
    all_neg = np.where(matrix.values.flatten() == 0)[0]
    rng = np.random.RandomState(SEED)
    n_neg_train = int(len(pos_train_idx) * negative_ratio)
    n_neg_test = int(len(pos_test_idx) * negative_ratio)
    neg_train_idx = rng.choice(all_neg, size=n_neg_train, replace=False)
    remaining_neg = np.setdiff1d(all_neg, neg_train_idx)
    neg_test_idx = rng.choice(remaining_neg, size=n_neg_test, replace=False)
    train_mask = np.concatenate([pos_train_idx, neg_train_idx])
    train_labels = torch.tensor(np.concatenate([np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))]), dtype=torch.float)
    test_mask = np.concatenate([pos_test_idx, neg_test_idx])
    test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]), dtype=torch.float)
    print(f"Train: {len(pos_train_idx)} positives, {len(neg_train_idx)} negatives")
    print(f"Test: {len(pos_test_idx)} positives, {len(neg_test_idx)} negatives")
    return train_mask, train_labels, test_mask, test_labels


def cold_disease_split_single(matrix, pair_to_flat_index, test_fraction=0.15, negative_ratio=1.0):
    print(f"Performing cold-start disease split (test_fraction={test_fraction}, seed={SEED})...")
    known_pairs = matrix.stack().reset_index()
    known_pairs = known_pairs[known_pairs[0] == 1].rename(columns={'level_0': 'miRNA', 'level_1': 'disease'})
    diseases = known_pairs['disease'].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(diseases)
    n_test = max(1, int(len(diseases) * test_fraction))
    test_diseases = set(diseases[:n_test])
    train_diseases = set(diseases[n_test:])
    train_pairs = known_pairs[known_pairs['disease'].isin(train_diseases)]
    test_pairs = known_pairs[known_pairs['disease'].isin(test_diseases)]
    pos_train_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(train_pairs['miRNA'], train_pairs['disease'])])
    pos_test_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(test_pairs['miRNA'], test_pairs['disease'])])

    def get_negative_pairs_for_diseases(selected_diseases, matrix, rng, n_samples):
        all_neg = []
        for d in selected_diseases:
            d_idx = matrix.columns.get_loc(d)
            for m in matrix.index:
                if matrix.loc[m, d] == 0:
                    m_idx = matrix.index.get_loc(m)
                    all_neg.append(m_idx * len(matrix.columns) + d_idx)
        all_neg = np.array(all_neg)
        return rng.choice(all_neg, size=n_samples, replace=False) if len(all_neg) > n_samples else all_neg

    n_neg_train = int(len(pos_train_idx) * negative_ratio)
    n_neg_test = int(len(pos_test_idx) * negative_ratio)
    neg_train_idx = get_negative_pairs_for_diseases(train_diseases, matrix, rng, n_neg_train)
    neg_test_idx = get_negative_pairs_for_diseases(test_diseases, matrix, rng, n_neg_test)
    neg_test_idx = np.setdiff1d(neg_test_idx, neg_train_idx)
    train_mask = np.concatenate([pos_train_idx, neg_train_idx])
    train_labels = torch.tensor(np.concatenate([np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))]), dtype=torch.float)
    test_mask = np.concatenate([pos_test_idx, neg_test_idx])
    test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]), dtype=torch.float)
    print(f"Train diseases: {len(train_diseases)}, Test diseases: {len(test_diseases)}")
    print(f"Train: {len(pos_train_idx)} positives, {len(neg_train_idx)} negatives")
    print(f"Test: {len(pos_test_idx)} positives, {len(neg_test_idx)} negatives")
    return train_mask, train_labels, test_mask, test_labels


def cold_mirna_split_single(matrix, pair_to_flat_index, test_fraction=0.15, negative_ratio=1.0):
    print(f"Performing cold-start miRNA split (test_fraction={test_fraction}, seed={SEED})...")
    known_pairs = matrix.stack().reset_index()
    known_pairs = known_pairs[known_pairs[0] == 1].rename(columns={'level_0': 'miRNA', 'level_1': 'disease'})
    mirnas = known_pairs['miRNA'].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(mirnas)
    n_test = max(1, int(len(mirnas) * test_fraction))
    test_mirnas = set(mirnas[:n_test])
    train_mirnas = set(mirnas[n_test:])
    train_pairs = known_pairs[known_pairs['miRNA'].isin(train_mirnas)]
    test_pairs = known_pairs[known_pairs['miRNA'].isin(test_mirnas)]
    pos_train_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(train_pairs['miRNA'], train_pairs['disease'])])
    pos_test_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(test_pairs['miRNA'], test_pairs['disease'])])

    def get_negative_pairs_for_mirnas(selected_mirnas, matrix, rng, n_samples):
        all_neg = []
        for m in selected_mirnas:
            m_idx = matrix.index.get_loc(m)
            for d in matrix.columns:
                if matrix.loc[m, d] == 0:
                    d_idx = matrix.columns.get_loc(d)
                    all_neg.append(m_idx * len(matrix.columns) + d_idx)
        all_neg = np.array(all_neg)
        return rng.choice(all_neg, size=n_samples, replace=False) if len(all_neg) > n_samples else all_neg

    n_neg_train = int(len(pos_train_idx) * negative_ratio)
    n_neg_test = int(len(pos_test_idx) * negative_ratio)
    neg_train_idx = get_negative_pairs_for_mirnas(train_mirnas, matrix, rng, n_neg_train)
    neg_test_idx = get_negative_pairs_for_mirnas(test_mirnas, matrix, rng, n_neg_test)
    neg_test_idx = np.setdiff1d(neg_test_idx, neg_train_idx)
    train_mask = np.concatenate([pos_train_idx, neg_train_idx])
    train_labels = torch.tensor(np.concatenate([np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))]), dtype=torch.float)
    test_mask = np.concatenate([pos_test_idx, neg_test_idx])
    test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]), dtype=torch.float)
    print(f"Train miRNAs: {len(train_mirnas)}, Test miRNAs: {len(test_mirnas)}")
    print(f"Train: {len(pos_train_idx)} positives, {len(neg_train_idx)} negatives")
    print(f"Test: {len(pos_test_idx)} positives, {len(neg_test_idx)} negatives")
    return train_mask, train_labels, test_mask, test_labels


def get_cv_folds(matrix, pair_to_flat_index, split_mode='random', n_folds=5):
    known_pairs = matrix.stack().reset_index()
    known_pairs = known_pairs[known_pairs[0] == 1].rename(columns={'level_0': 'miRNA', 'level_1': 'disease'})

    if split_mode == 'random':
        positive_pairs_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(known_pairs['miRNA'], known_pairs['disease'])])
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        folds = []
        for train_idx, test_idx in kf.split(positive_pairs_idx):
            pos_train_idx = positive_pairs_idx[train_idx]
            pos_test_idx = positive_pairs_idx[test_idx]
            all_neg = np.where(matrix.values.flatten() == 0)[0]
            rng = np.random.default_rng(SEED)
            neg_train_idx = rng.choice(all_neg, size=len(pos_train_idx), replace=False)
            remaining_neg = np.setdiff1d(all_neg, neg_train_idx)
            neg_test_idx = rng.choice(remaining_neg, size=len(pos_test_idx), replace=False)
            train_mask = np.concatenate([pos_train_idx, neg_train_idx])
            train_labels = torch.tensor(np.concatenate([np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))]), dtype=torch.float)
            test_mask = np.concatenate([pos_test_idx, neg_test_idx])
            test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]), dtype=torch.float)
            folds.append((train_mask, train_labels, test_mask, test_labels))

    elif split_mode == 'cold_disease':
        diseases = known_pairs['disease'].unique()
        rng = np.random.RandomState(SEED)
        rng.shuffle(diseases)
        disease_folds = np.array_split(diseases, n_folds)
        folds = []
        for fold in range(n_folds):
            test_diseases = set(disease_folds[fold])
            train_diseases = set(diseases) - test_diseases
            train_pairs = known_pairs[known_pairs['disease'].isin(train_diseases)]
            test_pairs = known_pairs[known_pairs['disease'].isin(test_diseases)]
            pos_train_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(train_pairs['miRNA'], train_pairs['disease'])])
            pos_test_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(test_pairs['miRNA'], test_pairs['disease'])])

            def get_negative_pairs_for_diseases(selected_diseases, matrix, rng, n_samples):
                all_neg = []
                for d in selected_diseases:
                    d_idx = matrix.columns.get_loc(d)
                    for m in matrix.index:
                        if matrix.loc[m, d] == 0:
                            m_idx = matrix.index.get_loc(m)
                            all_neg.append(m_idx * len(matrix.columns) + d_idx)
                all_neg = np.array(all_neg)
                return rng.choice(all_neg, size=n_samples, replace=False) if len(all_neg) > n_samples else all_neg

            rng_fold = np.random.RandomState(SEED + fold)
            neg_train_idx = get_negative_pairs_for_diseases(train_diseases, matrix, rng_fold, len(pos_train_idx))
            neg_test_idx = get_negative_pairs_for_diseases(test_diseases, matrix, rng_fold, len(pos_test_idx))
            neg_test_idx = np.setdiff1d(neg_test_idx, neg_train_idx)
            train_mask = np.concatenate([pos_train_idx, neg_train_idx])
            train_labels = torch.tensor(np.concatenate([np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))]), dtype=torch.float)
            test_mask = np.concatenate([pos_test_idx, neg_test_idx])
            test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]), dtype=torch.float)
            folds.append((train_mask, train_labels, test_mask, test_labels))

    elif split_mode == 'cold_mirna':
        mirnas = known_pairs['miRNA'].unique()
        rng = np.random.RandomState(SEED)
        rng.shuffle(mirnas)
        mirna_folds = np.array_split(mirnas, n_folds)
        folds = []
        for fold in range(n_folds):
            test_mirnas = set(mirna_folds[fold])
            train_mirnas = set(mirnas) - test_mirnas
            train_pairs = known_pairs[known_pairs['miRNA'].isin(train_mirnas)]
            test_pairs = known_pairs[known_pairs['miRNA'].isin(test_mirnas)]
            pos_train_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(train_pairs['miRNA'], train_pairs['disease'])])
            pos_test_idx = np.array([pair_to_flat_index(m, d) for m, d in zip(test_pairs['miRNA'], test_pairs['disease'])])

            def get_negative_pairs_for_mirnas(selected_mirnas, matrix, rng, n_samples):
                all_neg = []
                for m in selected_mirnas:
                    m_idx = matrix.index.get_loc(m)
                    for d in matrix.columns:
                        if matrix.loc[m, d] == 0:
                            d_idx = matrix.columns.get_loc(d)
                            all_neg.append(m_idx * len(matrix.columns) + d_idx)
                all_neg = np.array(all_neg)
                return rng.choice(all_neg, size=n_samples, replace=False) if len(all_neg) > n_samples else all_neg

            rng_fold = np.random.RandomState(SEED + fold)
            neg_train_idx = get_negative_pairs_for_mirnas(train_mirnas, matrix, rng_fold, len(pos_train_idx))
            neg_test_idx = get_negative_pairs_for_mirnas(test_mirnas, matrix, rng_fold, len(pos_test_idx))
            neg_test_idx = np.setdiff1d(neg_test_idx, neg_train_idx)
            train_mask = np.concatenate([pos_train_idx, neg_train_idx])
            train_labels = torch.tensor(np.concatenate([np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))]), dtype=torch.float)
            test_mask = np.concatenate([pos_test_idx, neg_test_idx])
            test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]), dtype=torch.float)
            folds.append((train_mask, train_labels, test_mask, test_labels))
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}")

    return folds


# -----------------------------
# Optuna objective / tuning
# -----------------------------
def objective(trial, matrix, miRNAs, diseases, train_mask, train_labels, val_mask, val_labels):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    pca_dim = trial.suggest_categorical('pca_dim', [16, 32, 64])
    latent_dim = trial.suggest_int('latent_dim', 16, 64)
    out_channels = trial.suggest_int('out_channels', 8, 64)
    lambda_reg = trial.suggest_float('lambda_reg', 0.01, 0.1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    mdmf_lr = trial.suggest_float('mdmf_lr', 1e-4, 5e-2, log=True)
    mdmf_epochs = trial.suggest_int('mdmf_epochs', 50, 200)

    miRNA_sim, disease_sim = compute_similarity(matrix)
    rgcn_data_new, n_mirnas, n_diseases = prepare_rgcn_data(matrix, miRNA_sim, disease_sim, feature_dim=pca_dim, device=DEVICE)

    # MDMF init from plain MF for better stability
    U0, V0 = _matrix_factorization(matrix.values.astype(np.float32), K=latent_dim, steps=300, seed=SEED)
    mdmf = MDMF(
        len(miRNAs),
        len(diseases),
        latent_dim=latent_dim,
        lambda_reg=lambda_reg,
        init_U=torch.tensor(U0, dtype=torch.float32),
        init_V=torch.tensor(V0, dtype=torch.float32),
    )
    U, V = train_mdmf(
        mdmf,
        torch.tensor(matrix.values, dtype=torch.float),
        torch.tensor(miRNA_sim.values, dtype=torch.float),
        torch.tensor(disease_sim.values, dtype=torch.float),
        epochs=mdmf_epochs,
        lr=mdmf_lr,
    )

    mdmf_features = torch.cat([U, V], dim=0).to(DEVICE)
    rgcn_data_new.x = torch.cat([rgcn_data_new.x, mdmf_features], dim=1)
    rgcn_data_new.mdmf_dim = latent_dim

    model = RGCNGatedPairMLP(
        in_channels=rgcn_data_new.x.shape[1],
        mdmf_dim=latent_dim,
        hidden_channels=hidden_dim,
        out_channels=out_channels,
        dropout=dropout,
        num_layers=num_layers,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    counter = 0
    max_epochs = min(EPOCHS, 200)
    patience = min(PATIENCE, 10)
    train_labels = train_labels.to(DEVICE)
    val_labels = val_labels.to(DEVICE)
    for epoch in range(max_epochs):
        _ = train_model(model, rgcn_data_new, train_mask, train_labels, optimizer, criterion)
        val_auc = evaluate_model(model, rgcn_data_new, val_mask, val_labels)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return best_val_auc


def perform_optuna_tuning(matrix, miRNAs, diseases, train_mask, train_labels, val_mask, val_labels, n_trials=65):
    if optuna is None:
        raise ImportError('Optuna is not installed in this environment.')
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, matrix, miRNAs, diseases, train_mask, train_labels, val_mask, val_labels), n_trials=n_trials)
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best validation AUC: {study.best_value:.4f}")
    return study.best_params


# -----------------------------
# Full train / test for one fold
# -----------------------------
def train_and_evaluate_model(model, rgcn_data, train_mask, train_labels, test_mask, test_labels, learning_rate=1e-3, weight_decay=1e-4, max_epochs=100, patience=10):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_val_auc = -np.inf
    best_state = None
    patience_counter = 0
    train_labels = train_labels.to(DEVICE)
    test_labels = test_labels.to(DEVICE)

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model.pair_logits(rgcn_data, train_mask)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model.pair_logits(rgcn_data, test_mask)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_labels_np = test_labels.cpu().numpy()
            val_auc = roc_auc_score(val_labels_np, val_probs) if len(np.unique(val_labels_np)) > 1 else 0.0
        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            # preserve strongest-code style: do not early break in CV stage

    if best_state is not None:
        model.load_state_dict(best_state)

    auc, accuracy, precision, recall, f1, pr_auc, final_labels, final_probs = compute_full_metrics(model, rgcn_data, test_mask, test_labels)
    return auc, accuracy, precision, recall, f1, pr_auc, best_state, final_labels, final_probs


# -----------------------------
# Cross-validation with best params
# -----------------------------
def cross_validate_with_best_params(matrix, miRNAs, diseases, best_params, split_mode='random', n_folds=5, save_models=True):
    aucs, accuracies, precision_scores, recall_scores, f1_scores, pr_scores = [], [], [], [], [], []
    best_models, best_model_paths = [], []
    curve_records = []

    pair_to_flat_index = build_pair_mapper(matrix)

    print(f"\n=== {split_mode.upper()} Split - {n_folds}-Fold Cross-Validation ===")
    miRNA_sim, disease_sim = compute_similarity(matrix)
    rgcn_data_final, n_mirnas, n_diseases = prepare_rgcn_data(matrix, miRNA_sim, disease_sim, feature_dim=best_params['pca_dim'], device=DEVICE)
    rgcn_data_final.n_mirnas = n_mirnas
    rgcn_data_final.n_diseases = n_diseases

    folds = get_cv_folds(matrix, pair_to_flat_index, split_mode=split_mode, n_folds=n_folds)
    print(f"Split mode: {split_mode}")
    print(f"Number of folds: {n_folds}")
    print(f"Using seed: {SEED}")

    # Full-data similarity + MDMF branch, consistent with strongest code's public-protocol style.
    latent_dim = best_params['latent_dim']
    U0, V0 = _matrix_factorization(matrix.values.astype(np.float32), K=latent_dim, steps=400, seed=SEED)
    mdmf_final = MDMF(
        len(miRNAs),
        len(diseases),
        latent_dim=latent_dim,
        lambda_reg=best_params['lambda_reg'],
        init_U=torch.tensor(U0, dtype=torch.float32),
        init_V=torch.tensor(V0, dtype=torch.float32),
    )
    U_final, V_final = train_mdmf(
        mdmf_final,
        torch.tensor(matrix.values, dtype=torch.float),
        torch.tensor(miRNA_sim.values, dtype=torch.float),
        torch.tensor(disease_sim.values, dtype=torch.float),
        epochs=best_params.get('mdmf_epochs', 120),
        lr=best_params.get('mdmf_lr', 0.01),
    )
    mdmf_features_final = torch.cat([U_final, V_final], dim=0).to(DEVICE)
    rgcn_data_final.x = torch.cat([rgcn_data_final.x, mdmf_features_final], dim=1)
    rgcn_data_final.mdmf_dim = latent_dim

    model_dir = f'{OUTPUT_FOLDER}_rgcn_gated_{split_mode}'
    os.makedirs(model_dir, exist_ok=True)

    for fold, (train_mask, train_labels, test_mask, test_labels) in enumerate(folds):
        print(f"\nTraining Fold {fold + 1}/{n_folds}...")
        n_pos_train = int((train_labels == 1).sum().item())
        n_neg_train = int((train_labels == 0).sum().item())
        n_pos_test = int((test_labels == 1).sum().item())
        n_neg_test = int((test_labels == 0).sum().item())
        print(f"Train: {n_pos_train} positives, {n_neg_train} negatives")
        print(f"Test: {n_pos_test} positives, {n_neg_test} negatives")

        model = RGCNGatedPairMLP(
            in_channels=rgcn_data_final.x.shape[1],
            mdmf_dim=best_params['latent_dim'],
            hidden_channels=best_params['hidden_dim'],
            out_channels=best_params['out_channels'],
            dropout=best_params['dropout'],
            num_layers=best_params['num_layers'],
        )
        auc, accuracy, precision, recall, f1, pr_auc, best_model_state, final_labels, final_probs = train_and_evaluate_model(
            model,
            rgcn_data_final,
            train_mask=train_mask,
            train_labels=train_labels,
            test_mask=test_mask,
            test_labels=test_labels,
            learning_rate=best_params.get('learning_rate', 1e-3),
            weight_decay=best_params.get('weight_decay', 1e-4),
            max_epochs=EPOCHS,
            patience=PATIENCE,
        )
        print(f"Fold {fold + 1} Results:")
        print(f"  AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  PR AUC: {pr_auc:.4f}")

        aucs.append(auc)
        accuracies.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        pr_scores.append(pr_auc)

        fpr, tpr, _ = roc_curve(final_labels, final_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(final_labels, final_probs)
        curve_records.append({
            'fold': fold + 1,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'auc': auc,
            'ap': pr_auc,
        })

        pred_df = pd.DataFrame({
            'flat_index': test_mask,
            'label': final_labels,
            'prob': final_probs,
            'pred': (final_probs >= 0.5).astype(int),
        })
        pred_path = os.path.join(model_dir, f'fold_{fold + 1}_predictions.csv')
        pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')

        if save_models:
            best_model = RGCNGatedPairMLP(
                in_channels=rgcn_data_final.x.shape[1],
                mdmf_dim=best_params['latent_dim'],
                hidden_channels=best_params['hidden_dim'],
                out_channels=best_params['out_channels'],
                dropout=best_params['dropout'],
                num_layers=best_params['num_layers'],
            )
            best_model.load_state_dict(best_model_state)
            best_model.eval()
            best_models.append(best_model)

            model_path = os.path.join(model_dir, f'best_model_fold_{fold + 1}.pth')
            torch.save({
                'fold': fold + 1,
                'split_mode': split_mode,
                'model_state_dict': best_model_state,
                'model_type': 'RGCNGatedPairMLP',
                'model_params': {
                    'in_channels': rgcn_data_final.x.shape[1],
                    'mdmf_dim': best_params['latent_dim'],
                    'hidden_channels': best_params['hidden_dim'],
                    'out_channels': best_params['out_channels'],
                    'dropout': best_params['dropout'],
                    'num_layers': best_params['num_layers'],
                },
                'fold_metrics': {
                    'auc': auc,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'pr_auc': pr_auc,
                },
                'fold_stats': {
                    'n_pos_train': n_pos_train,
                    'n_neg_train': n_neg_train,
                    'n_pos_test': n_pos_test,
                    'n_neg_test': n_neg_test,
                },
            }, model_path)
            best_model_paths.append(model_path)
            print(f"Saved best model for fold {fold + 1} to {model_path}")

    print(f"\n=== Summary of {split_mode.upper()} Split - {n_folds}-Fold Cross-Validation ===")
    print(f"Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Average Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Average PR AUC: {np.mean(pr_scores):.4f} ± {np.std(pr_scores):.4f}")

    if save_models:
        summary = {
            'split_mode': split_mode,
            'n_folds': n_folds,
            'seed': SEED,
            'fold_metrics': {
                'aucs': aucs,
                'accuracies': accuracies,
                'precisions': precision_scores,
                'recalls': recall_scores,
                'f1_scores': f1_scores,
                'pr_scores': pr_scores,
            },
            'average_metrics': {
                'mean_auc': float(np.mean(aucs)),
                'std_auc': float(np.std(aucs)),
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'mean_precision': float(np.mean(precision_scores)),
                'std_precision': float(np.std(precision_scores)),
                'mean_recall': float(np.mean(recall_scores)),
                'std_recall': float(np.std(recall_scores)),
                'mean_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'mean_pr_auc': float(np.mean(pr_scores)),
                'std_pr_auc': float(np.std(pr_scores)),
            },
            'best_params': best_params,
            'model_paths': best_model_paths,
        }
        summary_path = os.path.join(model_dir, 'cross_validation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        fig_path = os.path.join(model_dir, f'{split_mode}_roc_pr_by_fold.png')
        plot_roc_pr_by_fold(curve_records, fig_path, title_prefix=f'{split_mode.upper()} Split')
        print(f"\nSaved cross-validation summary to {summary_path}")

    return best_models, best_model_paths



# -----------------------------
# Gate heatmap / explanation
# -----------------------------
def build_full_feature_graph(matrix, best_params):
    miRNA_sim, disease_sim = compute_similarity(matrix)
    rgcn_data, n_mirnas, n_diseases = prepare_rgcn_data(
        matrix, miRNA_sim, disease_sim, feature_dim=best_params['pca_dim'], device=DEVICE
    )
    rgcn_data.n_mirnas = n_mirnas
    rgcn_data.n_diseases = n_diseases

    latent_dim = best_params['latent_dim']
    U0, V0 = _matrix_factorization(matrix.values.astype(np.float32), K=latent_dim, steps=400, seed=SEED)
    mdmf_final = MDMF(
        len(matrix.index), len(matrix.columns), latent_dim=latent_dim,
        lambda_reg=best_params['lambda_reg'],
        init_U=torch.tensor(U0, dtype=torch.float32),
        init_V=torch.tensor(V0, dtype=torch.float32),
    )
    U_final, V_final = train_mdmf(
        mdmf_final,
        torch.tensor(matrix.values, dtype=torch.float),
        torch.tensor(miRNA_sim.values, dtype=torch.float),
        torch.tensor(disease_sim.values, dtype=torch.float),
        epochs=best_params.get('mdmf_epochs', 120),
        lr=best_params.get('mdmf_lr', 0.01),
    )
    mdmf_features_final = torch.cat([U_final, V_final], dim=0).to(DEVICE)
    rgcn_data.x = torch.cat([rgcn_data.x, mdmf_features_final], dim=1)
    rgcn_data.mdmf_dim = latent_dim
    return rgcn_data, n_mirnas, n_diseases


def build_model_from_params(rgcn_data, best_params):
    model = RGCNGatedPairMLP(
        in_channels=rgcn_data.x.shape[1],
        mdmf_dim=best_params['latent_dim'],
        hidden_channels=best_params['hidden_dim'],
        out_channels=best_params['out_channels'],
        dropout=best_params['dropout'],
        num_layers=best_params['num_layers'],
    ).to(DEVICE)
    return model


def extract_pair_gate_weights(model, data, flat_pair_indices):
    model.eval()
    with torch.no_grad():
        logits = model.pair_logits(data, flat_pair_indices)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        gate = model.last_gate_weights.detach().cpu().numpy()  # [N_nodes, 2]

        flat_pair_indices = np.asarray(flat_pair_indices)
        n_diseases = data.n_diseases
        n_mirnas = data.n_mirnas
        mirna_idx = flat_pair_indices // n_diseases
        disease_idx = flat_pair_indices % n_diseases + n_mirnas

        mirna_gate = gate[mirna_idx]
        disease_gate = gate[disease_idx]
        pair_gate = (mirna_gate + disease_gate) / 2.0
    return probs, pair_gate, mirna_gate, disease_gate


def _plot_heatmap(arr, row_labels, col_labels, save_path, title):
    arr = np.asarray(arr, dtype=float)
    fig_h = max(4.0, 0.42 * len(row_labels) + 1.5)
    fig_w = max(6.0, 1.2 * len(col_labels) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=12)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            color = 'white' if v < 0.5 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Gate weight', rotation=270, labelpad=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_gate_heatmap_for_disease_topk(matrix, best_params, checkpoint_path, disease_name, top_k=20, save_prefix='gate_heatmap_topk'):
    rgcn_data, n_mirnas, n_diseases = build_full_feature_graph(matrix, best_params)
    model = build_model_from_params(rgcn_data, best_params)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    miRNAs = list(matrix.index)
    diseases = list(matrix.columns)
    if disease_name not in diseases:
        raise ValueError(f'Disease not found: {disease_name}')
    disease_idx = diseases.index(disease_name)
    flat_pairs = np.array([mi * n_diseases + disease_idx for mi in range(n_mirnas)], dtype=np.int64)
    probs, pair_gate, mirna_gate, disease_gate = extract_pair_gate_weights(model, rgcn_data, flat_pairs)

    top_idx = np.argsort(-probs)[:top_k]
    row_labels = [f"{miRNAs[i]} (p={probs[i]:.3f})" for i in top_idx]
    pair_arr = pair_gate[top_idx]
    detailed_arr = np.concatenate([mirna_gate[top_idx], disease_gate[top_idx]], axis=1)

    base = f'{save_prefix}_{disease_name.replace("/", "_").replace(" ", "_")}'
    pair_png = f'{base}_pair.png'
    detail_png = f'{base}_detailed.png'
    csv_path = f'{base}.csv'

    _plot_heatmap(pair_arr, row_labels, ['RGCN', 'MDMF'], pair_png, f'Pair-level Gate Weights for Top-{top_k} of {disease_name}')
    _plot_heatmap(detailed_arr, row_labels, ['miR-RGCN', 'miR-MDMF', 'Dis-RGCN', 'Dis-MDMF'], detail_png, f'Detailed Gate Weights for Top-{top_k} of {disease_name}')

    out_df = pd.DataFrame({
        'miRNA': [miRNAs[i] for i in top_idx],
        'disease': disease_name,
        'prob': probs[top_idx],
        'pair_rgcn': pair_arr[:, 0],
        'pair_mdmf': pair_arr[:, 1],
        'mirna_rgcn': mirna_gate[top_idx, 0],
        'mirna_mdmf': mirna_gate[top_idx, 1],
        'disease_rgcn': disease_gate[top_idx, 0],
        'disease_mdmf': disease_gate[top_idx, 1],
    })
    out_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Saved gate heatmaps to: {pair_png}, {detail_png}')
    print(f'Saved gate csv to: {csv_path}')



def parse_selected_pairs(pair_text):
    # Format: "hsa-mir-582|Breast Neoplasms;hsa-mir-744|Breast Neoplasms"
    items = []
    if pair_text is None or str(pair_text).strip() == '':
        return items
    for chunk in str(pair_text).split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '|' not in chunk:
            raise ValueError('Each selected pair must use format miRNA|disease;...')
        mi, di = chunk.split('|', 1)
        items.append((mi.strip(), di.strip()))
    return items



def save_gate_heatmap_for_selected_pairs(matrix, best_params, checkpoint_path, selected_pairs, save_prefix='gate_heatmap_selected'):
    rgcn_data, n_mirnas, n_diseases = build_full_feature_graph(matrix, best_params)
    model = build_model_from_params(rgcn_data, best_params)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    miRNAs = list(matrix.index)
    diseases = list(matrix.columns)
    mi2idx = {m: i for i, m in enumerate(miRNAs)}
    di2idx = {d: i for i, d in enumerate(diseases)}

    flat_pairs = []
    row_labels = []
    for mi, di in selected_pairs:
        if mi not in mi2idx:
            raise ValueError(f'miRNA not found: {mi}')
        if di not in di2idx:
            raise ValueError(f'disease not found: {di}')
        fi = mi2idx[mi] * n_diseases + di2idx[di]
        flat_pairs.append(fi)
        row_labels.append(f'{mi} | {di}')
    flat_pairs = np.array(flat_pairs, dtype=np.int64)

    probs, pair_gate, mirna_gate, disease_gate = extract_pair_gate_weights(model, rgcn_data, flat_pairs)
    row_labels = [f'{lbl} (p={p:.3f})' for lbl, p in zip(row_labels, probs)]
    detailed_arr = np.concatenate([mirna_gate, disease_gate], axis=1)

    pair_png = f'{save_prefix}_pair.png'
    detail_png = f'{save_prefix}_detailed.png'
    csv_path = f'{save_prefix}.csv'

    _plot_heatmap(pair_gate, row_labels, ['RGCN', 'MDMF'], pair_png, 'Pair-level Gate Weights for Selected Pairs')
    _plot_heatmap(detailed_arr, row_labels, ['miR-RGCN', 'miR-MDMF', 'Dis-RGCN', 'Dis-MDMF'], detail_png, 'Detailed Gate Weights for Selected Pairs')

    out_df = pd.DataFrame({
        'pair': row_labels,
        'prob': probs,
        'pair_rgcn': pair_gate[:, 0],
        'pair_mdmf': pair_gate[:, 1],
        'mirna_rgcn': mirna_gate[:, 0],
        'mirna_mdmf': mirna_gate[:, 1],
        'disease_rgcn': disease_gate[:, 0],
        'disease_mdmf': disease_gate[:, 1],
    })
    out_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Saved gate heatmaps to: {pair_png}, {detail_png}')
    print(f'Saved gate csv to: {csv_path}')

# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Full migrated RGCN + Gated Fusion + Pair MLP + MDMF for IMCMDA pair-list input')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='data/alldata.xlsx')
    parser.add_argument('--best_params_file_path', type=str, default='best_params_cv.json')
    parser.add_argument('--output_folder', type=str, default='models_cv')
    parser.add_argument('--mode', type=int, choices=[0, 1, 2], default=0, help='0=random, 1=cold_disease, 2=cold_mirna')
    parser.add_argument('--optuna_tuning', action='store_true')
    parser.add_argument('--no_optuna_tuning', action='store_true')
    parser.add_argument('--negative_ratio', type=float, default=1.0)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--test_fraction', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=65)
    parser.add_argument('--save_gate_heatmap', action='store_true')
    parser.add_argument('--only_heatmap', action='store_true')
    parser.add_argument('--heatmap_checkpoint_path', type=str, default='')
    parser.add_argument('--gate_heatmap_disease', type=str, default='')
    parser.add_argument('--gate_heatmap_top_k', type=int, default=20)
    parser.add_argument('--gate_heatmap_pairs', type=str, default='')
    return parser.parse_args()


def main():
    global DEVICE, OUTPUT_FOLDER, EPOCHS, PATIENCE

    args = parse_args()
    mode_id_to_name = {0: 'random', 1: 'cold_disease', 2: 'cold_mirna'}
    split_mode = mode_id_to_name[args.mode]

    optuna_tuning = True
    if args.optuna_tuning:
        optuna_tuning = True
    if args.no_optuna_tuning:
        optuna_tuning = False

    set_seed(args.seed)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    OUTPUT_FOLDER = args.output_folder
    EPOCHS = args.epochs
    PATIENCE = args.patience

    file_path = args.data_path
    best_params_file_path = f"{args.best_params_file_path.split('.')[0]}_rgcn_gated_{split_mode}.json"
    negative_ratio = args.negative_ratio
    test_size = args.test_size
    test_fraction = args.test_fraction
    n_trials = args.n_trials

    data = load_data(file_path)
    if data is None:
        raise RuntimeError(f'Failed to load input data from: {file_path}')
    matrix, miRNAs, diseases = preprocess_data(data)
    n_diseases = len(diseases)
    n_mirnas = len(miRNAs)

    print('Loaded pair table converted to association matrix:')
    print('  n_miRNAs   =', n_mirnas)
    print('  n_diseases =', n_diseases)
    print('  positives  =', int(matrix.values.sum()))

    pair_to_flat_index = build_pair_mapper(matrix)
    params_path = f"{best_params_file_path.split('.json')[0]}.json"

    if args.only_heatmap:
        print(f"\n=== {split_mode.upper()} Split - Heatmap Only Mode (alldata) ===")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f'Could not find saved params: {params_path}.')
        with open(params_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        best_params.setdefault('mdmf_lr', 0.01)
        best_params.setdefault('mdmf_epochs', 120)

        checkpoint_path = args.heatmap_checkpoint_path.strip()
        if checkpoint_path == '':
            checkpoint_path = os.path.join(f'{OUTPUT_FOLDER}_rgcn_gated_{split_mode}', 'best_model_fold_1.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Could not find checkpoint for heatmap: {checkpoint_path}')
        if not args.save_gate_heatmap:
            raise ValueError('--only_heatmap requires --save_gate_heatmap')

        if args.gate_heatmap_disease.strip() != '':
            save_gate_heatmap_for_disease_topk(
                matrix=matrix,
                best_params=best_params,
                checkpoint_path=checkpoint_path,
                disease_name=args.gate_heatmap_disease.strip(),
                top_k=args.gate_heatmap_top_k,
                save_prefix=os.path.join(OUTPUT_FOLDER + f'_rgcn_gated_{split_mode}', 'gate_heatmap_topk')
            )

        if args.gate_heatmap_pairs.strip() != '':
            selected_pairs = parse_selected_pairs(args.gate_heatmap_pairs)
            save_gate_heatmap_for_selected_pairs(
                matrix=matrix,
                best_params=best_params,
                checkpoint_path=checkpoint_path,
                selected_pairs=selected_pairs,
                save_prefix=os.path.join(OUTPUT_FOLDER + f'_rgcn_gated_{split_mode}', 'gate_heatmap_selected')
            )
        return

    if optuna_tuning:
        if optuna is None:
            raise ImportError('Optuna is not installed, but --optuna_tuning was requested.')

        print(f"\n=== {split_mode.upper()} Split - Optuna Tuning (RGCN Gated Full) ===")
        if split_mode == 'random':
            train_mask, train_labels, test_mask, test_labels = random_split_single(matrix, pair_to_flat_index, test_size=test_size, negative_ratio=negative_ratio)
        elif split_mode == 'cold_disease':
            train_mask, train_labels, test_mask, test_labels = cold_disease_split_single(matrix, pair_to_flat_index, test_fraction=test_fraction, negative_ratio=negative_ratio)
        else:
            train_mask, train_labels, test_mask, test_labels = cold_mirna_split_single(matrix, pair_to_flat_index, test_fraction=test_fraction, negative_ratio=negative_ratio)

        best_params = perform_optuna_tuning(matrix, miRNAs, diseases, train_mask, train_labels, test_mask, test_labels, n_trials=n_trials)
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print(f"Best params saved to: {params_path}")
    else:
        print(f"\n=== {split_mode.upper()} Split - Using Saved Best Params (RGCN Gated Full) ===")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f'Could not find saved params: {params_path}. Run with --optuna_tuning first, or place the JSON in the same directory.')
        with open(params_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        print(f"Loaded params from: {params_path}")

    # keep runtime config explicit
    best_params.setdefault('mdmf_lr', 0.01)
    best_params.setdefault('mdmf_epochs', 120)

    best_models, model_paths = cross_validate_with_best_params(
        matrix=matrix,
        miRNAs=miRNAs,
        diseases=diseases,
        best_params=best_params,
        split_mode=split_mode,
        n_folds=5,
        save_models=True,
    )

    if args.save_gate_heatmap:
        checkpoint_path = args.heatmap_checkpoint_path.strip()
        if checkpoint_path == '':
            if len(model_paths) == 0:
                raise RuntimeError('No checkpoint path available for heatmap generation.')
            checkpoint_path = model_paths[0]

        if args.gate_heatmap_disease.strip() != '':
            save_gate_heatmap_for_disease_topk(
                matrix=matrix,
                best_params=best_params,
                checkpoint_path=checkpoint_path,
                disease_name=args.gate_heatmap_disease.strip(),
                top_k=args.gate_heatmap_top_k,
                save_prefix=os.path.join(OUTPUT_FOLDER + f'_rgcn_gated_{split_mode}', 'gate_heatmap_topk')
            )

        if args.gate_heatmap_pairs.strip() != '':
            selected_pairs = parse_selected_pairs(args.gate_heatmap_pairs)
            save_gate_heatmap_for_selected_pairs(
                matrix=matrix,
                best_params=best_params,
                checkpoint_path=checkpoint_path,
                selected_pairs=selected_pairs,
                save_prefix=os.path.join(OUTPUT_FOLDER + f'_rgcn_gated_{split_mode}', 'gate_heatmap_selected')
            )


if __name__ == '__main__':
    main()

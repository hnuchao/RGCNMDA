#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_gate_distribution_alldata.py

基于:
- data/alldata.xlsx
- best_params json
- best_model_fold_X.pth

统计全数据 gate 分布，包括：
1) 全局平均 RGCN gate / MDMF gate
2) 全局平均 miRNA-side / disease-side gate
3) 不同 disease 的 gate 分布
4) 导出 summary + per-disease csv
5) 画图：
   - 全局平均 gate 柱状图
   - 不同 disease 的 gate 分布箱线图
   - 不同 disease 的 mean RGCN gate 排名前 N 条形图

支持两种统计范围：
- --pair_scope known   仅统计已知正样本 pair（默认，推荐）
- --pair_scope all     统计所有 miRNA-disease 组合（会更慢）

推荐先用 known。
"""

import os
import json
import random
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data


# =========================
# 基础工具
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(file_path):
    return pd.read_excel(file_path)


def preprocess_data(data):
    associations = data[['miRNA', 'disease']].drop_duplicates()
    miRNAs = associations['miRNA'].unique()
    diseases = associations['disease'].unique()

    matrix = pd.DataFrame(0, index=miRNAs, columns=diseases)
    matrix.index.name = 'miRNA'
    matrix.columns.name = 'disease'

    for _, row in associations.iterrows():
        matrix.loc[row['miRNA'], row['disease']] = 1

    return matrix, miRNAs, diseases


def compute_similarity(matrix):
    gamma_m = 1.0 / (matrix.sum(axis=1).mean() + 1e-10)
    miRNA_sim = pd.DataFrame(rbf_kernel(matrix, gamma=gamma_m), index=matrix.index, columns=matrix.index)

    gamma_d = 1.0 / (matrix.sum(axis=0).mean() + 1e-10)
    disease_sim = pd.DataFrame(rbf_kernel(matrix.T, gamma=gamma_d), index=matrix.columns, columns=matrix.columns)

    return miRNA_sim, disease_sim


def prepare_rgcn_data_from_cached_similarity(matrix, miRNA_sim, disease_sim, feature_dim=16, device="cpu"):
    miRNAs = matrix.index
    diseases = matrix.columns
    n_mirnas = len(miRNAs)
    n_diseases = len(diseases)

    pca_m = PCA(n_components=feature_dim, random_state=42)
    miRNA_features = pca_m.fit_transform(miRNA_sim)

    pca_d = PCA(n_components=feature_dim, random_state=42)
    disease_features = pca_d.fit_transform(disease_sim)

    x = torch.tensor(np.vstack([miRNA_features, disease_features]), dtype=torch.float).to(device)

    edges = matrix.stack().reset_index()
    edges = edges[edges[0] == 1]

    miRNA_indices = edges['miRNA'].apply(lambda x: list(miRNAs).index(x)).values
    disease_indices = edges['disease'].apply(lambda x: list(diseases).index(x) + n_mirnas).values

    forward_edge_index = np.array([miRNA_indices, disease_indices])
    reverse_edge_index = np.array([disease_indices, miRNA_indices])

    edge_index = torch.tensor(np.concatenate([forward_edge_index, reverse_edge_index], axis=1), dtype=torch.long).to(device)
    edge_type = torch.tensor(
        np.concatenate([
            np.zeros(len(miRNA_indices), dtype=np.int64),
            np.ones(len(miRNA_indices), dtype=np.int64),
        ]),
        dtype=torch.long,
    ).to(device)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type).to(device)
    data.n_mirnas = n_mirnas
    data.n_diseases = n_diseases
    return data, n_mirnas, n_diseases


class MDMF(nn.Module):
    def __init__(self, num_mirnas, num_diseases, latent_dim, lambda_reg, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.U = nn.Parameter(torch.randn(num_mirnas, latent_dim))
        self.V = nn.Parameter(torch.randn(num_diseases, latent_dim))
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


def train_mdmf(mdmf, A, S_m, S_d, epochs=100, lr=0.01, device="cpu"):
    optimizer = torch.optim.Adam(mdmf.parameters(), lr=lr)
    mdmf.to(device)
    A, S_m, S_d = A.to(device), S_m.to(device), S_d.to(device)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = mdmf(A, S_m, S_d)
        loss.backward()
        optimizer.step()

    return mdmf.get_features()


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
        for _ in range(num_layers - 2):
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

        flat_pair_indices = torch.as_tensor(flat_pair_indices, dtype=torch.long, device=node_out.device)
        mirna_idx = flat_pair_indices // n_diseases
        disease_idx = flat_pair_indices % n_diseases

        return self.decoder(mirna_emb[mirna_idx], disease_emb[disease_idx])


def pair_to_flat_index(miRNA_label, disease_label, matrix):
    mi_idx = matrix.index.get_loc(miRNA_label)
    d_idx = matrix.columns.get_loc(disease_label)
    return mi_idx * len(matrix.columns) + d_idx


def rebuild_everything(data_path, params_path, checkpoint_path, device="cpu", seed=42):
    set_seed(seed)

    df = load_data(data_path)
    matrix, miRNAs, diseases = preprocess_data(df)

    with open(params_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f)

    miRNA_sim, disease_sim = compute_similarity(matrix)
    rgcn_data, n_mirnas, n_diseases = prepare_rgcn_data_from_cached_similarity(
        matrix, miRNA_sim, disease_sim,
        feature_dim=best_params['pca_dim'],
        device=device
    )

    mdmf_final = MDMF(
        num_mirnas=len(miRNAs),
        num_diseases=len(diseases),
        latent_dim=best_params['latent_dim'],
        lambda_reg=best_params['lambda_reg'],
        seed=seed
    )

    U_final, V_final = train_mdmf(
        mdmf_final,
        torch.tensor(matrix.values, dtype=torch.float),
        torch.tensor(miRNA_sim.values, dtype=torch.float),
        torch.tensor(disease_sim.values, dtype=torch.float),
        epochs=best_params.get('mdmf_epochs_final', 100),
        lr=best_params.get('mdmf_lr', 0.01),
        device=device
    )

    mdmf_features_final = torch.cat([U_final, V_final], dim=0).to(device)
    rgcn_data.x = torch.cat([rgcn_data.x, mdmf_features_final], dim=1)
    rgcn_data.mdmf_dim = best_params['latent_dim']

    model = RGCNGatedPairMLP(
        in_channels=rgcn_data.x.shape[1],
        mdmf_dim=best_params['latent_dim'],
        hidden_channels=best_params['hidden_dim'],
        out_channels=best_params['out_channels'],
        dropout=best_params['dropout'],
        num_layers=best_params['num_layers'],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    return matrix, model, rgcn_data


def build_pair_table(matrix, pair_scope="known"):
    if pair_scope == "known":
        edges = matrix.stack().reset_index()
        edges = edges[edges[0] == 1]
        out = edges[['miRNA', 'disease']].copy()
        out.columns = ['miRNA', 'disease']
        return out.reset_index(drop=True)

    elif pair_scope == "all":
        mirnas = matrix.index.tolist()
        diseases = matrix.columns.tolist()
        rows = []
        for m in mirnas:
            for d in diseases:
                rows.append((m, d))
        out = pd.DataFrame(rows, columns=['miRNA', 'disease'])
        return out
    else:
        raise ValueError(f"Unsupported pair_scope: {pair_scope}")


def compute_gate_for_pairs(model, rgcn_data, matrix, pair_df, batch_size=20000, device="cpu"):
    n_diseases = rgcn_data.n_diseases
    all_records = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(pair_df), batch_size):
            batch = pair_df.iloc[start:start + batch_size].copy()
            flat_idx = np.array(
                [pair_to_flat_index(m, d, matrix) for m, d in zip(batch['miRNA'], batch['disease'])],
                dtype=np.int64
            )

            logits = model.pair_logits(rgcn_data, flat_idx)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            gate = model.last_gate_weights.detach().cpu().numpy()

            mirna_idx = flat_idx // n_diseases
            disease_idx_local = flat_idx % n_diseases
            disease_idx_global = disease_idx_local + rgcn_data.n_mirnas

            mirna_gate = gate[mirna_idx]
            disease_gate = gate[disease_idx_global]
            pair_gate = (mirna_gate + disease_gate) / 2.0

            rec = pd.DataFrame({
                'miRNA': batch['miRNA'].values,
                'disease': batch['disease'].values,
                'prob': probs,
                'pair_rgcn': pair_gate[:, 0],
                'pair_mdmf': pair_gate[:, 1],
                'mirna_rgcn': mirna_gate[:, 0],
                'mirna_mdmf': mirna_gate[:, 1],
                'disease_rgcn': disease_gate[:, 0],
                'disease_mdmf': disease_gate[:, 1],
            })
            all_records.append(rec)

    return pd.concat(all_records, axis=0, ignore_index=True)


def plot_global_mean(summary_dict, save_path):
    vals = [
        summary_dict['mean_pair_rgcn'],
        summary_dict['mean_pair_mdmf'],
        summary_dict['mean_mirna_rgcn'],
        summary_dict['mean_mirna_mdmf'],
        summary_dict['mean_disease_rgcn'],
        summary_dict['mean_disease_mdmf'],
    ]
    labels = [
        'Pair-RGCN', 'Pair-MDMF',
        'miRNA-RGCN', 'miRNA-MDMF',
        'Disease-RGCN', 'Disease-MDMF'
    ]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=20)
    plt.ylabel('Mean gate weight')
    plt.ylim(0, 1.0)
    plt.title('Global Mean Gate Weights')

    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_disease_boxplot(disease_df, save_path):
    plt.figure(figsize=(8, 5))
    data = [disease_df['mean_pair_rgcn'].values, disease_df['mean_pair_mdmf'].values]
    plt.boxplot(data, labels=['RGCN', 'MDMF'])
    plt.ylabel('Disease-level mean gate')
    plt.title('Distribution of Disease-level Mean Gate Weights')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_disease_bar(disease_df, save_path, top_n=20):
    df = disease_df.sort_values('mean_pair_rgcn', ascending=False).head(top_n)

    plt.figure(figsize=(10, max(6, 0.35 * len(df))))
    y = np.arange(len(df))
    plt.barh(y, df['mean_pair_rgcn'].values)
    plt.yticks(y, df['disease'].values, fontsize=8)
    plt.xlabel('Mean pair-level RGCN gate')
    plt.title(f'Top-{top_n} Diseases by Mean RGCN Gate')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/alldata.xlsx')
    parser.add_argument('--params_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='gate_distribution_outputs')
    parser.add_argument('--pair_scope', type=str, choices=['known', 'all'], default='known')
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--top_n_disease', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    matrix, model, rgcn_data = rebuild_everything(
        data_path=args.data_path,
        params_path=args.params_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        seed=args.seed
    )

    pair_df = build_pair_table(matrix, pair_scope=args.pair_scope)
    print(f'Pair scope = {args.pair_scope}, number of pairs = {len(pair_df)}')

    gate_df = compute_gate_for_pairs(
        model=model,
        rgcn_data=rgcn_data,
        matrix=matrix,
        pair_df=pair_df,
        batch_size=args.batch_size,
        device=device
    )

    # 全局平均
    summary = {
        'pair_scope': args.pair_scope,
        'num_pairs': int(len(gate_df)),
        'mean_pair_rgcn': float(gate_df['pair_rgcn'].mean()),
        'mean_pair_mdmf': float(gate_df['pair_mdmf'].mean()),
        'mean_mirna_rgcn': float(gate_df['mirna_rgcn'].mean()),
        'mean_mirna_mdmf': float(gate_df['mirna_mdmf'].mean()),
        'mean_disease_rgcn': float(gate_df['disease_rgcn'].mean()),
        'mean_disease_mdmf': float(gate_df['disease_mdmf'].mean()),
        'std_pair_rgcn': float(gate_df['pair_rgcn'].std()),
        'std_pair_mdmf': float(gate_df['pair_mdmf'].std()),
    }

    # 不同 disease 的分布
    disease_df = gate_df.groupby('disease').agg(
        n_pairs=('disease', 'size'),
        mean_prob=('prob', 'mean'),
        mean_pair_rgcn=('pair_rgcn', 'mean'),
        mean_pair_mdmf=('pair_mdmf', 'mean'),
        std_pair_rgcn=('pair_rgcn', 'std'),
        std_pair_mdmf=('pair_mdmf', 'std'),
        mean_mirna_rgcn=('mirna_rgcn', 'mean'),
        mean_mirna_mdmf=('mirna_mdmf', 'mean'),
        mean_disease_rgcn=('disease_rgcn', 'mean'),
        mean_disease_mdmf=('disease_mdmf', 'mean'),
    ).reset_index()

    # 保存结果
    summary_path = os.path.join(args.output_dir, f'gate_summary_{args.pair_scope}.json')
    gate_csv_path = os.path.join(args.output_dir, f'gate_all_pairs_{args.pair_scope}.csv')
    disease_csv_path = os.path.join(args.output_dir, f'gate_by_disease_{args.pair_scope}.csv')

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    gate_df.to_csv(gate_csv_path, index=False, encoding='utf-8-sig')
    disease_df.to_csv(disease_csv_path, index=False, encoding='utf-8-sig')

    # 作图
    plot_global_mean(summary, os.path.join(args.output_dir, f'global_mean_gate_{args.pair_scope}.png'))
    plot_disease_boxplot(disease_df, os.path.join(args.output_dir, f'disease_gate_boxplot_{args.pair_scope}.png'))
    plot_top_disease_bar(disease_df, os.path.join(args.output_dir, f'top_disease_rgcn_bar_{args.pair_scope}.png'), top_n=args.top_n_disease)

    print('=== Global Gate Summary ===')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Saved: {summary_path}')
    print(f'Saved: {gate_csv_path}')
    print(f'Saved: {disease_csv_path}')


if __name__ == '__main__':
    main()

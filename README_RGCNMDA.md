# RGCNMDA

Adaptive Fusion of Relational Graph Learning and Matrix Factorization for miRNA–Disease Association Prediction.

This repository contains the implementation of **RGCNMDA**, an interpretable hybrid framework for miRNA–disease association prediction. The model combines:

- **Similarity + PCA initialization**
- **RGCN-based graph representation learning**
- **MDMF-based global latent factor learning**
- **Node-wise gated fusion**
- **Pairwise MLP decoding**

The main training / evaluation script is `cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py`. The gate interpretation script is `analyze_gate_distribution_alldata.py`.

## 1. Repository overview

A typical repository layout is:

```text
RGCNMDA/
├── data/
│   └── alldata.xlsx
├── cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py
├── analyze_gate_distribution_alldata.py
├── best_params_cv_rgcn_gated_random.json
├── best_params_cv_rgcn_gated_cold_disease.json
├── best_params_cv_rgcn_gated_cold_mirna.json
├── models_cv_rgcn_gated_random/
├── models_cv_rgcn_gated_cold_disease/
├── models_cv_rgcn_gated_cold_mirna/
├── gate_distribution_outputs/
└── README.md
```

## 2. Model architecture and where each module is implemented

### 2.1 Data loading and preprocessing
Implemented in:

- `load_data(...)`
- `preprocess_data(...)`

These functions load the association file, identify the miRNA and disease columns, remove duplicates, and convert the pair table into a binary association matrix.

### 2.2 Similarity construction and PCA initialization
Implemented in:

- `compute_similarity(...)`
- `prepare_rgcn_data(...)`

This module computes Gaussian interaction-profile similarities for miRNAs and diseases, then applies PCA to obtain node features for graph initialization.

### 2.3 MDMF branch
Implemented in:

- `_matrix_factorization(...)`
- `MDMF`
- `train_mdmf(...)`

This branch provides global latent representations for miRNAs and diseases. A lightweight matrix factorization initializer is first used for stability, and then the trainable MDMF module is optimized with similarity regularization.

### 2.4 RGCN branch
Implemented in:

- `RGCNGatedPairMLP`
- internal `self.rgcn_layers`
- `encode(...)`

This branch builds a relation-aware bipartite graph and learns graph embeddings through multi-layer `RGCNConv`.

### 2.5 Gated fusion
Implemented in:

- `NodewiseSoftGateFusion`

This module adaptively fuses graph embeddings and MDMF embeddings with a node-wise soft gate.

### 2.6 Pair decoder
Implemented in:

- `PairMLPDecoder`

For each miRNA–disease pair, the decoder uses concatenation, absolute difference, and element-wise product to construct pair features and outputs the final score.

### 2.7 Data splitting and evaluation
Implemented in:

- `random_split_single(...)`
- `cold_disease_split_single(...)`
- `cold_mirna_split_single(...)`
- `cross_validate_with_best_params(...)`
- `compute_full_metrics(...)`

Supported evaluation settings:

- `--mode 0`: random split
- `--mode 1`: cold-disease split
- `--mode 2`: cold-miRNA split

### 2.8 Hyperparameter tuning
Implemented in:

- `objective(...)`
- `perform_optuna_tuning(...)`

Optuna is used to search over hidden dimension, learning rate, dropout, number of layers, PCA dimension, latent dimension, output channels, regularization strength, weight decay, MDMF learning rate, and MDMF training epochs.

### 2.9 Gate heatmap generation
Implemented in the main training script and triggered by arguments such as:

- `--save_gate_heatmap`
- `--only_heatmap`
- `--heatmap_checkpoint_path`
- `--gate_heatmap_disease`
- `--gate_heatmap_top_k`
- `--gate_heatmap_pairs`

This is used for:
- disease-specific top-k candidate gate heatmaps
- manually selected pair gate heatmaps

### 2.10 Global gate distribution analysis
Implemented in:

- `analyze_gate_distribution_alldata.py`

This script computes:
- global mean gate weights
- miRNA-side and disease-side mean gates
- disease-level gate distributions
- disease-level summary CSV files
- gate interpretation figures

## 3. Environment

Recommended environment:

- Python 3.10+
- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- scikit-learn
- matplotlib
- Optuna

Example installation:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy pandas scikit-learn matplotlib optuna openpyxl
```

If you use GPU, make sure your installed PyTorch build matches your CUDA version.

## 4. Input data format

The main script expects a pair table such as `data/alldata.xlsx` with at least two columns:

- `miRNA`
- `disease`

Example:

```text
miRNA           disease
hsa-mir-145     Breast Neoplasms
hsa-mir-21      Breast Neoplasms
hsa-mir-29a     Liver Neoplasms
...
```

The script will automatically convert this pair table into a binary association matrix.

## 5. Main training and evaluation commands

The main script is:

```bash
python cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py
```

### 5.1 Random split with Optuna tuning
```bash
python cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py --mode 0 --data_path data/alldata.xlsx --best_params_file_path best_params_cv.json --output_folder models_cv --optuna_tuning --n_trials 65
```

### 5.2 Cold-disease split with Optuna tuning
```bash
python cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py --mode 1 --data_path data/alldata.xlsx --best_params_file_path best_params_cv.json --output_folder models_cv --optuna_tuning --n_trials 65
```

### 5.3 Cold-miRNA split with Optuna tuning
```bash
python cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py --mode 2 --data_path data/alldata.xlsx --best_params_file_path best_params_cv.json --output_folder models_cv --optuna_tuning --n_trials 65
```

### 5.4 Random split using saved best parameters
```bash
python cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py --mode 0 --data_path data/alldata.xlsx --best_params_file_path best_params_cv.json --output_folder models_cv --no_optuna_tuning
```

### 5.5 Cold-disease split using saved best parameters
```bash
python cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py --mode 1 --data_path data/alldata.xlsx --best_params_file_path best_params_cv.json --output_folder models_cv --no_optuna_tuning
```

### 5.6 Cold-miRNA split using saved best parameters
```bash
python cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py --mode 2 --data_path data/alldata.xlsx --best_params_file_path best_params_cv.json --output_folder models_cv --no_optuna_tuning
```

## 6. Important command-line arguments

Main arguments in `cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py`:

- `--seed`: random seed
- `--data_path`: input pair table
- `--best_params_file_path`: base name of parameter JSON file
- `--output_folder`: output directory prefix
- `--mode`: split mode (`0`, `1`, `2`)
- `--optuna_tuning`: enable hyperparameter tuning
- `--no_optuna_tuning`: disable tuning and use saved parameters
- `--negative_ratio`: negative sampling ratio
- `--test_size`: random split test proportion
- `--test_fraction`: cold split held-out fraction
- `--epochs`: training epochs
- `--patience`: early stopping patience
- `--n_trials`: number of Optuna trials
- `--save_gate_heatmap`: save heatmaps after training
- `--only_heatmap`: skip training and only draw heatmaps
- `--heatmap_checkpoint_path`: checkpoint path used for heatmap generation
- `--gate_heatmap_disease`: disease name for top-k heatmap
- `--gate_heatmap_top_k`: number of pairs shown in disease heatmap
- `--gate_heatmap_pairs`: selected pairs for pair-level heatmap

## 7. Output files from training

For each split, the script generates a parameter file with a split-specific suffix, for example:

- `best_params_cv_rgcn_gated_random.json`
- `best_params_cv_rgcn_gated_cold_disease.json`
- `best_params_cv_rgcn_gated_cold_mirna.json`

The model checkpoints and summaries are saved into split-specific folders, for example:

- `models_cv_rgcn_gated_random/`
- `models_cv_rgcn_gated_cold_disease/`
- `models_cv_rgcn_gated_cold_mirna/`

Typical outputs include:

- `best_model_fold_1.pth`
- `best_model_fold_2.pth`
- ...
- `cross_validation_summary.json`

The summary file records:
- fold-wise AUC / Accuracy / Precision / Recall / F1 / PR-AUC
- average metrics
- best parameters
- checkpoint paths

## 8. Heatmap generation commands

### 8.1 Generate disease-specific top-k gate heatmap
Use an already trained checkpoint and a saved parameter JSON file.

Random split example:

```bash
python cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py --mode 0 --data_path data/alldata.xlsx --best_params_file_path best_params_cv.json --output_folder models_cv --no_optuna_tuning --save_gate_heatmap --only_heatmap --heatmap_checkpoint_path models_cv_rgcn_gated_random/best_model_fold_1.pth --gate_heatmap_disease "Breast Neoplasms" --gate_heatmap_top_k 20
```

### 8.2 Generate heatmap for selected miRNA–disease pairs
The `--gate_heatmap_pairs` argument should contain manually selected pairs.

Example format:

```text
hsa-mir-582|Breast Neoplasms;hsa-mir-744|Breast Neoplasms
```

Command:

```bash
python cv_mirna_split_rgcn_gated_pair_mlp_gate_alldata.py --mode 0 --data_path data/alldata.xlsx --best_params_file_path best_params_cv.json --output_folder models_cv --no_optuna_tuning --save_gate_heatmap --only_heatmap --heatmap_checkpoint_path models_cv_rgcn_gated_random/best_model_fold_1.pth --gate_heatmap_pairs "hsa-mir-582|Breast Neoplasms;hsa-mir-744|Breast Neoplasms"
```

## 9. Gate distribution analysis

The second script is:

```bash
python analyze_gate_distribution_alldata.py
```

It summarizes gate behavior over the full dataset.

### 9.1 Analyze known positive pairs
```bash
python analyze_gate_distribution_alldata.py --data_path data/alldata.xlsx --params_path best_params_cv_rgcn_gated_random.json --checkpoint_path models_cv_rgcn_gated_random/best_model_fold_1.pth --output_dir gate_distribution_outputs --pair_scope known --device cuda
```

### 9.2 Analyze all possible miRNA–disease pairs
```bash
python analyze_gate_distribution_alldata.py --data_path data/alldata.xlsx --params_path best_params_cv_rgcn_gated_random.json --checkpoint_path models_cv_rgcn_gated_random/best_model_fold_1.pth --output_dir gate_distribution_outputs_all --pair_scope all --device cuda
```

### 9.3 Output files from gate analysis
Typical outputs include:

- `gate_summary_known.json`
- `gate_all_pairs_known.csv`
- `gate_by_disease_known.csv`
- `global_mean_gate_known.png`
- `disease_gate_boxplot_known.png`
- `top_disease_rgcn_bar_known.png`

If `--pair_scope all` is used, the file suffix becomes `_all`.

## 10. Recommended workflow

### Step 1. Prepare the data
Place your association file at:

```text
data/alldata.xlsx
```

### Step 2. Run Optuna tuning
Choose one split mode and tune parameters.

### Step 3. Run full 5-fold cross-validation
Use `--no_optuna_tuning` to load the saved JSON and train/evaluate.

### Step 4. Draw gate heatmaps
Use `--save_gate_heatmap` together with a trained checkpoint.

### Step 5. Run global gate analysis
Use `analyze_gate_distribution_alldata.py` for global and disease-level gate interpretation.

## 11. Notes

- The script automatically chooses `cuda:0` if CUDA is available, otherwise CPU.
- Input files can be `.xlsx`, `.xls`, `.csv`, `.txt`, or `.tsv`.
- The script expects the input table to contain miRNA and disease columns; if exact column names are not found, the first two columns are used.
- For reproducibility, fix `--seed`.
- For stable runs, tune each split mode separately.
- For gate interpretation, it is recommended to use the checkpoint from fold 1 for consistent visualization.

## 12. Citation

If you use this code in your work, please cite the corresponding paper:

**RGCNMDA: Adaptive Fusion of Relational Graph Learning and Matrix Factorization for miRNA–Disease Association Prediction**

## 13. Contact

For questions about the code or paper, please open an issue or contact the authors.

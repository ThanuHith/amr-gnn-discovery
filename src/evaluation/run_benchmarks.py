"""End-to-End Training, Zero-Leakage Scaffold Benchmarking, and 300 DPI Figure Generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from torch_geometric.loader import DataLoader

from src.data_pipeline.prepare_amr_dataset import DESCRIPTOR_NAMES, prepare_dataset
from src.data_pipeline.reference_antibiotics import (
    BENCHMARK_DISCOVERY_MOLECULES,
    CLINICAL_ANTIBIOTICS,
    compute_antibiotic_novelty,
)
from src.models.amr_gnn import AMRGraphNet
from src.models.baselines import MultiTaskAMRBaseline
from src.models.featurizer import DescriptorScaler, mol_to_graph_data
from src.xai.substructure_explainer import compute_atom_attributions, render_atom_attribution_image


def set_global_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    corr, _ = spearmanr(y_true, y_pred)
    return {
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "spearman_r": round(float(corr if np.isfinite(corr) else 0.0), 4),
    }


def _evaluate_predictions(df_split: pd.DataFrame, preds: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    return {
        "active_abx": _classification_metrics(df_split["active_abx"].to_numpy(), preds["prob_abx"]),
        "gram_neg_accum": _classification_metrics(df_split["gram_neg_accum"].to_numpy(), preds["prob_gram_neg"]),
        "cytotoxic_human": _classification_metrics(df_split["cytotoxic_human"].to_numpy(), preds["prob_cytotox"]),
        "pMIC": _regression_metrics(df_split["pMIC"].to_numpy(), preds["pred_pMIC"]),
        "pCC50": _regression_metrics(df_split["pCC50"].to_numpy(), preds["pred_pCC50"]),
    }


@torch.no_grad()
def _predict_gnn_loader(model: AMRGraphNet, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    probs_abx, pmics, probs_gn, probs_tox, pcc50s = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs_abx.append(torch.sigmoid(out["logit_abx"]).cpu().numpy())
        pmics.append(out["pred_pMIC"].cpu().numpy())
        probs_gn.append(torch.sigmoid(out["logit_gram_neg"]).cpu().numpy())
        probs_tox.append(torch.sigmoid(out["logit_cytotox"]).cpu().numpy())
        pcc50s.append(out["pred_pCC50"].cpu().numpy())
    return {
        "prob_abx": np.concatenate(probs_abx),
        "pred_pMIC": np.concatenate(pmics),
        "prob_gram_neg": np.concatenate(probs_gn),
        "prob_cytotox": np.concatenate(probs_tox),
        "pred_pCC50": np.concatenate(pcc50s),
    }


def train_amr_gnn(
    train_graphs: List,
    val_graphs: List,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    scaler: DescriptorScaler,
    checkpoint_path: Path,
    epochs: int = 55,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> Tuple[AMRGraphNet, Dict[str, List[float]]]:
    """Train Multi-Task AMRGraphNet with class-balanced BCE + SmoothL1 losses."""
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

    model = AMRGraphNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    def _pos_weight(col: str) -> torch.Tensor:
        pos = max(float(df_train[col].sum()), 1.0)
        neg = max(float(len(df_train) - pos), 1.0)
        return torch.tensor([min(neg / pos, 5.0)], dtype=torch.float32, device=device)

    bce_abx = nn.BCEWithLogitsLoss(pos_weight=_pos_weight("active_abx"))
    bce_gn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight("gram_neg_accum"))
    bce_tox = nn.BCEWithLogitsLoss(pos_weight=_pos_weight("cytotoxic_human"))
    reg_loss = nn.SmoothL1Loss(beta=0.35)

    best_score = -1e9
    best_state = None
    history: Dict[str, List[float]] = {"train_loss": [], "val_mean_auc": [], "val_mean_rmse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = (
                1.15 * bce_abx(out["logit_abx"], batch.y_abx)
                + 1.00 * bce_gn(out["logit_gram_neg"], batch.y_gn)
                + 1.15 * bce_tox(out["logit_cytotox"], batch.y_tox)
                + 0.70 * reg_loss(out["pred_pMIC"], batch.y_pmic)
                + 0.70 * reg_loss(out["pred_pCC50"], batch.y_pcc50)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        val_preds = _predict_gnn_loader(model, val_loader, device)
        val_metrics = _evaluate_predictions(df_val, val_preds)
        mean_auc = (
            val_metrics["active_abx"]["roc_auc"]
            + val_metrics["gram_neg_accum"]["roc_auc"]
            + val_metrics["cytotoxic_human"]["roc_auc"]
        ) / 3.0
        mean_rmse = (val_metrics["pMIC"]["rmse"] + val_metrics["pCC50"]["rmse"]) / 2.0
        composite = mean_auc - 0.15 * mean_rmse

        history["train_loss"].append(round(running_loss / max(n_batches, 1), 4))
        history["val_mean_auc"].append(round(mean_auc, 4))
        history["val_mean_rmse"].append(round(mean_rmse, 4))

        if composite > best_score:
            best_score = composite
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": {
                "node_dim": 34,
                "edge_dim": 10,
                "global_dim": len(DESCRIPTOR_NAMES),
                "hidden_dim": 128,
                "heads": 4,
                "dropout": 0.25,
            },
            "best_val_composite": round(float(best_score), 4),
            "history": history,
        },
        checkpoint_path,
    )
    return model, history


def _plot_roc_pr_benchmarks(
    df_test: pd.DataFrame,
    preds_by_model: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    """Create 2x3 ROC & Precision-Recall comparison across scaffold-split test set."""
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.9})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), dpi=300)

    tasks = [
        ("active_abx", "prob_abx", "Antibacterial Growth Inhibition"),
        ("gram_neg_accum", "prob_gram_neg", "Gram-Negative Outer-Membrane Entry"),
        ("cytotoxic_human", "prob_cytotox", "Human HepG2 Cytotoxicity"),
    ]
    styles = {
        "AMRGraphNet (GATv2+GIN)": ("#0f766e", "-", 2.4),
        "XGBoost (ECFP4+eNTRy)": ("#2563eb", "--", 1.9),
        "RandomForest (ECFP4+eNTRy)": ("#d97706", ":", 1.9),
    }

    for col_idx, (label_col, prob_key, title) in enumerate(tasks):
        y_true = df_test[label_col].to_numpy()
        ax_roc = axes[0, col_idx]
        ax_pr = axes[1, col_idx]

        for model_name, preds in preds_by_model.items():
            color, ls, lw = styles[model_name]
            y_prob = preds[prob_key]

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            ax_roc.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, label=f"{model_name} (AUC={roc_auc:.3f})")

            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            ax_pr.plot(rec, prec, color=color, linestyle=ls, linewidth=lw, label=f"{model_name} (AP={pr_auc:.3f})")

        ax_roc.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.0)
        ax_roc.set_title(f"{title}\nROC Curve (Zero-Leakage Test Split)", fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right", fontsize=8, frameon=True)
        ax_roc.grid(alpha=0.25)

        baseline_rate = float(y_true.mean())
        ax_pr.axhline(baseline_rate, color="#94a3b8", linestyle="--", linewidth=1.0, label=f"Chance ({baseline_rate:.2f})")
        ax_pr.set_title(f"{title}\nPrecision-Recall Curve", fontweight="bold")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(loc="lower left", fontsize=8, frameon=True)
        ax_pr.grid(alpha=0.25)

    fig.suptitle(
        "PathoGraph-AMR Benchmark: Scaffold-Split Generalization Across Antibacterial, Porin Entry & Safety Tasks",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_chemical_space_and_novelty(
    df: pd.DataFrame,
    fp_matrix: np.ndarray,
    out_path: Path,
) -> None:
    """Plot 2D ECFP4 PCA chemical space and Tanimoto novelty distribution vs clinical classes."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(fp_matrix.astype(np.float32))
    var_exp = pca.explained_variance_ratio_ * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=300)

    is_clinical = df["source"].str.startswith("clinical_")
    is_active = (df["active_abx"] == 1) & (~is_clinical)
    is_inactive = (df["active_abx"] == 0) & (~is_clinical)

    ax1.scatter(
        coords[is_inactive, 0],
        coords[is_inactive, 1],
        c="#cbd5e1",
        alpha=0.55,
        s=22,
        label="Inactive / Cytotoxic Decoys",
    )
    ax1.scatter(
        coords[is_active, 0],
        coords[is_active, 1],
        c="#0d9488",
        alpha=0.72,
        s=28,
        label="Novel Antibacterial Scaffolds",
    )
    ax1.scatter(
        coords[is_clinical, 0],
        coords[is_clinical, 1],
        c="#2563eb",
        marker="^",
        s=75,
        edgecolors="white",
        linewidth=0.7,
        label="FDA/EMA Clinical Antibiotics (n=36)",
        zorder=5,
    )

    showcase_names = [
        ("Halicin (SU3327) — Novel AI-Discovered Antibiotic", "Halicin (SU3327)", "#dc2626"),
        ("Abaucin (RS102895) — Narrow-Spectrum A. baumannii Hit", "Abaucin (RS102895)", "#9333ea"),
        ("Ciprofloxacin", "Ciprofloxacin", "#1d4ed8"),
    ]
    for full_name, short_label, col in showcase_names:
        sub = df.index[df["compound_id"] == full_name]
        if len(sub) > 0:
            i = int(sub[0])
            ax1.scatter(coords[i, 0], coords[i, 1], c=col, s=160, marker="*", edgecolors="black", linewidth=0.8, zorder=7)
            ax1.annotate(
                short_label,
                (coords[i, 0], coords[i, 1]),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color=col,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.88),
            )

    ax1.set_title("ECFP4 Chemical Space: Novel Scaffolds vs. 10 Clinical Antibiotic Classes", fontweight="bold")
    ax1.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
    ax1.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
    ax1.legend(loc="best", fontsize=8.5, frameon=True)
    ax1.grid(alpha=0.25)

    non_clinical_sims = df.loc[~is_clinical, "max_tanimoto_abx"].to_numpy()
    ax2.hist(non_clinical_sims, bins=28, color="#0f766e", alpha=0.82, edgecolor="white")
    ax2.axvline(0.40, color="#dc2626", linestyle="--", linewidth=2.0, label="Novelty Cutoff ($T_{\\max} = 0.40$)")
    novel_pct = 100.0 * float((non_clinical_sims < 0.40).mean())
    ax2.set_title(
        f"Structural Novelty Distribution vs. 36 Clinical Antibiotics\n({novel_pct:.1f}% below $T_{{\\max}} < 0.40$ cross-resistance filter)",
        fontweight="bold",
    )
    ax2.set_xlabel("Max ECFP4 Tanimoto Similarity ($T_{\\max}$) to Closest Clinical Antibiotic")
    ax2.set_ylabel("Compound Count")
    ax2.legend(loc="upper right", fontsize=9, frameon=True)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_entry_and_selectivity(
    df_test: pd.DataFrame,
    gnn_test_preds: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Plot Gram-negative accumulation by eNTRy score & Selectivity Index landscape."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=300)

    scores = [0, 1, 2, 3]
    rates = []
    counts = []
    for s in scores:
        sub = df_test[df_test["eNTRy_Score"].round().astype(int) == s]
        counts.append(len(sub))
        rates.append(100.0 * float(sub["gram_neg_accum"].mean()) if len(sub) > 0 else 0.0)

    bars = ax1.bar(
        [f"eNTRy = {s}\n(n={c})" for s, c in zip(scores, counts)],
        rates,
        color=["#94a3b8", "#60a5fa", "#14b8a6", "#0f766e"],
        edgecolor="#1e293b",
        linewidth=0.8,
    )
    for bar, r in zip(bars, rates):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.0,
            f"{r:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )
    ax1.set_ylim(0, 112)
    ax1.set_title("Gram-Negative Outer-Membrane Accumulation vs. Richter eNTRy Score\n(Held-Out Test Scaffold Split)", fontweight="bold")
    ax1.set_ylabel("Gram-Negative Accumulation Rate (%)")
    ax1.grid(axis="y", alpha=0.25)

    pred_pmic = gnn_test_preds["pred_pMIC"]
    pred_pcc50 = gnn_test_preds["pred_pCC50"]
    si = 10.0 ** (pred_pmic - pred_pcc50)

    # Shade clinical selectivity window (pMIC - pCC50 >= 1.0 => SI >= 10)
    x_line = np.linspace(3.4, 8.5, 100)
    ax2.plot(x_line, x_line - 1.0, color="#0f766e", linestyle="--", linewidth=1.6, label="Therapeutic Window ($\\mathrm{SI} = 10$)")
    ax2.fill_between(x_line, 3.3, x_line - 1.0, color="#14b8a6", alpha=0.10)
    ax2.plot(x_line, x_line, color="#dc2626", linestyle=":", linewidth=1.4, label="Parity ($\\mathrm{SI} = 1$, Non-Selective)")

    sc = ax2.scatter(
        pred_pmic,
        pred_pcc50,
        c=np.log10(np.clip(si, 0.1, 10000.0)),
        cmap="RdYlGn",
        s=55,
        edgecolors="#1e293b",
        linewidth=0.5,
        alpha=0.88,
    )
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("$\\log_{10}(\\mathrm{Selectivity\\ Index})$", fontsize=9)

    for cid, label in [
        ("Halicin (SU3327) — Novel AI-Discovered Antibiotic", "Halicin"),
        ("Abaucin (RS102895) — Narrow-Spectrum A. baumannii Hit", "Abaucin"),
        ("Ciprofloxacin", "Ciprofloxacin"),
        ("Doxorubicin-like Cytotoxic Control (Anthracycline Core)", "Doxorubicin-like"),
    ]:
        matches = np.where(df_test["compound_id"].to_numpy() == cid)[0]
        if len(matches) > 0:
            idx = int(matches[0])
            ax2.annotate(
                label,
                (pred_pmic[idx], pred_pcc50[idx]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#334155", alpha=0.9),
            )

    ax2.set_xlim(3.5, 8.3)
    ax2.set_ylim(3.5, 7.5)
    ax2.set_title("Predicted Potency (pMIC) vs. Human Cytotoxicity (pCC50)\nSelectivity Index Landscape on Held-Out Scaffolds", fontweight="bold")
    ax2.set_xlabel("Predicted Antibacterial Potency ($\\hat{\\mathrm{pMIC}}$)")
    ax2.set_ylabel("Predicted Human HepG2 Cytotoxicity ($\\hat{\\mathrm{pCC}}_{50}$)")
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_xai_case_study(
    model: AMRGraphNet,
    scaler: DescriptorScaler,
    out_path: Path,
) -> None:
    """Render side-by-side XAI pharmacophore vs. toxicophore maps for Halicin, Abaucin, Ciprofloxacin."""
    targets = [
        ("Halicin (SU3327)", BENCHMARK_DISCOVERY_MOLECULES["Halicin (SU3327) — Novel AI-Discovered Antibiotic"]["smiles"]),
        ("Abaucin (RS102895)", BENCHMARK_DISCOVERY_MOLECULES["Abaucin (RS102895) — Narrow-Spectrum A. baumannii Hit"]["smiles"]),
        ("Ciprofloxacin", BENCHMARK_DISCOVERY_MOLECULES["Ciprofloxacin — Clinical Fluoroquinolone"]["smiles"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0), dpi=300)
    for col_idx, (name, smi) in enumerate(targets):
        attr = compute_atom_attributions(model, smi, scaler=scaler)
        graph = mol_to_graph_data(attr["mol"], scaler=scaler, standardize=False)
        pred = model.predict_with_uncertainty(graph, num_samples=20)
        nov = compute_antibiotic_novelty(attr["mol"])

        pharm_img = render_atom_attribution_image(
            attr["mol"],
            attr["abx_weights"],
            mode="pharmacophore",
            size=(460, 320),
            legend=f"P(Abx)={pred['prob_abx']:.2f}±{pred['abx_std']:.2f} | pMIC={pred['pred_pMIC']:.2f}",
        )
        tox_img = render_atom_attribution_image(
            attr["mol"],
            attr["tox_weights"],
            mode="toxicophore",
            size=(460, 320),
            legend=f"P(Tox)={pred['prob_cytotox']:.2f} | SI={pred['selectivity_index']:.1f} | Tmax={nov['max_tanimoto']:.2f}",
        )

        axes[0, col_idx].imshow(pharm_img)
        axes[0, col_idx].axis("off")
        axes[0, col_idx].set_title(f"{name}\nAntibacterial Pharmacophore Attribution", fontweight="bold", fontsize=10.5)

        axes[1, col_idx].imshow(tox_img)
        axes[1, col_idx].axis("off")
        axes[1, col_idx].set_title("Human Cytotoxicity Attribution", fontweight="bold", fontsize=10.5)

    fig.suptitle(
        "Explainable AMR Discovery (Out-of-Sample Test Scaffolds): Emerald = Antibacterial Pharmacophore | Crimson = Cytotoxic Liability",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_all_benchmarks(
    data_dir: Path = Path("data"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
    epochs: int = 55,
    seed: int = 42,
) -> Dict[str, object]:
    """Train all models, benchmark on zero-leakage test split, and save 300 DPI figures."""
    set_global_seed(seed)
    csv_path = data_dir / "processed" / "amr_curated_dataset.csv"
    npz_path = data_dir / "processed" / "amr_features.npz"
    if not csv_path.exists() or not npz_path.exists():
        prepare_dataset(data_dir=data_dir, seed=seed)

    df = pd.read_csv(csv_path)
    npz = np.load(npz_path, allow_pickle=True)
    fp_matrix = npz["morgan_fp"]
    desc_matrix = npz["descriptors"]

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    scaler = DescriptorScaler.fit(desc_matrix[train_mask])
    scaled_desc = scaler.transform(desc_matrix)
    X_all = np.hstack([fp_matrix.astype(np.float32), scaled_desc])

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_val = df.loc[val_mask].reset_index(drop=True)
    df_test = df.loc[test_mask].reset_index(drop=True)

    y_train = {
        "active_abx": df_train["active_abx"].to_numpy(),
        "pMIC": df_train["pMIC"].to_numpy(),
        "gram_neg_accum": df_train["gram_neg_accum"].to_numpy(),
        "cytotoxic_human": df_train["cytotoxic_human"].to_numpy(),
        "pCC50": df_train["pCC50"].to_numpy(),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train classical baselines
    rf_model = MultiTaskAMRBaseline(model_type="rf", random_state=seed).fit(X_all[train_mask], y_train)
    rf_model.save(models_dir / "rf_baseline.joblib")
    rf_test_preds = rf_model.predict_dict(X_all[test_mask])

    xgb_model = MultiTaskAMRBaseline(model_type="xgb", random_state=seed).fit(X_all[train_mask], y_train)
    xgb_model.save(models_dir / "xgb_baseline.joblib")
    xgb_test_preds = xgb_model.predict_dict(X_all[test_mask])

    # 2. Build PyG graph splits and train AMRGraphNet
    def _build_graph_list(sub_df: pd.DataFrame) -> List:
        graphs = []
        for _, row in sub_df.iterrows():
            labels = {
                "active_abx": row["active_abx"],
                "pMIC": row["pMIC"],
                "gram_neg_accum": row["gram_neg_accum"],
                "cytotoxic_human": row["cytotoxic_human"],
                "pCC50": row["pCC50"],
            }
            g = mol_to_graph_data(row["smiles"], scaler=scaler, labels=labels, standardize=False)
            if g is not None:
                graphs.append(g)
        return graphs

    train_graphs = _build_graph_list(df_train)
    val_graphs = _build_graph_list(df_val)
    test_graphs = _build_graph_list(df_test)

    gnn_ckpt_path = models_dir / "amr_gnn_best.pt"
    gnn_model, history = train_amr_gnn(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        df_train=df_train,
        df_val=df_val,
        scaler=scaler,
        checkpoint_path=gnn_ckpt_path,
        epochs=epochs,
        seed=seed,
    )

    device = next(gnn_model.parameters()).device
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)
    gnn_test_preds = _predict_gnn_loader(gnn_model, test_loader, device)

    # 3. Compile scaffold-split benchmark metrics
    metrics_payload = {
        "dataset_summary": {
            "total_compounds": int(len(df)),
            "train_count": int(len(df_train)),
            "val_count": int(len(df_val)),
            "test_count": int(len(df_test)),
            "unique_scaffolds": int(df["murcko_scaffold"].nunique()),
            "zero_scaffold_leakage": True,
        },
        "test_metrics": {
            "AMRGraphNet": _evaluate_predictions(df_test, gnn_test_preds),
            "XGBoost": _evaluate_predictions(df_test, xgb_test_preds),
            "RandomForest": _evaluate_predictions(df_test, rf_test_preds),
        },
        "training_history": history,
    }
    (models_dir / "benchmark_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    # 4. Generate 4 publication-grade 300 DPI figures
    _plot_roc_pr_benchmarks(
        df_test,
        {
            "AMRGraphNet (GATv2+GIN)": gnn_test_preds,
            "XGBoost (ECFP4+eNTRy)": xgb_test_preds,
            "RandomForest (ECFP4+eNTRy)": rf_test_preds,
        },
        figures_dir / "roc_pr_benchmark.png",
    )
    _plot_chemical_space_and_novelty(df, fp_matrix, figures_dir / "chemical_space_novelty.png")
    _plot_entry_and_selectivity(df_test, gnn_test_preds, figures_dir / "entry_selectivity_analysis.png")
    _plot_xai_case_study(gnn_model, scaler, figures_dir / "halicin_abaucin_xai.png")

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PathoGraph-AMR models and run benchmarks.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = run_all_benchmarks(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps(payload["test_metrics"], indent=2))


if __name__ == "__main__":
    main()

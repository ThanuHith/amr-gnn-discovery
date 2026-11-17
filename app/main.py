"""PathoGraph-AMR: Interactive Streamlit Workbench for Novel Antibiotic Discovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_pipeline.prepare_amr_dataset import compute_amr_descriptors, standardize_smiles
from src.data_pipeline.reference_antibiotics import (
    BENCHMARK_DISCOVERY_MOLECULES,
    compute_antibiotic_novelty,
)
from src.generative.antibiotic_designer import compute_amr_discovery_score, eNTRyAntibioticDesigner
from src.models.amr_gnn import AMRGraphNet
from src.models.featurizer import DescriptorScaler, mol_to_graph_data
from src.xai.substructure_explainer import (
    audit_structural_alerts_and_entry,
    compute_atom_attributions,
    render_atom_attribution_image,
)

MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


st.set_page_config(
    page_title="PathoGraph-AMR | AI Antibiotic Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_amr_artifacts() -> Tuple[AMRGraphNet, DescriptorScaler, Dict]:
    """Load trained AMRGraphNet checkpoint, train-fit DescriptorScaler, and metrics."""
    ckpt_path = MODELS_DIR / "amr_gnn_best.pt"
    metrics_path = MODELS_DIR / "benchmark_metrics.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint missing at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = AMRGraphNet(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        global_dim=cfg["global_dim"],
        hidden_dim=cfg["hidden_dim"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler = DescriptorScaler.from_state_dict(ckpt["scaler_state_dict"])

    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return model, scaler, metrics


@st.cache_data
def load_curated_dataset() -> pd.DataFrame:
    csv_path = DATA_DIR / "processed" / "amr_curated_dataset.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _profile_single_molecule(
    smiles: str,
    model: AMRGraphNet,
    scaler: DescriptorScaler,
    mc_samples: int,
    novelty_threshold: float,
) -> Dict[str, object]:
    std = standardize_smiles(smiles)
    if std is None:
        raise ValueError("Could not parse or standardize SMILES. Check chemical valence and syntax.")
    canon_smi, mol = std
    desc = compute_amr_descriptors(mol)
    novelty = compute_antibiotic_novelty(mol, novelty_threshold=novelty_threshold)
    graph = mol_to_graph_data(mol, scaler=scaler, standardize=False)
    pred = model.predict_with_uncertainty(graph, num_samples=mc_samples)
    audit = audit_structural_alerts_and_entry(canon_smi)
    amr_score = compute_amr_discovery_score(
        prob_abx=pred["prob_abx"],
        pred_pmic=pred["pred_pMIC"],
        prob_cytotox=pred["prob_cytotox"],
        entry_score=desc["eNTRy_Score"],
        max_tanimoto=novelty["max_tanimoto"],
    )
    return {
        "smiles": canon_smi,
        "mol": mol,
        "descriptors": desc,
        "novelty": novelty,
        "prediction": pred,
        "audit": audit,
        "amr_score": amr_score,
    }


def main() -> None:
    st.title("PathoGraph-AMR: Graph Deep Learning for Novel Antibiotic Discovery")
    st.caption(
        "Multi-task GATv2+GIN architecture predicting antibacterial potency (pMIC), Gram-negative outer-membrane "
        "accumulation (Richter eNTRy rules), human HepG2 cytotoxicity, Monte Carlo epistemic uncertainty, and structural novelty."
    )

    try:
        model, scaler, metrics = load_amr_artifacts()
    except FileNotFoundError:
        st.error(
            "Trained model artifacts were not found in `models/`. "
            "Run `py -m src.evaluation.run_benchmarks` in the repository root first."
        )
        return

    df_dataset = load_curated_dataset()

    # --- Sidebar Controls ---
    st.sidebar.header("Query Compound & Inference")
    preset_names = list(BENCHMARK_DISCOVERY_MOLECULES.keys())
    selected_preset = st.sidebar.selectbox("Benchmark / Showcase Molecule", preset_names, index=0)
    preset_info = BENCHMARK_DISCOVERY_MOLECULES[selected_preset]

    custom_smiles = st.sidebar.text_input(
        "Canonical or Custom SMILES",
        value=preset_info["smiles"],
        help="Paste any valid SMILES string to run live graph inference, XAI attribution, and de novo optimization.",
    )
    mc_samples = st.sidebar.slider("MC Dropout Epistemic Passes (T)", min_value=5, max_value=50, value=20, step=5)
    novelty_cutoff = st.sidebar.slider(
        "Structural Novelty Cutoff (T_max)",
        min_value=0.20,
        max_value=0.65,
        value=0.40,
        step=0.05,
        help="Compounds with T_max below this threshold vs. 36 FDA/EMA clinical antibiotics are flagged as novel scaffolds.",
    )

    st.sidebar.markdown(f"**Category:** {preset_info['category']}")
    st.sidebar.caption(preset_info["notes"])

    try:
        profile = _profile_single_molecule(custom_smiles, model, scaler, mc_samples, novelty_cutoff)
    except Exception as exc:
        st.error(f"SMILES Error: {exc}")
        return

    mol = profile["mol"]
    pred = profile["prediction"]
    desc = profile["descriptors"]
    nov = profile["novelty"]
    audit = profile["audit"]

    sidebar_img = Draw.MolToImage(mol, size=(280, 200))
    st.sidebar.image(sidebar_img, caption=f"Standardized: {profile['smiles']}", use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1. AMR & Safety Profiler",
            "2. Explainable AI (XAI) & eNTRy Audit",
            "3. Virtual Screening",
            "4. De Novo Lead Optimizer",
            "5. Scaffold Benchmarks & Research Fit",
        ]
    )

    # === TAB 1: PROFILER ===
    with tab1:
        st.subheader("Multi-Task Graph Prediction & Clinical Selectivity Profile")
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        c1.metric(
            "Antibacterial Prob.",
            f"{pred['prob_abx']:.1%}",
            f"± {pred['abx_std']:.3f} σ",
        )
        c2.metric(
            "Predicted Potency",
            f"pMIC {pred['pred_pMIC']:.2f}",
            f"MIC ≈ {pred['mic_uM']:.2f} µM",
        )
        c3.metric(
            "Human Cytotoxicity",
            f"{pred['prob_cytotox']:.1%}",
            f"pCC50 {pred['pred_pCC50']:.2f}",
            delta_color="inverse",
        )
        si_val = pred["selectivity_index"]
        si_label = "Favorable Window" if si_val >= 10.0 else "Narrow / Toxic Risk"
        c4.metric(
            "Selectivity Index (SI)",
            f"{si_val:.1f}x",
            si_label,
            delta_color="normal" if si_val >= 10.0 else "inverse",
        )
        c5.metric(
            "Gram-Neg eNTRy",
            f"{int(desc['eNTRy_Score'])} / 3",
            f"Entry P={pred['prob_gram_neg']:.1%}",
        )
        nov_tag = "Novel Scaffold" if nov["is_novel_scaffold"] else f"Near {nov['nearest_class']}"
        c6.metric(
            "Scaffold Novelty",
            f"Tmax = {nov['max_tanimoto']:.2f}",
            nov_tag,
        )

        left_col, right_col = st.columns([1.2, 1.0])
        with left_col:
            st.markdown("#### Unified Discovery Assessment")
            st.progress(min(max(profile["amr_score"], 0.0), 1.0))
            st.markdown(
                f"**Composite AMR Discovery Score:** `{profile['amr_score']:.3f} / 1.000`  \n"
                f"**Nearest Clinical Antibiotic:** `{nov['nearest_antibiotic']}` ({nov['nearest_class']}, "
                f"ECFP4 Tanimoto = `{nov['max_tanimoto']:.3f}`)  \n"
                f"**Cross-Resistance Risk Assessment:** "
                + (
                    "Low structural overlap ($T_{\\max} < %.2f$) — distinct scaffold likely to bypass existing target-modification resistance."
                    % novelty_cutoff
                    if nov["is_novel_scaffold"]
                    else "Shares substantial structural homology with an existing clinical antibiotic class."
                )
            )

        with right_col:
            st.markdown("#### Physicochemical & eNTRy Descriptors")
            desc_df = pd.DataFrame(
                [{"Descriptor": k, "Value": v} for k, v in desc.items()]
            )
            st.dataframe(desc_df, use_container_width=True, height=290, hide_index=True)

    # === TAB 2: EXPLAINABLE AI ===
    with tab2:
        st.subheader("Substructure Rationalization: Pharmacophore vs. Toxicophore Mapping")
        st.caption(
            "Batched counterfactual atom-masking attribution computed directly on AMRGraphNet. "
            "Emerald highlights atoms driving antibacterial growth inhibition; Crimson highlights human cytotoxicity liabilities."
        )
        with st.spinner("Computing atom-level graph attributions..."):
            attr = compute_atom_attributions(model, mol, scaler=scaler)

        col_ph, col_tx = st.columns(2)
        with col_ph:
            img_ph = render_atom_attribution_image(
                mol,
                attr["abx_weights"],
                mode="pharmacophore",
                size=(540, 380),
                legend=f"Antibacterial Pharmacophore | P(Abx)={pred['prob_abx']:.2f}",
            )
            st.image(img_ph, use_container_width=True)
            st.caption("Emerald = Positive antibacterial contribution | Amber = Activity-attenuating")

        with col_tx:
            img_tx = render_atom_attribution_image(
                mol,
                attr["tox_weights"],
                mode="toxicophore",
                size=(540, 380),
                legend=f"Human Cytotoxicity Liability | P(Tox)={pred['prob_cytotox']:.2f}",
            )
            st.image(img_tx, use_container_width=True)
            st.caption("Crimson = Cytotoxic liability driver | Sky-Blue = Safety-conferring")

        st.markdown("---")
        audit_col1, audit_col2 = st.columns([1.3, 1.0])
        with audit_col1:
            st.markdown("#### Richter et al. (Nature 2017) Gram-Negative eNTRy Rule Checklist")
            chk_df = pd.DataFrame(audit["entry_checklist"])
            chk_df["Status"] = chk_df["passed"].map(lambda x: "PASS" if x else "FAIL")
            st.dataframe(
                chk_df[["criterion", "measured", "threshold", "Status"]],
                use_container_width=True,
                hide_index=True,
            )

        with audit_col2:
            st.markdown("#### Medicinal Chemistry PAINS & Brenk Alert Audit")
            pains = audit["pains_alerts"]
            brenk = audit["brenk_alerts"]
            if not pains and not brenk:
                st.success("Zero PAINS or Brenk structural alerts detected.")
            else:
                if pains:
                    st.warning(f"**PAINS Alerts ({len(pains)}):** " + ", ".join(pains))
                if brenk:
                    st.info(f"**Brenk Alerts ({len(brenk)}):** " + ", ".join(brenk))
            st.caption(
                "Note: Clinical nitro-heterocyclic antibiotics (e.g., Nitrofurantoin, Halicin) intentionally trigger "
                "Brenk nitro alerts because bacterial nitroreductase activation relies on the nitro warhead."
            )

    # === TAB 3: VIRTUAL SCREENING ===
    with tab3:
        st.subheader("Virtual Screening & Novel Hit Prioritization")
        screen_mode = st.radio(
            "Screening Library Source",
            ["Held-Out Zero-Leakage Test Split", "Custom Batch SMILES Input"],
            horizontal=True,
        )

        screen_df = pd.DataFrame()
        if screen_mode == "Held-Out Zero-Leakage Test Split":
            if df_dataset.empty:
                st.warning("Processed dataset CSV not found.")
            else:
                test_sub = df_dataset[df_dataset["split"] == "test"].copy()
                rows = []
                for _, r in test_sub.iterrows():
                    g = mol_to_graph_data(r["smiles"], scaler=scaler, standardize=False)
                    if g is None:
                        continue
                    p = model.predict_with_uncertainty(g, num_samples=1)
                    rows.append(
                        {
                            "compound_id": r["compound_id"],
                            "smiles": r["smiles"],
                            "pred_P_Abx": round(p["prob_abx"], 3),
                            "pred_pMIC": round(p["pred_pMIC"], 2),
                            "pred_P_GramNeg": round(p["prob_gram_neg"], 3),
                            "pred_P_Tox": round(p["prob_cytotox"], 3),
                            "Selectivity_Index": round(p["selectivity_index"], 1),
                            "eNTRy_Score": int(r["eNTRy_Score"]),
                            "T_max_Clinical": round(float(r["max_tanimoto_abx"]), 3),
                            "Nearest_Clinical_Abx": r["nearest_clinical_abx"],
                            "Novel_Scaffold": bool(r["max_tanimoto_abx"] < novelty_cutoff),
                        }
                    )
                screen_df = pd.DataFrame(rows)
        else:
            default_batch = "\n".join(
                [
                    "NC1=NN=C(S1)SC2=NC=C(S2)[N+](=O)[O-]",
                    "C1=CC2=C(C=C1)NC(=O)OC32CCN(CCC4=CC=C(C=C4)C(F)(F)F)CC3",
                    "Cc1ccc2ccc(N)cc2n1",
                    "C1CC1N2C=C(C(=O)C3=C2C=C(C(=C3)F)N4CCNCC4)C(=O)O",
                    "O=C1c2ccccc2C(=O)c2c(O)cccc12",
                    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
                ]
            )
            batch_txt = st.text_area("Paste SMILES (one per line):", value=default_batch, height=140)
            if st.button("Screen Batch with AMRGraphNet", type="primary"):
                rows = []
                for idx, line in enumerate(batch_txt.splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        res = _profile_single_molecule(line, model, scaler, mc_samples=8, novelty_threshold=novelty_cutoff)
                        p = res["prediction"]
                        d = res["descriptors"]
                        nv = res["novelty"]
                        rows.append(
                            {
                                "compound_id": f"Query_{idx}",
                                "smiles": res["smiles"],
                                "pred_P_Abx": round(p["prob_abx"], 3),
                                "pred_pMIC": round(p["pred_pMIC"], 2),
                                "pred_P_GramNeg": round(p["prob_gram_neg"], 3),
                                "pred_P_Tox": round(p["prob_cytotox"], 3),
                                "Selectivity_Index": round(p["selectivity_index"], 1),
                                "eNTRy_Score": int(d["eNTRy_Score"]),
                                "T_max_Clinical": round(float(nv["max_tanimoto"]), 3),
                                "Nearest_Clinical_Abx": nv["nearest_antibiotic"],
                                "Novel_Scaffold": bool(nv["is_novel_scaffold"]),
                            }
                        )
                    except Exception:
                        continue
                st.session_state["custom_screen_df"] = pd.DataFrame(rows)
            screen_df = st.session_state.get("custom_screen_df", pd.DataFrame())

        if not screen_df.empty:
            fc1, fc2, fc3, fc4 = st.columns(4)
            min_abx = fc1.slider("Min P(Antibacterial)", 0.0, 0.95, 0.50, 0.05)
            max_tmax = fc2.slider("Max T_max (Novelty)", 0.20, 1.00, float(novelty_cutoff), 0.05)
            min_si = fc3.slider("Min Selectivity Index", 1.0, 100.0, 5.0, 1.0)
            min_entry = fc4.slider("Min eNTRy Score", 0, 3, 1, 1)

            filtered = screen_df[
                (screen_df["pred_P_Abx"] >= min_abx)
                & (screen_df["T_max_Clinical"] <= max_tmax)
                & (screen_df["Selectivity_Index"] >= min_si)
                & (screen_df["eNTRy_Score"] >= min_entry)
            ].sort_values(by=["pred_P_Abx", "Selectivity_Index"], ascending=False)

            st.markdown(f"**Prioritized Novel Hits:** `{len(filtered)}` of `{len(screen_df)}` screened molecules")

            fig_sc, ax_sc = plt.subplots(figsize=(8.5, 4.0))
            sc = ax_sc.scatter(
                screen_df["pred_pMIC"],
                screen_df["pred_P_Tox"],
                c=screen_df["T_max_Clinical"],
                cmap="viridis_r",
                s=55,
                alpha=0.85,
                edgecolors="#1e293b",
                linewidth=0.5,
            )
            plt.colorbar(sc, ax=ax_sc, label="T_max vs Clinical Antibiotics")
            ax_sc.set_xlabel("Predicted pMIC")
            ax_sc.set_ylabel("Predicted Human Cytotoxicity P(Tox)")
            ax_sc.set_title("Virtual Screening Landscape (Ideal Hits: Bottom-Right, Bright Yellow/Green Novelty)")
            ax_sc.grid(alpha=0.25)
            st.pyplot(fig_sc, use_container_width=False)
            plt.close(fig_sc)

            st.dataframe(filtered, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Prioritized Hits (CSV)",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="pathograph_amr_prioritized_hits.csv",
                mime="text/csv",
            )

    # === TAB 4: DE NOVO DESIGNER ===
    with tab4:
        st.subheader("eNTRy-Guided De Novo Antibiotic Analog Optimizer")
        st.caption(
            "Combines medicinal-chemistry bioisosteric SMARTS transformations (installing unhindered primary amines, "
            "detoxifying nitro/lipophilic motifs) with SELFIES token mutations, scored by AMRGraphNet."
        )
        d1, d2, d3 = st.columns([1.0, 1.0, 1.2])
        top_k = d1.slider("Top Ranked Analogs", 5, 20, 10, 1)
        enforce_novel = d2.checkbox("Require Novel Scaffold (T_max < 0.40)", value=True)
        run_opt = d3.button("Generate & Rank Optimized Analogs", type="primary")

        if run_opt:
            with st.spinner("Mutating scaffold, enforcing eNTRy rules, and scoring with MC Dropout..."):
                designer = eNTRyAntibioticDesigner(model=model, scaler=scaler, seed=42)
                analogs_df = designer.design_analogs(
                    seed_smiles=profile["smiles"],
                    top_k=top_k,
                    mc_samples=10,
                    require_novel_scaffold=enforce_novel,
                )
            st.session_state["analogs_df"] = analogs_df

        analogs_df = st.session_state.get("analogs_df", pd.DataFrame())
        if not analogs_df.empty:
            st.markdown(
                f"**Seed Molecule Baseline:** `AMR Score = {profile['amr_score']:.3f}` | "
                f"`pMIC = {pred['pred_pMIC']:.2f}` | `eNTRy = {int(desc['eNTRy_Score'])}/3` | "
                f"`SI = {pred['selectivity_index']:.1f}x`"
            )
            top_mols = [Chem.MolFromSmiles(s) for s in analogs_df["smiles"].head(6)]
            legends = [
                f"#{i+1} | Score={row['AMR_Score']:.2f} | eNTRy={row['eNTRy_Score']}/3 | SI={row['selectivity_index']:.1f}x"
                for i, (_, row) in enumerate(analogs_df.head(6).iterrows())
            ]
            grid_img = Draw.MolsToGridImage(top_mols, molsPerRow=3, subImgSize=(340, 240), legends=legends)
            st.image(grid_img, caption="Top 6 Generated Analogs", use_container_width=True)
            st.dataframe(analogs_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Optimized Analogs (CSV)",
                data=analogs_df.to_csv(index=False).encode("utf-8"),
                file_name="pathograph_amr_de_novo_analogs.csv",
                mime="text/csv",
            )
        else:
            st.info("Click **Generate & Rank Optimized Analogs** above to optimize the active compound.")

    # === TAB 5: BENCHMARKS & RESEARCH FIT ===
    with tab5:
        st.subheader("Zero-Leakage Bemis-Murcko Scaffold-Split Evaluation")
        if metrics and "test_metrics" in metrics:
            tm = metrics["test_metrics"]
            cls_rows = []
            for m_name in ["AMRGraphNet", "XGBoost", "RandomForest"]:
                for task in ["active_abx", "gram_neg_accum", "cytotoxic_human"]:
                    m_vals = tm[m_name][task]
                    cls_rows.append(
                        {
                            "Model": m_name,
                            "Task": task,
                            "ROC-AUC": m_vals["roc_auc"],
                            "PR-AUC": m_vals["pr_auc"],
                            "F1": m_vals["f1"],
                            "Brier Score": m_vals["brier"],
                        }
                    )
            reg_rows = []
            for m_name in ["AMRGraphNet", "XGBoost", "RandomForest"]:
                for task in ["pMIC", "pCC50"]:
                    m_vals = tm[m_name][task]
                    reg_rows.append(
                        {
                            "Model": m_name,
                            "Endpoint": task,
                            "RMSE": m_vals["rmse"],
                            "MAE": m_vals["mae"],
                            "Spearman r": m_vals["spearman_r"],
                        }
                    )

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown("#### Classification Benchmarks (Test Scaffold Split)")
                st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)
            with m_col2:
                st.markdown("#### Potency & Cytotoxicity Regression Benchmarks")
                st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Publication-Grade 300 DPI Diagnostic Figures")
        fig_files = [
            ("roc_pr_benchmark.png", "Figure 1: Scaffold-Split ROC & Precision-Recall Curves (AMRGraphNet vs. XGBoost & RandomForest)"),
            ("chemical_space_novelty.png", "Figure 2: ECFP4 Chemical Space & Structural Novelty vs. 36 Clinical Antibiotics"),
            ("entry_selectivity_analysis.png", "Figure 3: Richter eNTRy Permeation Rules & Clinical Selectivity Landscape"),
            ("halicin_abaucin_xai.png", "Figure 4: Out-of-Sample XAI Attribution Case Study (Halicin, Abaucin, Ciprofloxacin)"),
        ]
        for fname, caption in fig_files:
            fpath = FIGURES_DIR / fname
            if fpath.exists():
                st.image(str(fpath), caption=caption, use_container_width=True)

        st.markdown("#### Translational Relevance for European / German AMR Research")
        st.info(
            "Designed to address key bottlenecks targeted by German infection & pharmaceutical research centers "
            "(**HIPS Saarbrücken**, **DZIF**, **Helmholtz Munich**, **Universität Tübingen**, **DKFZ/Heidelberg**, "
            "and **Evotec / Boehringer Ingelheim**): overcoming Gram-negative outer-membrane impermeability, "
            "filtering cross-resistant clinical scaffolds ($T_{\\max} < 0.40$), and eliminating mammalian cytotoxicity early."
        )


if __name__ == "__main__":
    main()

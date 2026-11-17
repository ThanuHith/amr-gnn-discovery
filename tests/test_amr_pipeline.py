"""Unit tests for PathoGraph-AMR data curation, GNN, XAI, and de novo designer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rdkit import Chem

from src.data_pipeline.prepare_amr_dataset import (
    DESCRIPTOR_NAMES,
    compute_amr_descriptors,
    prepare_dataset,
    standardize_smiles,
)
from src.data_pipeline.reference_antibiotics import (
    BENCHMARK_DISCOVERY_MOLECULES,
    CLINICAL_ANTIBIOTICS,
    compute_antibiotic_novelty,
)
from src.generative.antibiotic_designer import eNTRyAntibioticDesigner
from src.models.amr_gnn import AMRGraphNet
from src.models.featurizer import (
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    DescriptorScaler,
    mol_to_graph_data,
)
from src.xai.substructure_explainer import (
    audit_structural_alerts_and_entry,
    compute_atom_attributions,
    render_atom_attribution_image,
)


class TestAMRPipeline(unittest.TestCase):
    """Automated verification of scientific correctness and model contracts."""

    def test_reference_panel_and_novelty(self) -> None:
        self.assertGreaterEqual(len(CLINICAL_ANTIBIOTICS), 35)
        for abx in CLINICAL_ANTIBIOTICS:
            self.assertIsNotNone(Chem.MolFromSmiles(abx.smiles), f"Invalid clinical SMILES: {abx.name}")

        halicin_smi = BENCHMARK_DISCOVERY_MOLECULES[
            "Halicin (SU3327) — Novel AI-Discovered Antibiotic"
        ]["smiles"]
        nov_halicin = compute_antibiotic_novelty(halicin_smi)
        self.assertTrue(nov_halicin["is_novel_scaffold"])
        self.assertLess(nov_halicin["max_tanimoto"], 0.40)

        cipro_smi = BENCHMARK_DISCOVERY_MOLECULES["Ciprofloxacin — Clinical Fluoroquinolone"]["smiles"]
        nov_cipro = compute_antibiotic_novelty(cipro_smi)
        self.assertFalse(nov_cipro["is_novel_scaffold"])
        self.assertGreater(nov_cipro["max_tanimoto"], 0.99)

    def test_standardization_and_entry_descriptors(self) -> None:
        # Salt stripping & neutralization
        std = standardize_smiles("CC(=O)O.[Na+]")
        self.assertIsNotNone(std)
        canon_smi, _ = std
        self.assertEqual(canon_smi, "CC(=O)O")

        # 6-amino-2-methylquinoline (Richter et al. primary amine + rigid core)
        _, mol_q = standardize_smiles("Cc1ccc2ccc(N)cc2n1")
        desc_q = compute_amr_descriptors(mol_q)
        self.assertEqual(len(desc_q), len(DESCRIPTOR_NAMES))
        self.assertEqual(desc_q["HasPrimaryAmine"], 1.0)
        self.assertLessEqual(desc_q["RotBonds"], 5.0)
        self.assertEqual(desc_q["eNTRy_Score"], 3.0)

        # Ibuprofen lacks primary amine
        _, mol_ibu = standardize_smiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
        desc_ibu = compute_amr_descriptors(mol_ibu)
        self.assertEqual(desc_ibu["HasPrimaryAmine"], 0.0)
        self.assertLess(desc_ibu["eNTRy_Score"], 3.0)

    def test_zero_scaffold_leakage_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = prepare_dataset(Path(tmpdir) / "data", seed=42)
            self.assertTrue(stats["zero_scaffold_leakage"])
            df = pd.read_csv(Path(tmpdir) / "data" / "processed" / "amr_curated_dataset.csv")

            train_s = set(df.loc[df["split"] == "train", "murcko_scaffold"])
            val_s = set(df.loc[df["split"] == "val", "murcko_scaffold"])
            test_s = set(df.loc[df["split"] == "test", "murcko_scaffold"])

            self.assertEqual(len(train_s & val_s), 0)
            self.assertEqual(len(train_s & test_s), 0)
            self.assertEqual(len(val_s & test_s), 0)

            # Verify Halicin & Abaucin are held out in test split
            halicin_split = set(
                df.loc[
                    df["compound_id"] == "Halicin (SU3327) — Novel AI-Discovered Antibiotic",
                    "split",
                ]
            )
            self.assertEqual(halicin_split, {"test"})

    def test_gnn_forward_and_mc_dropout(self) -> None:
        scaler = DescriptorScaler(
            mean=np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32),
            std=np.ones(GLOBAL_FEATURE_DIM, dtype=np.float32),
        )
        graph = mol_to_graph_data("NC1=NN=C(S1)SC2=NC=C(S2)[N+](=O)[O-]", scaler=scaler)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.x.shape[1], NODE_FEATURE_DIM)
        self.assertEqual(graph.edge_attr.shape[1], EDGE_FEATURE_DIM)
        self.assertEqual(graph.u.shape, (1, GLOBAL_FEATURE_DIM))

        model = AMRGraphNet()
        pred = model.predict_with_uncertainty(graph, num_samples=8)
        self.assertGreaterEqual(pred["prob_abx"], 0.0)
        self.assertLessEqual(pred["prob_abx"], 1.0)
        self.assertGreater(pred["abx_std"], 0.0)
        self.assertEqual(pred["graph_embedding"].shape, (128,))

    def test_xai_attribution_and_audit(self) -> None:
        model = AMRGraphNet()
        halicin_smi = "NC1=NN=C(S1)SC2=NC=C(S2)[N+](=O)[O-]"
        attr = compute_atom_attributions(model, halicin_smi)
        self.assertEqual(len(attr["abx_weights"]), attr["mol"].GetNumAtoms())
        self.assertEqual(len(attr["tox_weights"]), attr["mol"].GetNumAtoms())

        img = render_atom_attribution_image(attr["mol"], attr["abx_weights"], mode="pharmacophore")
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (480, 340))

        audit = audit_structural_alerts_and_entry(halicin_smi)
        self.assertGreaterEqual(audit["alert_count"], 1)
        self.assertEqual(len(audit["entry_checklist"]), 3)

    def test_de_novo_designer(self) -> None:
        model = AMRGraphNet()
        designer = eNTRyAntibioticDesigner(model=model, scaler=None, seed=42)
        analogs = designer.design_analogs(
            seed_smiles="Cc1ccc2ccc(N)cc2n1",
            top_k=5,
            mc_samples=3,
            require_novel_scaffold=False,
        )
        self.assertFalse(analogs.empty)
        for smi in analogs["smiles"]:
            self.assertIsNotNone(Chem.MolFromSmiles(smi))


if __name__ == "__main__":
    unittest.main()

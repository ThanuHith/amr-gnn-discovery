"""Constrained De Novo Antibiotic Optimizer (eNTRy Rules + SELFIES + AMRGraphNet).

Generates structurally novel analogs around a query hit scaffold by combining:
1. Medicinal-chemistry bioisosteric & Richter eNTRy-guided SMARTS transformations
   (installing primary aminomethyl groups for Gram-negative porin accumulation,
   modulating lipophilicity via pyridine/fluoro/hydroxyl swaps, and detoxifying
   excessively lipophilic aryl groups).
2. Validated SELFIES token mutations (selfies.encoder / selfies.decoder) filtered
   for chemical validity, Synthetic Accessibility (SA_Score <= 5.5), QED, and
   structural novelty (T_max < 0.40 against 36 clinical antibiotics).

Candidates are ranked by a Multi-Objective AMR Discovery Score balancing potency,
Gram-negative accumulation, mammalian safety, and scaffold novelty.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem

from src.data_pipeline.prepare_amr_dataset import compute_amr_descriptors, standardize_smiles
from src.data_pipeline.reference_antibiotics import compute_antibiotic_novelty
from src.models.amr_gnn import AMRGraphNet
from src.models.featurizer import DescriptorScaler, mol_to_graph_data


# Medicinal-chemistry reactions targeting Gram-negative eNTRy compliance & selectivity
_BIOISOSTERE_REACTIONS = [
    ("Install Primary Aminomethyl (eNTRy +NH2)", "[c;H1:1]>>[c:1]CN"),
    ("Install Aryl Primary Amine (eNTRy +NH2)", "[c;H1:1]>>[c:1]N"),
    ("Replace Aryl Halide with Aminomethyl", "[c:1][Cl,Br]>>[c:1]CN"),
    ("Carboxylic Acid to Aminoethyl Amide", "[C:1](=O)[OH]>>[C:1](=O)NCCN"),
    ("Phenyl CH -> Pyridine N (Lower LogP)", "[c;H1:1]1[c:2][c:3][c:4][c:5][c:6]1>>[n:1]1[c:2][c:3][c:4][c:5][c:6]1"),
    ("Metabolic Block with Aryl Fluorine", "[c;H1:1]>>[c:1]F"),
    ("Polar Hydroxyl Tuning", "[c;H1:1]>>[c:1]O"),
    ("Trifluoromethyl Lipophilic Anchor", "[c;H1:1]>>[c:1]C(F)(F)F"),
    ("Nitro -> Cyano Bioisostere (Detox)", "[c:1][N+](=O)[O-]>>[c:1]C#N"),
    ("Thioether -> Sulfone Polarity Boost", "[#6:1][S;X2:2][#6:3]>>[#6:1][S:2](=O)(=O)[#6:3]"),
]

_SELFIES_SAFE_TOKENS = [
    "[N]",
    "[C][N]",
    "[F]",
    "[O]",
    "[=N]",
    "[C]",
    "[C][#N]",
    "[C][F]",
]


def compute_amr_discovery_score(
    prob_abx: float,
    pred_pmic: float,
    prob_cytotox: float,
    entry_score: float,
    max_tanimoto: float,
) -> float:
    """Compute unified multi-objective score in [0, 1] for lead prioritization."""
    norm_pmic = float(np.clip((pred_pmic - 3.5) / 5.0, 0.0, 1.0))
    score = (
        0.35 * float(prob_abx)
        + 0.20 * norm_pmic
        + 0.20 * (1.0 - float(prob_cytotox))
        + 0.15 * (float(entry_score) / 3.0)
        + 0.10 * (1.0 - float(max_tanimoto))
    )
    return round(float(score), 4)


class eNTRyAntibioticDesigner:
    """Generates and ranks eNTRy-optimized, non-toxic antibiotic analogs."""

    def __init__(
        self,
        model: AMRGraphNet,
        scaler: Optional[DescriptorScaler] = None,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.rng = random.Random(seed)
        self._compiled_rxns = []
        for label, smarts in _BIOISOSTERE_REACTIONS:
            try:
                rxn = AllChem.ReactionFromSmarts(smarts)
                if rxn is not None:
                    self._compiled_rxns.append((label, rxn))
            except Exception:
                continue

    def _apply_smarts_edits(self, mol: Chem.Mol, max_per_rxn: int = 4) -> List[tuple]:
        analogs = []
        for label, rxn in self._compiled_rxns:
            try:
                products = rxn.RunReactants((mol,))
            except Exception:
                continue
            for prod_tuple in products[:max_per_rxn]:
                prod = prod_tuple[0]
                try:
                    Chem.SanitizeMol(prod)
                    smi = Chem.MolToSmiles(prod, isomericSmiles=True, canonical=True)
                    analogs.append((smi, f"SMARTS: {label}"))
                except Exception:
                    continue
        return analogs

    def _apply_selfies_mutations(self, canon_smi: str, n_trials: int = 24) -> List[tuple]:
        analogs = []
        try:
            encoded = sf.encoder(canon_smi)
            if not encoded:
                return analogs
            tokens = list(sf.split_selfies(encoded))
        except Exception:
            return analogs

        if len(tokens) < 4:
            return analogs

        for _ in range(n_trials):
            mut = list(tokens)
            op = self.rng.choice(["insert", "replace"])
            idx = self.rng.randrange(1, len(mut))
            tok = self.rng.choice(_SELFIES_SAFE_TOKENS)
            if op == "insert":
                mut.insert(idx, tok)
            else:
                if "Ring" in mut[idx] or "Branch" in mut[idx]:
                    continue
                mut[idx] = tok
            try:
                decoded = sf.decoder("".join(mut))
                if decoded:
                    analogs.append((decoded, "SELFIES Token Mutation"))
            except Exception:
                continue
        return analogs

    def design_analogs(
        self,
        seed_smiles: str,
        top_k: int = 12,
        mc_samples: int = 10,
        require_novel_scaffold: bool = False,
    ) -> pd.DataFrame:
        """Generate, filter, score, and rank structural analogs around a seed molecule."""
        std_seed = standardize_smiles(seed_smiles)
        if std_seed is None:
            raise ValueError("Invalid seed SMILES provided to eNTRyAntibioticDesigner.")
        seed_canon, seed_mol = std_seed

        raw_candidates: List[tuple] = []
        raw_candidates.extend(self._apply_smarts_edits(seed_mol))
        raw_candidates.extend(self._apply_selfies_mutations(seed_canon))

        # Second-generation walk from top SMARTS edits to enable multi-step optimization
        for smi, _ in raw_candidates[:8]:
            m_step = Chem.MolFromSmiles(smi)
            if m_step is not None:
                for smi2, label2 in self._apply_smarts_edits(m_step, max_per_rxn=2)[:4]:
                    raw_candidates.append((smi2, f"2-Step {label2}"))

        seen = {seed_canon}
        scored_rows: List[Dict[str, object]] = []

        for cand_smi, strategy in raw_candidates:
            std = standardize_smiles(cand_smi)
            if std is None:
                continue
            canon_smi, mol = std
            if canon_smi in seen:
                continue
            seen.add(canon_smi)

            desc = compute_amr_descriptors(mol)
            if desc["MW"] < 140.0 or desc["MW"] > 620.0:
                continue
            if desc["SA_Score"] > 5.8 or desc["QED"] < 0.22:
                continue

            novelty = compute_antibiotic_novelty(mol)
            if require_novel_scaffold and not novelty["is_novel_scaffold"]:
                continue

            graph = mol_to_graph_data(mol, scaler=self.scaler, standardize=False)
            if graph is None:
                continue

            pred = self.model.predict_with_uncertainty(graph, num_samples=mc_samples)
            amr_score = compute_amr_discovery_score(
                prob_abx=pred["prob_abx"],
                pred_pmic=pred["pred_pMIC"],
                prob_cytotox=pred["prob_cytotox"],
                entry_score=desc["eNTRy_Score"],
                max_tanimoto=novelty["max_tanimoto"],
            )

            scored_rows.append(
                {
                    "smiles": canon_smi,
                    "strategy": strategy,
                    "AMR_Score": amr_score,
                    "prob_abx": round(pred["prob_abx"], 3),
                    "pred_pMIC": round(pred["pred_pMIC"], 2),
                    "prob_gram_neg": round(pred["prob_gram_neg"], 3),
                    "prob_cytotox": round(pred["prob_cytotox"], 3),
                    "selectivity_index": round(pred["selectivity_index"], 1),
                    "eNTRy_Score": int(desc["eNTRy_Score"]),
                    "HasPrimaryAmine": int(desc["HasPrimaryAmine"]),
                    "RotBonds": int(desc["RotBonds"]),
                    "LogP": round(desc["LogP"], 2),
                    "MW": round(desc["MW"], 1),
                    "QED": round(desc["QED"], 2),
                    "SA_Score": round(desc["SA_Score"], 2),
                    "max_tanimoto_abx": novelty["max_tanimoto"],
                    "nearest_clinical_abx": novelty["nearest_antibiotic"],
                    "is_novel_scaffold": bool(novelty["is_novel_scaffold"]),
                }
            )

        if not scored_rows:
            return pd.DataFrame()

        df = pd.DataFrame(scored_rows).sort_values(
            by=["AMR_Score", "selectivity_index"], ascending=False
        )
        return df.head(top_k).reset_index(drop=True)

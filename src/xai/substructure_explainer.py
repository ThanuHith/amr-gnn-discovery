"""Substructure Rationalization (XAI), RDKit 2D Attribution Maps, and eNTRy/PAINS Audit.

Provides:
1. Batched counterfactual atom-masking attribution for AMRGraphNet across both
   antibacterial potency (pharmacophore) and human cytotoxicity (toxicophore).
2. High-resolution RDKit Cairo 2D molecular heatmaps highlighting positive and
   negative atom/bond contributions.
3. Automated PAINS & Brenk structural alert screening plus Richter eNTRy rule audit.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Batch, Data

from src.data_pipeline.prepare_amr_dataset import compute_amr_descriptors, standardize_smiles
from src.models.amr_gnn import AMRGraphNet
from src.models.featurizer import DescriptorScaler, mol_to_graph_data


@lru_cache(maxsize=1)
def _get_filter_catalogs() -> Tuple[FilterCatalog.FilterCatalog, FilterCatalog.FilterCatalog]:
    pains_params = FilterCatalog.FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    brenk_params = FilterCatalog.FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(pains_params), FilterCatalog.FilterCatalog(brenk_params)


@torch.no_grad()
def compute_atom_attributions(
    model: AMRGraphNet,
    mol_or_smiles: str | Chem.Mol,
    scaler: Optional[DescriptorScaler] = None,
) -> Dict[str, object]:
    """Compute atom-level counterfactual attributions for antibacterial & toxicity heads.

    Uses a single batched forward pass masking each atom's feature vector to 0
    and measuring the signed change in predicted logit relative to the unmasked graph.
    """
    if isinstance(mol_or_smiles, str):
        std = standardize_smiles(mol_or_smiles)
        if std is None:
            raise ValueError("Invalid SMILES string provided for XAI attribution.")
        canon_smi, mol = std
    else:
        mol = mol_or_smiles
        canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    base_data = mol_to_graph_data(mol, scaler=scaler, standardize=False)
    if base_data is None:
        raise ValueError("Failed to featurize molecule for XAI attribution.")

    device = next(model.parameters()).device
    model.eval()

    base_batch = Batch.from_data_list([base_data]).to(device)
    base_out = model.forward(base_batch)
    base_abx = float(base_out["logit_abx"].item())
    base_tox = float(base_out["logit_cytotox"].item())
    base_gn = float(base_out["logit_gram_neg"].item())

    num_atoms = mol.GetNumAtoms()
    masked_list: List[Data] = []
    for idx in range(num_atoms):
        d = base_data.clone()
        d.x = d.x.clone()
        d.x[idx, :] = 0.0
        masked_list.append(d)

    masked_batch = Batch.from_data_list(masked_list).to(device)
    masked_out = model.forward(masked_batch)

    raw_abx = (base_abx - masked_out["logit_abx"].cpu().numpy()).astype(np.float32)
    raw_tox = (base_tox - masked_out["logit_cytotox"].cpu().numpy()).astype(np.float32)
    raw_gn = (base_gn - masked_out["logit_gram_neg"].cpu().numpy()).astype(np.float32)

    def _normalize(arr: np.ndarray) -> np.ndarray:
        max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        if max_abs < 1e-6:
            return np.zeros_like(arr)
        return np.clip(arr / max_abs, -1.0, 1.0)

    return {
        "smiles": canon_smi,
        "mol": mol,
        "abx_weights": _normalize(raw_abx),
        "tox_weights": _normalize(raw_tox),
        "gram_neg_weights": _normalize(raw_gn),
        "abx_raw": raw_abx,
        "tox_raw": raw_tox,
    }


def render_atom_attribution_image(
    mol: Chem.Mol,
    weights: np.ndarray,
    mode: str = "pharmacophore",
    size: Tuple[int, int] = (480, 340),
    legend: str = "",
) -> Image.Image:
    """Render a 2D RDKit molecular drawing with atom & bond attribution heatmaps.

    Args:
        mol: RDKit Mol object.
        weights: Signed normalized atom weights in [-1, 1].
        mode: 'pharmacophore' (emerald positive / amber negative) or
              'toxicophore' (crimson positive / sky-blue negative).
        size: (width, height) in pixels.
        legend: Optional caption rendered at bottom of image.
    """
    draw_mol = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(draw_mol)

    if mode == "pharmacophore":
        pos_rgb = (0.08, 0.72, 0.50)  # Emerald teal (antibacterial pharmacophore)
        neg_rgb = (0.92, 0.58, 0.15)  # Amber (activity-reducing)
    else:
        pos_rgb = (0.88, 0.20, 0.24)  # Crimson (human toxicophore)
        neg_rgb = (0.20, 0.58, 0.88)  # Sky blue (safety-conferring)

    atom_colors: Dict[int, Tuple[float, float, float, float]] = {}
    atom_radii: Dict[int, float] = {}
    highlight_atoms: List[int] = []

    for idx, w in enumerate(weights):
        val = float(w)
        if abs(val) < 0.10:
            continue
        highlight_atoms.append(idx)
        alpha = min(0.85, 0.22 + 0.63 * abs(val))
        rgb = pos_rgb if val > 0 else neg_rgb
        atom_colors[idx] = (rgb[0], rgb[1], rgb[2], alpha)
        atom_radii[idx] = 0.28 + 0.18 * abs(val)

    highlight_bonds: List[int] = []
    bond_colors: Dict[int, Tuple[float, float, float, float]] = {}
    for bond in draw_mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        w1 = float(weights[a1])
        w2 = float(weights[a2])
        if w1 * w2 > 0 and min(abs(w1), abs(w2)) >= 0.15:
            bidx = bond.GetIdx()
            highlight_bonds.append(bidx)
            avg_abs = 0.5 * (abs(w1) + abs(w2))
            rgb = pos_rgb if w1 > 0 else neg_rgb
            bond_colors[bidx] = (rgb[0], rgb[1], rgb[2], min(0.75, 0.20 + 0.55 * avg_abs))

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.padding = 0.08
    opts.legendFontSize = 14
    opts.addStereoAnnotation = True
    drawer.DrawMolecule(
        draw_mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
        highlightAtomRadii=atom_radii,
        legend=legend,
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def audit_structural_alerts_and_entry(smiles: str) -> Dict[str, object]:
    """Audit PAINS/Brenk structural alerts and Richter et al. Gram-negative eNTRy rules."""
    std = standardize_smiles(smiles)
    if std is None:
        raise ValueError("Invalid SMILES provided for structural audit.")
    canon_smi, mol = std

    pains_cat, brenk_cat = _get_filter_catalogs()
    pains_alerts = sorted({entry.GetDescription() for entry in pains_cat.GetMatches(mol)})
    brenk_alerts = sorted({entry.GetDescription() for entry in brenk_cat.GetMatches(mol)})

    desc = compute_amr_descriptors(mol)
    has_amine = bool(desc["HasPrimaryAmine"] >= 1.0)
    rigid_ok = bool(desc["RotBonds"] <= 5.0)
    amphiphilic_ok = bool(150.0 <= desc["MW"] <= 550.0 and -1.5 <= desc["LogP"] <= 3.5)

    checklist = [
        {
            "criterion": "Ionizable Primary Amine (OmpF/OmpC Electrostatic Anchor)",
            "measured": "Present" if has_amine else "Absent",
            "threshold": "Primary NH2 (non-amide)",
            "passed": has_amine,
        },
        {
            "criterion": "Molecular Rigidity (Low Entropic Penalty in Porin)",
            "measured": f"{int(desc['RotBonds'])} rotatable bonds",
            "threshold": "<= 5 rotatable bonds",
            "passed": rigid_ok,
        },
        {
            "criterion": "Amphiphilic & Size Window (Porin Constriction Fit)",
            "measured": f"MW={desc['MW']:.1f} Da, LogP={desc['LogP']:.2f}",
            "threshold": "150<=MW<=550, -1.5<=LogP<=3.5",
            "passed": amphiphilic_ok,
        },
    ]

    return {
        "smiles": canon_smi,
        "descriptors": desc,
        "pains_alerts": pains_alerts,
        "brenk_alerts": brenk_alerts,
        "alert_count": len(pains_alerts) + len(brenk_alerts),
        "entry_score": int(desc["eNTRy_Score"]),
        "entry_checklist": checklist,
    }

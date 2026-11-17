"""Molecular Graph Featurizer for PathoGraph-AMR.

Converts standardized SMILES or RDKit Mol objects into PyTorch Geometric `Data`
objects with:
- x: [num_atoms, 34] atom feature tensor (element, degree, formal charge,
  hybridization, H-count, aromaticity, ring membership, primary-amine marker)
- edge_index: [2, num_edges] bidirectional bond connectivity
- edge_attr: [num_edges, 10] bond feature tensor (bond order, stereo, conjugated, ring)
- u: [1, 12] global Richter eNTRy & physicochemical descriptor tensor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.data_pipeline.prepare_amr_dataset import (
    DESCRIPTOR_NAMES,
    compute_amr_descriptors,
    standardize_smiles,
)

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si"]
DEGREES = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

_PRIMARY_AMINE_SMARTS = Chem.MolFromSmarts("[NX3;H2;!$(NC=[O,S,N])]")

NODE_FEATURE_DIM = 34
EDGE_FEATURE_DIM = 10
GLOBAL_FEATURE_DIM = len(DESCRIPTOR_NAMES)


def _one_hot_with_unknown(value, choices: Sequence) -> List[float]:
    vec = [0.0] * (len(choices) + 1)
    if value in choices:
        vec[choices.index(value)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


def _atom_features(atom: Chem.Atom, is_primary_amine_atom: bool) -> List[float]:
    feats: List[float] = []
    # 12 dims: atom symbol + Other
    feats.extend(_one_hot_with_unknown(atom.GetSymbol(), ATOM_SYMBOLS))
    # 6 dims: degree 0..4 + 5+
    feats.extend(_one_hot_with_unknown(atom.GetTotalDegree(), DEGREES))
    # 4 dims: formal charge (-1, 0, +1, Other)
    fc = atom.GetFormalCharge()
    fc_vec = [0.0, 0.0, 0.0, 0.0]
    if fc == -1:
        fc_vec[0] = 1.0
    elif fc == 0:
        fc_vec[1] = 1.0
    elif fc == 1:
        fc_vec[2] = 1.0
    else:
        fc_vec[3] = 1.0
    feats.extend(fc_vec)
    # 5 dims: hybridization (SP, SP2, SP3, SP3D/SP3D2, Other)
    hyb = atom.GetHybridization()
    hyb_vec = [0.0, 0.0, 0.0, 0.0, 0.0]
    if hyb == Chem.rdchem.HybridizationType.SP:
        hyb_vec[0] = 1.0
    elif hyb == Chem.rdchem.HybridizationType.SP2:
        hyb_vec[1] = 1.0
    elif hyb == Chem.rdchem.HybridizationType.SP3:
        hyb_vec[2] = 1.0
    elif hyb in (Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2):
        hyb_vec[3] = 1.0
    else:
        hyb_vec[4] = 1.0
    feats.extend(hyb_vec)
    # 4 dims: total Hs (0, 1, 2, 3+)
    total_h = atom.GetTotalNumHs()
    feats.extend(_one_hot_with_unknown(min(total_h, 3), [0, 1, 2]))
    # 3 dims: aromatic, in_ring, primary_amine_N
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats.append(1.0 if is_primary_amine_atom else 0.0)
    assert len(feats) == NODE_FEATURE_DIM
    return feats


def _bond_features(bond: Chem.Bond) -> List[float]:
    feats: List[float] = []
    # 5 dims: bond type + Other
    feats.extend(_one_hot_with_unknown(bond.GetBondType(), BOND_TYPES))
    # 3 dims: stereo (None, Z/E/Cis/Trans, Other)
    st = bond.GetStereo()
    st_vec = [0.0, 0.0, 0.0]
    if st == Chem.rdchem.BondStereo.STEREONONE:
        st_vec[0] = 1.0
    elif st in (
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ):
        st_vec[1] = 1.0
    else:
        st_vec[2] = 1.0
    feats.extend(st_vec)
    # 2 dims: conjugated, in_ring
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    assert len(feats) == EDGE_FEATURE_DIM
    return feats


@dataclass
class DescriptorScaler:
    """Train-set z-score scaler for the 12 global physicochemical eNTRy features."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "DescriptorScaler":
        mean = matrix.mean(axis=0).astype(np.float32)
        std = matrix.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, vec: np.ndarray) -> np.ndarray:
        return ((vec - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, List[float]]) -> "DescriptorScaler":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


def mol_to_graph_data(
    mol_or_smiles: Union[str, Chem.Mol],
    scaler: Optional[DescriptorScaler] = None,
    labels: Optional[Dict[str, float]] = None,
    standardize: bool = True,
) -> Optional[Data]:
    """Convert a SMILES or Mol into a PyTorch Geometric `Data` graph."""
    if isinstance(mol_or_smiles, str):
        if standardize:
            std = standardize_smiles(mol_or_smiles)
            if std is None:
                return None
            canon_smi, mol = std
        else:
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                return None
            canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    else:
        mol = mol_or_smiles
        canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    primary_amine_atoms = set()
    if _PRIMARY_AMINE_SMARTS is not None:
        for match in mol.GetSubstructMatches(_PRIMARY_AMINE_SMARTS):
            primary_amine_atoms.update(match)

    node_feats = [
        _atom_features(atom, atom.GetIdx() in primary_amine_atoms)
        for atom in mol.GetAtoms()
    ]
    x = torch.tensor(node_feats, dtype=torch.float32)

    edge_indices: List[List[int]] = []
    edge_attrs: List[List[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    desc = compute_amr_descriptors(mol)
    raw_u = np.array([desc[name] for name in DESCRIPTOR_NAMES], dtype=np.float32)
    norm_u = scaler.transform(raw_u) if scaler is not None else raw_u
    u = torch.tensor(norm_u, dtype=torch.float32).unsqueeze(0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, smiles=canon_smi)

    if labels is not None:
        data.y_abx = torch.tensor([float(labels["active_abx"])], dtype=torch.float32)
        data.y_pmic = torch.tensor([float(labels["pMIC"])], dtype=torch.float32)
        data.y_gn = torch.tensor([float(labels["gram_neg_accum"])], dtype=torch.float32)
        data.y_tox = torch.tensor([float(labels["cytotoxic_human"])], dtype=torch.float32)
        data.y_pcc50 = torch.tensor([float(labels["pCC50"])], dtype=torch.float32)

    return data

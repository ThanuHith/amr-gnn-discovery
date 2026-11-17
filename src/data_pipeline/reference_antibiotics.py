"""Reference panel of 35 FDA/EMA-approved clinical antibiotics and discovery benchmarks.

Used for:
1. Structural novelty scoring via ECFP4 (Morgan radius=2, 2048-bit) Tanimoto similarity
   against known clinical antibiotics across 10 mechanism classes (Stokes et al., Cell 2020).
2. Benchmarking classic vs. novel antibiotic scaffolds (Halicin, Abaucin) in XAI and UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


@dataclass(frozen=True)
class ReferenceAntibiotic:
    """Metadata and structure for a clinical or benchmark antibiotic."""

    name: str
    smiles: str
    drug_class: str
    mechanism: str
    spectrum: str


CLINICAL_ANTIBIOTICS: List[ReferenceAntibiotic] = [
    # 1. Beta-Lactams (6)
    ReferenceAntibiotic(
        "Penicillin G",
        "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
        "Beta-Lactam",
        "PBP / Cell wall transpeptidase inhibitor",
        "Gram-positive",
    ),
    ReferenceAntibiotic(
        "Amoxicillin",
        "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C",
        "Beta-Lactam",
        "PBP / Cell wall transpeptidase inhibitor",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Ampicillin",
        "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C",
        "Beta-Lactam",
        "PBP / Cell wall transpeptidase inhibitor",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Cefotaxime",
        "O=C2N1/C(=C(\\CS[C@@H]1[C@@H]2NC(=O)C(=N\\OC)\\c3nc(sc3)N)COC(=O)C)C(=O)O",
        "Beta-Lactam",
        "3rd-gen Cephalosporin PBP inhibitor",
        "Broad-spectrum (Gram-negative)",
    ),
    ReferenceAntibiotic(
        "Ceftriaxone",
        "CN1C(=NC(=O)C(=O)N1)SCC2=C(N3[C@@H]([C@@H](C3=O)NC(=O)/C(=N\\OC)/C4=CSC(=N4)N)SC2)C(=O)O",
        "Beta-Lactam",
        "3rd-gen Cephalosporin PBP inhibitor",
        "Broad-spectrum (Gram-negative)",
    ),
    ReferenceAntibiotic(
        "Meropenem",
        "C[C@@H](O)[C@@H]1[C@H]2[C@@H](C)C(=C(N2C1=O)C(O)=O)S[C@@H]3CN[C@@H](C3)C(=O)N(C)C",
        "Beta-Lactam",
        "Carbapenem PBP inhibitor",
        "Broad-spectrum (ESKAPE)",
    ),
    ReferenceAntibiotic(
        "Imipenem",
        "CC(C1C2CC(=C(N2C1=O)C(=O)O)SCCN=CN)O",
        "Beta-Lactam",
        "Carbapenem PBP inhibitor",
        "Broad-spectrum (ESKAPE)",
    ),
    ReferenceAntibiotic(
        "Aztreonam",
        "C[C@H]1[C@@H](C(=O)N1S(=O)(=O)O)NC(=O)/C(=N\\OC(C)(C)C(=O)O)/C2=CSC(=N2)N",
        "Beta-Lactam",
        "Monobactam PBP3 inhibitor",
        "Gram-negative",
    ),
    # 2. Fluoroquinolones (4)
    ReferenceAntibiotic(
        "Ciprofloxacin",
        "C1CC1N2C=C(C(=O)C3=C2C=C(C(=C3)F)N4CCNCC4)C(=O)O",
        "Fluoroquinolone",
        "DNA gyrase & Topoisomerase IV inhibitor",
        "Broad-spectrum (Gram-negative)",
    ),
    ReferenceAntibiotic(
        "Levofloxacin",
        "C[C@H]1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O",
        "Fluoroquinolone",
        "DNA gyrase & Topoisomerase IV inhibitor",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Moxifloxacin",
        "COC1=C2C(=CC(=C1N3C[C@@H]4CCCN[C@@H]4C3)F)C(=O)C(=CN2C5CC5)C(=O)O",
        "Fluoroquinolone",
        "DNA gyrase & Topoisomerase IV inhibitor",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Norfloxacin",
        "CCN1C=C(C(=O)C2=CC(=C(C=C21)N3CCNCC3)F)C(=O)O",
        "Fluoroquinolone",
        "DNA gyrase inhibitor",
        "Gram-negative",
    ),
    # 3. Macrolides (3)
    ReferenceAntibiotic(
        "Erythromycin",
        "CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O",
        "Macrolide",
        "50S ribosomal subunit translocation inhibitor",
        "Gram-positive / Atypical",
    ),
    ReferenceAntibiotic(
        "Azithromycin",
        "CC[C@@H]1[C@@]([C@@H]([C@H](N(C[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O",
        "Macrolide",
        "50S ribosomal subunit translocation inhibitor",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Clarithromycin",
        "CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)OC)C)C)O)(C)O",
        "Macrolide",
        "50S ribosomal subunit translocation inhibitor",
        "Gram-positive / Atypical",
    ),
    # 4. Tetracyclines (4)
    ReferenceAntibiotic(
        "Tetracycline",
        "CN(C)C1C2CC3CC4=C(C=CC=C4O)C(=O)C3=C(C2(C(=O)C(=C1O)C(=O)N)O)O",
        "Tetracycline",
        "30S ribosomal A-site aminoacyl-tRNA blocker",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Doxycycline",
        "C1(=O)[C@]2(O)[C@@]([H])([C@@H](O)[C@@]3([H])C(=C2O)C(=O)C2=C(C=CC=C2O)[C@@H]3C)[C@H](N(C)C)C(O)=C1C(N)=O",
        "Tetracycline",
        "30S ribosomal A-site aminoacyl-tRNA blocker",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Minocycline",
        "CN(C)C1C2CC3CC4=C(C=CC(=C4C(=C3C(=O)C2(C(=C(C1=O)C(=O)N)O)O)O)O)N(C)C",
        "Tetracycline",
        "30S ribosomal A-site aminoacyl-tRNA blocker",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Tigecycline",
        "CC(C)(C)NCC(=O)NC1=CC(=C2CC3CC4C(C(=O)C(=C(C4(C(=O)C3=C(C2=C1O)O)O)O)C(=O)N)N(C)C)N(C)C",
        "Tetracycline",
        "Glycylcycline 30S ribosomal inhibitor",
        "Broad-spectrum (MDR ESKAPE)",
    ),
    # 5. Aminoglycosides (4)
    ReferenceAntibiotic(
        "Gentamicin",
        "CN[C@@H]1[C@@H](O)[C@@H](O)[C@@H](CO)O[C@@H]1O[C@@H]2[C@H](N)[C@@H](O)[C@H](N)[C@@H](O)[C@H]2O[C@H]3CC[C@@H](C(N)=C3)NC",
        "Aminoglycoside",
        "30S ribosomal decoding error inducer",
        "Gram-negative",
    ),
    ReferenceAntibiotic(
        "Kanamycin",
        "C1C(C(C(C(C1N)OC2C(C(C(C(O2)CN)O)O)O)O)OC3C(C(C(C(O3)CO)O)N)O)N",
        "Aminoglycoside",
        "30S ribosomal decoding error inducer",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Amikacin",
        "NCC[C@H](O)C(=O)N[C@@H]1C[C@H](N)[C@@H](O[C@H]2O[C@H](CN)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H]1O[C@H]3O[C@H](CO)[C@@H](O)[C@H](N)[C@H]3O",
        "Aminoglycoside",
        "30S ribosomal decoding error inducer",
        "Gram-negative (MDR)",
    ),
    ReferenceAntibiotic(
        "Streptomycin",
        "CC1C(C(C(O1)OC2C(C(C(C(C2O)O)N=C(N)N)O)N=C(N)N)OC3C(C(C(C(O3)CO)O)O)NC)(C=O)O",
        "Aminoglycoside",
        "30S ribosomal initiation inhibitor",
        "Broad-spectrum / Mycobacteria",
    ),
    # 6. Oxazolidinones (2)
    ReferenceAntibiotic(
        "Linezolid",
        "CC(=O)NC[C@H]1CN(C(=O)O1)C2=CC(=C(C=C2)N3CCOCC3)F",
        "Oxazolidinone",
        "50S 23S rRNA P-site inhibitor",
        "Gram-positive (MRSA/VRE)",
    ),
    ReferenceAntibiotic(
        "Tedizolid",
        "CN1N=NC(=N1)C1=CC=C(C=N1)C1=CC=C(C=C1F)N1C[C@@H](CO)OC1=O",
        "Oxazolidinone",
        "50S 23S rRNA P-site inhibitor",
        "Gram-positive (MRSA/VRE)",
    ),
    # 7. Sulfonamides & DHFR Inhibitors (4)
    ReferenceAntibiotic(
        "Sulfamethoxazole",
        "CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N",
        "Sulfonamide",
        "Dihydropteroate synthase (DHPS) inhibitor",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Sulfadiazine",
        "NC1=CC=C(C=C1)S(=O)(=O)NC2=NC=CC=N2",
        "Sulfonamide",
        "Dihydropteroate synthase (DHPS) inhibitor",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Trimethoprim",
        "COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N",
        "Diaminopyrimidine",
        "Bacterial Dihydrofolate reductase (DHFR) inhibitor",
        "Broad-spectrum",
    ),
    ReferenceAntibiotic(
        "Dapsone",
        "NC1=CC=C(C=C1)S(=O)(=O)C2=CC=C(C=C2)N",
        "Sulfone",
        "Folate pathway DHPS inhibitor",
        "Mycobacteria / Gram-positive",
    ),
    # 8. Glycopeptides & Lipopeptides (2)
    ReferenceAntibiotic(
        "Vancomycin",
        "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=C4C=C5C=C3OC6=C(C=C(C=C6)C(C(C(=O)NC(C(=O)NC5C(=O)NC7C8=CC(=C(C=C8)O)C9=C(C=C(C=C9O)O)C(NC(=O)C(C(C1=CC(=C(O4)C=C1)Cl)O)NC7=O)C(=O)O)CC(=O)N)NC(=O)C(CC(C)C)NC)O)Cl)CO)O)O)(C)N)O",
        "Glycopeptide",
        "D-Ala-D-Ala peptidoglycan cross-linking blocker",
        "Gram-positive (MRSA)",
    ),
    ReferenceAntibiotic(
        "Daptomycin",
        "CCCCCCCCCC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@H](CC(N)=O)C(=O)N[C@@H](CC(O)=O)C(=O)N[C@H]3[C@@H](C)OC(=O)[C@H](CC(=O)c4ccccc4N)NC(=O)[C@@H](NC(=O)[C@@H](CO)NC(=O)CNC(=O)[C@H](CC(O)=O)NC(=O)[C@@H](C)NC(=O)[C@H](CC(O)=O)NC(=O)[C@@H](CCCN)NC(=O)CNC3=O)[C@H](C)CC(O)=O",
        "Lipopeptide",
        "Ca2+-dependent membrane depolarization",
        "Gram-positive (MRSA/VRE)",
    ),
    # 9. Rifamycins (1)
    ReferenceAntibiotic(
        "Rifampicin",
        "C[C@H]1/C=C/C=C(/C)\\C(=O)NC2=C(C3=C(C(=C4C(=C3C(=C2/C=N/N5CCN(C)CC5)O)C(=O)[C@@](C)(O/C=C/[C@@H]([C@@H](C)[C@H]([C@H](C)[C@@H]([C@H](C)[C@H]1O)O)OC(=O)C)OC)O4)C)O)O",
        "Rifamycin",
        "Bacterial RNA polymerase beta-subunit inhibitor",
        "Broad-spectrum / Mycobacteria",
    ),
    # 10. Polymyxins, Nitroheterocycles & Phenicols (4)
    ReferenceAntibiotic(
        "Colistin A",
        "CC[C@@H](C)CCCCC(=O)N[C@@H](CCN)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H](CCN)C(=O)N[C@H]1CCNC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](NC(=O)[C@@H](NC1=O)CCN)CC(C)C)CC(C)C)CCN)CCN)[C@@H](C)O",
        "Polymyxin",
        "LPS outer membrane disruption",
        "Gram-negative (MDR last-resort)",
    ),
    ReferenceAntibiotic(
        "Nitrofurantoin",
        "C1C(=O)NC(=O)N1/N=C/C2=CC=C(O2)[N+](=O)[O-]",
        "Nitrofuran",
        "Nitroreductase-activated DNA/ribosome damage",
        "Broad-spectrum (Urinary)",
    ),
    ReferenceAntibiotic(
        "Metronidazole",
        "CC1=NC=C(N1CCO)[N+](=O)[O-]",
        "Nitroimidazole",
        "Anaerobic electron-sink DNA strand breakage",
        "Anaerobes",
    ),
    ReferenceAntibiotic(
        "Chloramphenicol",
        "C1=CC(=CC=C1[C@H]([C@@H](CO)NC(=O)C(Cl)Cl)O)[N+](=O)[O-]",
        "Phenicol",
        "50S peptidyl transferase inhibitor",
        "Broad-spectrum",
    ),
]


# Preset showcase compounds for Streamlit profiler & XAI validation
BENCHMARK_DISCOVERY_MOLECULES: Dict[str, Dict[str, str]] = {
    "Halicin (SU3327) — Novel AI-Discovered Antibiotic": {
        "smiles": "NC1=NN=C(S1)SC2=NC=C(S2)[N+](=O)[O-]",
        "category": "Novel AI Discovery (Stokes et al., Cell 2020)",
        "notes": "Proton motive force (PMF) disruptor active against MDR A. baumannii & E. coli; structurally distinct from all 10 clinical classes.",
    },
    "Abaucin (RS102895) — Narrow-Spectrum A. baumannii Hit": {
        "smiles": "C1=CC2=C(C=C1)NC(=O)OC32CCN(CCC4=CC=C(C=C4)C(F)(F)F)CC3",
        "category": "Novel AI Discovery (Liu et al., Nat Chem Biol 2023)",
        "notes": "Lipoprotein trafficking (LolE) inhibitor selective against CRAB Acinetobacter baumannii.",
    },
    "Ciprofloxacin — Clinical Fluoroquinolone": {
        "smiles": "C1CC1N2C=C(C(=O)C3=C2C=C(C(=C3)F)N4CCNCC4)C(=O)O",
        "category": "Clinical Standard (Fluoroquinolone)",
        "notes": "Broad-spectrum Gram-negative DNA gyrase inhibitor with strong outer-membrane accumulation.",
    },
    "6-Amino-2-Methyl-Quinoline (eNTRy-Compliant Lead)": {
        "smiles": "Cc1ccc2ccc(N)cc2n1",
        "category": "eNTRy-Compliant Primary Amine Scaffold",
        "notes": "Rigid planar bicyclic core bearing an unhindered primary amine for OmpF/OmpC porin permeation.",
    },
    "Trimethoprim — Folate Pathway Inhibitor": {
        "smiles": "COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N",
        "category": "Clinical Standard (Diaminopyrimidine)",
        "notes": "Bacterial DHFR inhibitor with dual aryl-amine motif and low human cytotoxicity.",
    },
    "Linezolid — MRSA Oxazolidinone": {
        "smiles": "CC(=O)NC[C@H]1CN(C(=O)O1)C2=CC(=C(C=C2)N3CCOCC3)F",
        "category": "Clinical Standard (Oxazolidinone)",
        "notes": "Gram-positive MRSA protein synthesis inhibitor lacking primary amine for Gram-negative porin entry.",
    },
    "Nitrofurantoin — Clinical Nitrofuran": {
        "smiles": "C1C(=O)NC(=O)N1/N=C/C2=CC=C(O2)[N+](=O)[O-]",
        "category": "Clinical Standard (Nitrofuran)",
        "notes": "Prodrug activated by bacterial nitroreductases.",
    },
    "Doxorubicin-like Cytotoxic Control (Anthracycline Core)": {
        "smiles": "COc1cccc2C(=O)c3c(O)c4C[C@](O)(C(=O)CO)C[C@H](O)c4c(O)c3C(=O)c12",
        "category": "Human Cytotoxicity Positive Control",
        "notes": "Intercalating quinone scaffold exhibiting high mammalian HepG2 cytotoxicity and poor selectivity.",
    },
    "Ibuprofen — Non-Antibiotic Negative Control": {
        "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "category": "Inactive Drug-Like Decoy",
        "notes": "NSAID negative control lacking antibacterial pharmacophores.",
    },
}


_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@lru_cache(maxsize=1)
def _get_reference_fingerprints() -> List[Tuple[ReferenceAntibiotic, DataStructs.ExplicitBitVect]]:
    """Precompute and cache ECFP4 fingerprints for all reference clinical antibiotics."""
    fps = []
    for abx in CLINICAL_ANTIBIOTICS:
        mol = Chem.MolFromSmiles(abx.smiles)
        if mol is not None:
            fps.append((abx, _MORGAN_GEN.GetFingerprint(mol)))
    return fps


def compute_antibiotic_novelty(
    mol_or_smiles: Union[str, Chem.Mol],
    novelty_threshold: float = 0.40,
) -> Dict[str, object]:
    """Compute maximum ECFP4 Tanimoto similarity against the 36 clinical antibiotics.

    Returns:
        Dict with keys:
        - max_tanimoto (float): Highest Tanimoto similarity in [0, 1].
        - nearest_antibiotic (str): Closest FDA/EMA reference antibiotic name.
        - nearest_class (str): Drug class of the nearest clinical antibiotic.
        - is_novel_scaffold (bool): True when max_tanimoto < novelty_threshold.
    """
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles

    if mol is None:
        return {
            "max_tanimoto": 1.0,
            "nearest_antibiotic": "Invalid SMILES",
            "nearest_class": "Unknown",
            "is_novel_scaffold": False,
        }

    query_fp = _MORGAN_GEN.GetFingerprint(mol)
    ref_fps = _get_reference_fingerprints()
    ref_bvs = [fp for _, fp in ref_fps]
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, ref_bvs)

    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    best_sim = float(sims[best_idx])
    best_abx = ref_fps[best_idx][0]

    return {
        "max_tanimoto": round(best_sim, 4),
        "nearest_antibiotic": best_abx.name,
        "nearest_class": best_abx.drug_class,
        "is_novel_scaffold": bool(best_sim < novelty_threshold),
    }


def get_reference_dataframe_records() -> List[Dict[str, str]]:
    """Return reference antibiotic records as plain dictionaries."""
    return [
        {
            "name": abx.name,
            "smiles": abx.smiles,
            "drug_class": abx.drug_class,
            "mechanism": abx.mechanism,
            "spectrum": abx.spectrum,
        }
        for abx in CLINICAL_ANTIBIOTICS
    ]

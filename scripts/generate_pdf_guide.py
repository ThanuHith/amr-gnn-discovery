"""Generate the comprehensive PathoGraph-AMR Master Technical & User Guide PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
DOCS_DIR = ROOT_DIR / "docs"


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    w, h = A4
    # Top accent bar
    canvas.setStrokeColor(colors.HexColor("#0f766e"))
    canvas.setLineWidth(1.2)
    canvas.line(1.8 * cm, h - 1.4 * cm, w - 1.8 * cm, h - 1.4 * cm)

    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.setFillColor(colors.HexColor("#0f766e"))
    canvas.drawString(1.8 * cm, h - 1.2 * cm, "PathoGraph-AMR | Complete Technical, Algorithmic & User Master Guide")

    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawRightString(w - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.drawString(1.8 * cm, 1.1 * cm, "GitHub Portfolio Documentation — Computational Drug Discovery & AMR")

    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.6)
    canvas.line(1.8 * cm, 1.5 * cm, w - 1.8 * cm, 1.5 * cm)
    canvas.restoreState()


def _make_table(data, col_widths, header_bg="#0f766e") -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8.0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 1), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ]
        )
    )
    return t


def build_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="PathoGraph-AMR Complete Technical & User Guide",
        author="PathoGraph-AMR Documentation",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f766e"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=12,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14.5,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=9,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "BulletItem",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=4,
    )
    box_style = ParagraphStyle(
        "CalloutBox",
        parent=body_style,
        backColor=colors.HexColor("#ecfdf5"),
        borderColor=colors.HexColor("#0f766e"),
        borderWidth=0.7,
        borderPadding=7,
        spaceBefore=6,
        spaceAfter=8,
    )
    code_style = ParagraphStyle(
        "CodeBlock",
        fontName="Courier",
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8,
    )
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor("#0f172a"),
    )
    cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )
    cell_header = ParagraphStyle(
        "TableCellHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.2,
        leading=11,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=10,
    )

    def P(text: str, style=body_style) -> Paragraph:
        return Paragraph(text, style)

    def B(text: str) -> Paragraph:
        return Paragraph(f"• {text}", bullet_style)

    story = []

    # ==================== COVER / SECTION 1 ====================
    story.append(P("PathoGraph-AMR: Complete Technical, Algorithmic & User Guide", title_style))
    story.append(
        P(
            "Multi-Task Edge-Aware Graph Attention (GATv2) & Graph Isomorphism Network (GIN) for Structurally Novel "
            "Antibiotic Discovery, Gram-Negative Outer-Membrane Permeation (eNTRy Rules), Explainable AI, and De Novo Design",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f766e"), spaceAfter=10))

    story.append(
        P(
            "<b>Repository:</b> https://github.com/ThanuHith/amr-gnn-discovery &nbsp;|&nbsp; "
            "<b>Local Workspace:</b> <font face='Courier'>C:\\Users\\GOD\\.gemini\\antigravity\\scratch\\amr-gnn-discovery</font><br/>"
            "<b>Core Stack:</b> PyTorch, PyTorch Geometric (GATv2Conv + GINConv), RDKit Cheminformatics, XGBoost, "
            "Scikit-Learn, SELFIES, and Streamlit.",
            box_style,
        )
    )

    story.append(P("1. The Real-World Industry Problem & Why This Project Matters", h1_style))
    story.append(
        P(
            "Antimicrobial Resistance (AMR) is one of the most urgent crises in modern medicine and pharmaceutical R&amp;D. "
            "Hospital-acquired superbugs known as the <b>ESKAPE pathogens</b> (<i>Enterococcus faecium, Staphylococcus aureus, "
            "Klebsiella pneumoniae, Acinetobacter baumannii, Pseudomonas aeruginosa,</i> and <i>Enterobacter</i> species) have "
            "evolved resistance to nearly all traditional antibiotics. Yet big pharmaceutical companies struggle to discover "
            "new antibiotics because over 95% of early chemical hits fail due to <b>three fundamental bottlenecks</b>:"
        )
    )
    story.append(
        B(
            "<b>Bottleneck 1 — Structural Redundancy &amp; Cross-Resistance:</b> Most screening pipelines rediscover slight "
            "variations of existing drug classes (like beta-lactams, fluoroquinolones, or macrolides). Bacteria already carry "
            "resistance enzymes (such as NDM-1 beta-lactamases or efflux pumps) that immediately destroy derivatives of old "
            "scaffolds. Inspired by <b>Stokes et al. (<i>Cell</i>, 2020 — discovery of Halicin)</b> and <b>Wong et al. "
            "(<i>Nature</i>, 2023)</b>, PathoGraph-AMR enforces a quantitative <b>Structural Novelty Filter (T<sub>max</sub> &lt; 0.40)</b> "
            "against 36 FDA/EMA-approved clinical antibiotics across 10 mechanism classes."
        )
    )
    story.append(
        B(
            "<b>Bottleneck 2 — Gram-Negative Outer-Membrane Impermeability:</b> Gram-negative superbugs (<i>E. coli, "
            "A. baumannii, P. aeruginosa</i>) are protected by a double membrane: an outer lipopolysaccharide (LPS) barrier "
            "with narrow water-filled protein channels called <b>OmpF/OmpC porins</b>, plus efflux pumps that eject foreign "
            "molecules. Most potent enzyme inhibitors cannot cross this barrier. Based on <b>Richter et al. (<i>Nature</i>, 2017)</b>, "
            "PathoGraph-AMR explicitly encodes the <b>eNTRy Rules</b> for Gram-negative accumulation: (1) an unhindered "
            "ionizable <b>primary amine (-NH<sub>2</sub>)</b>, (2) high molecular rigidity (<b>Rotatable Bonds &le; 5</b>), and "
            "(3) balanced amphiphilicity and size (<b>150 &le; MW &le; 550 Da</b>, <b>-1.5 &le; LogP &le; 3.5</b>)."
        )
    )
    story.append(
        B(
            "<b>Bottleneck 3 — Mammalian Cytotoxicity (Safety Attrition):</b> Many compounds that kill bacteria act like "
            "chemical detergents or DNA intercalators that also kill human liver and kidney cells (HepG2/HEK293). "
            "PathoGraph-AMR jointly predicts bacterial potency (<b>pMIC</b>) and human cytotoxicity (<b>pCC<sub>50</sub></b>) "
            "to compute the clinical <b>Selectivity Index (SI = 10<sup>pMIC - pCC50</sup>)</b>, prioritizing molecules that "
            "kill bacteria at low micromolar doses while leaving human cells unharmed (SI &ge; 10)."
        )
    )

    story.append(P("How This Impacts the Pharmaceutical &amp; Biotech Industry", h2_style))
    story.append(
        P(
            "In industrial drug discovery (at companies such as <b>Bayer, Boehringer Ingelheim, Merck KGaA, Evotec, BioNTech, "
            "and Roche</b>, or German research centers like <b>HIPS Saarbrücken, DZIF, and Helmholtz Munich</b>), synthesizing "
            "and wet-lab testing 10,000 compounds for MIC and HepG2 toxicity costs millions of euros and takes months. "
            "PathoGraph-AMR acts as an <b>in-silico early attrition &amp; lead-optimization engine</b>:"
        )
    )
    story.append(B("Screens thousands of virtual molecules in seconds before any wet-lab synthesis occurs."))
    story.append(B("Eliminates toxicophores (crimson XAI heatmaps) and non-permeable scaffolds early."))
    story.append(
        B(
            "Redesigns weak or Gram-positive-only hits by automatically installing primary amines and bioisosteres "
            "to unlock Gram-negative outer-membrane entry while preserving scaffold novelty."
        )
    )

    # ==================== SECTION 2: END-TO-END FLOW & EXAMPLES ====================
    story.append(PageBreak())
    story.append(P("2. End-to-End Workflow: How Input Becomes Output (With Examples)", h1_style))
    story.append(
        P(
            "To understand how PathoGraph-AMR works, let us trace exactly what happens when you give the system a molecule "
            "and what every output means."
        )
    )

    story.append(P("Step A: How to Provide Input (What is a SMILES String?)", h2_style))
    story.append(
        P(
            "The system accepts any molecule written as a <b>SMILES</b> (<i>Simplified Molecular-Input Line-Entry System</i>) "
            "string. SMILES is the universal text language used by chemists and databases (PubChem, ChEMBL, DrugBank) to "
            "write 2D/3D chemical structures as a single line of text:"
        )
    )
    story.append(
        B(
            "<b>Atoms</b> are written by their chemical symbols: <font face='Courier'>C</font> (carbon), "
            "<font face='Courier'>N</font> (nitrogen), <font face='Courier'>O</font> (oxygen), "
            "<font face='Courier'>S</font> (sulfur), <font face='Courier'>F</font> (fluorine), "
            "<font face='Courier'>Cl</font> (chlorine). Lowercase letters (<font face='Courier'>c, n, s, o</font>) "
            "represent atoms inside aromatic rings."
        )
    )
    story.append(
        B(
            "<b>Bonds &amp; Branches:</b> <font face='Courier'>=</font> is a double bond, <font face='Courier'>#</font> "
            "is a triple bond, parentheses <font face='Courier'>(...)</font> indicate a side-chain branch, and numbers "
            "(<font face='Courier'>1, 2</font>) mark where a ring closes."
        )
    )
    story.append(
        Preformatted(
            "Example 1 (Halicin / SU3327):   NC1=NN=C(S1)SC2=NC=C(S2)[N+](=O)[O-]\n"
            "Example 2 (Ciprofloxacin):      C1CC1N2C=C(C(=O)C3=C2C=C(C(=C3)F)N4CCNCC4)C(=O)O\n"
            "Example 3 (Ibuprofen - Decoy):  CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            code_style,
        )
    )

    story.append(P("Step B: Internal Processing Pipeline (5 Automated Stages)", h2_style))
    story.append(
        B(
            "<b>Stage 1 — Pharmaceutical Standardization (<font face='Courier'>standardize_smiles</font>):</b> "
            "Raw molecules from databases often contain counter-ions (like <font face='Courier'>.Na+</font> or "
            "<font face='Courier'>.Cl-</font>) or inconsistent charge states. RDKit's <font face='Courier'>LargestFragmentChooser</font> "
            "strips salts, <font face='Courier'>Uncharger</font> neutralizes protonation states, and the structure is converted "
            "to a unique canonical SMILES."
        )
    )
    story.append(
        B(
            "<b>Stage 2 — Molecular Graph Construction (<font face='Courier'>mol_to_graph_data</font>):</b> "
            "The molecule is converted into a mathematical graph <i>G = (V, E, u)</i> where every atom is a node "
            "(<b>34 features</b>), every chemical bond is a directed edge in both directions (<b>10 features</b>), and the "
            "whole molecule has <b>12 global physicochemical descriptors</b>."
        )
    )
    story.append(
        B(
            "<b>Stage 3 — Multi-Task Graph Neural Network (<font face='Courier'>AMRGraphNet</font>) + MC Dropout:</b> "
            "The graph passes through 2 edge-aware Graph Attention layers (<font face='Courier'>GATv2Conv</font>) and "
            "1 Graph Isomorphism layer (<font face='Courier'>GINConv</font>), fused with the 12 global descriptors. "
            "The model runs <b>T = 20 stochastic Monte Carlo Dropout passes</b> to output both the mean prediction "
            "and the epistemic uncertainty (&plusmn; &sigma;)."
        )
    )
    story.append(
        B(
            "<b>Stage 4 — Structural Novelty &amp; Medicinal Chemistry Audit:</b> "
            "The molecule's 2048-bit ECFP4 fingerprint is compared via Tanimoto similarity against all 36 FDA/EMA clinical "
            "antibiotics to compute <b>T<sub>max</sub></b>, while RDKit scans for PAINS/Brenk toxic alerts and checks "
            "Richter's 3 Gram-negative eNTRy rules."
        )
    )
    story.append(
        B(
            "<b>Stage 5 — Counterfactual Atom Attribution (XAI) &amp; De Novo Optimization:</b> "
            "Each atom is masked one-by-one in a batched pass to generate 2D emerald (pharmacophore) and crimson "
            "(toxicophore) heatmaps, and <font face='Courier'>eNTRyAntibioticDesigner</font> proposes optimized structural analogs."
        )
    )

    ex_data = [
        [
            P("<b>Output Metric</b>", cell_header),
            P("<b>Halicin (SU3327)</b><br/>Novel AI Antibiotic", cell_header),
            P("<b>Ciprofloxacin</b><br/>Clinical Fluoroquinolone", cell_header),
            P("<b>Doxorubicin-like</b><br/>Cytotoxic Control", cell_header),
            P("<b>Ibuprofen</b><br/>Inactive NSAID Control", cell_header),
        ],
        [
            P("<b>Antibacterial Prob. P(Abx)</b>", cell_bold),
            P("99.4% (&plusmn; 0.006 &sigma;)<br/><b>[Active Hit]</b>", cell_style),
            P("99.0% (&plusmn; 0.008 &sigma;)<br/><b>[Active Hit]</b>", cell_style),
            P("1.8%<br/>[Inactive as Abx]", cell_style),
            P("0.4%<br/>[Inactive Decoy]", cell_style),
        ],
        [
            P("<b>Predicted pMIC &amp; MIC</b>", cell_bold),
            P("pMIC = 5.50<br/>(MIC &approx; 3.15 &mu;M)", cell_style),
            P("pMIC = 5.71<br/>(MIC &approx; 1.95 &mu;M)", cell_style),
            P("pMIC = 3.92<br/>(MIC &approx; 120 &mu;M)", cell_style),
            P("pMIC = 3.65<br/>(MIC &approx; 224 &mu;M)", cell_style),
        ],
        [
            P("<b>Gram-Neg Entry Score</b>", cell_bold),
            P("eNTRy = 3 / 3<br/>P(Entry) = 100.0%", cell_style),
            P("eNTRy = 2 / 3<br/>P(Entry) = 98.5%", cell_style),
            P("eNTRy = 1 / 3<br/>P(Entry) = 4.2%", cell_style),
            P("eNTRy = 2 / 3<br/>P(Entry) = 6.1%", cell_style),
        ],
        [
            P("<b>Human Cytotoxicity P(Tox)</b>", cell_bold),
            P("2.1% (pCC50 = 4.13)<br/><b>[Safe]</b>", cell_style),
            P("1.2% (pCC50 = 4.04)<br/><b>[Safe]</b>", cell_style),
            P("96.8% (pCC50 = 6.45)<br/><b>[Highly Toxic]</b>", cell_style),
            P("3.5% (pCC50 = 3.95)<br/>[Safe]", cell_style),
        ],
        [
            P("<b>Selectivity Index (SI)</b>", cell_bold),
            P("<b>23.3x</b> (Favorable &ge;10)", cell_style),
            P("<b>47.1x</b> (Favorable &ge;10)", cell_style),
            P("<b>0.003x</b> (Lethal to host)", cell_style),
            P("0.5x (No antibacterial effect)", cell_style),
        ],
        [
            P("<b>Scaffold Novelty (T<sub>max</sub>)</b>", cell_bold),
            P("<b>T<sub>max</sub> = 0.2115</b><br/>Novel (vs Metronidazole)", cell_style),
            P("<b>T<sub>max</sub> = 1.0000</b><br/>Known (Ciprofloxacin)", cell_style),
            P("T<sub>max</sub> = 0.19<br/>Non-antibiotic", cell_style),
            P("T<sub>max</sub> = 0.14<br/>Non-antibiotic", cell_style),
        ],
    ]
    story.append(
        KeepTogether(
            [
                P("Step C: Complete Output Dictionary &amp; Worked Comparison of 4 Real Compounds", h2_style),
                _make_table(ex_data, [3.5 * cm, 3.5 * cm, 3.5 * cm, 3.4 * cm, 3.4 * cm]),
            ]
        )
    )

    # ==================== SECTION 3: ALGORITHMS & MATH ====================
    story.append(Spacer(1, 8))
    story.append(P("3. Complete Algorithmic &amp; Mathematical Architecture", h1_style))

    story.append(P("3.1 Node (34-D), Edge (10-D), and Global eNTRy (12-D) Featurization", h2_style))
    story.append(
        P(
            "Implemented in <font face='Courier'>src/models/featurizer.py</font>, every molecule is encoded into three tensors:"
        )
    )
    feat_table = [
        [
            P("<b>Tensor</b>", cell_header),
            P("<b>Shape</b>", cell_header),
            P("<b>Exact Feature Breakdown</b>", cell_header),
        ],
        [
            P("<b>Node Matrix (x)</b>", cell_bold),
            P("[N<sub>atoms</sub>, 34]", cell_style),
            P(
                "• <b>Atom Symbol (12-D one-hot):</b> C, N, O, S, F, Cl, Br, I, P, B, Si, Other<br/>"
                "• <b>Bond Degree (6-D one-hot):</b> 0, 1, 2, 3, 4, 5+ neighbors<br/>"
                "• <b>Formal Charge (4-D one-hot):</b> -1, 0, +1, Other<br/>"
                "• <b>Hybridization (5-D one-hot):</b> sp, sp<sup>2</sup>, sp<sup>3</sup>, sp<sup>3</sup>d/sp<sup>3</sup>d<sup>2</sup>, Other<br/>"
                "• <b>Total Hydrogens (4-D one-hot):</b> 0, 1, 2, 3+ attached H atoms<br/>"
                "• <b>Chemical Flags (3-D binary):</b> IsAromatic, IsInRing, IsPrimaryAmineNitrogen",
                cell_style,
            ),
        ],
        [
            P("<b>Edge Matrix (edge_attr)</b>", cell_bold),
            P("[2&times;N<sub>bonds</sub>, 10]", cell_style),
            P(
                "• <b>Bond Order (5-D one-hot):</b> Single, Double, Triple, Aromatic, Other<br/>"
                "• <b>Stereochemistry (3-D one-hot):</b> StereoNone, Z/E/Cis/Trans, Other<br/>"
                "• <b>Topological Flags (2-D binary):</b> IsConjugated, IsInRing",
                cell_style,
            ),
        ],
        [
            P("<b>Global Vector (u)</b>", cell_bold),
            P("[1, 12]", cell_style),
            P(
                "<b>12 Richter eNTRy &amp; ADMET Descriptors:</b> MW, LogP, TPSA, HBD, HBA, RotBonds, "
                "RingCount, FractionCSP3, HasPrimaryAmine (SMARTS <font face='Courier'>[NX3;H2;!$(NC=[O,S,N])]</font>), "
                "eNTRy_Score (0–3), QED, SA_Score. Standardized via train-set z-score (<font face='Courier'>DescriptorScaler</font>).",
                cell_style,
            ),
        ],
    ]
    story.append(_make_table(feat_table, [3.2 * cm, 2.6 * cm, 11.5 * cm]))
    story.append(Spacer(1, 6))

    story.append(P("3.2 AMRGraphNet: Edge-Aware GATv2 + GIN Architecture", h2_style))
    story.append(
        P(
            "Implemented in <font face='Courier'>src/models/amr_gnn.py</font>, <b>AMRGraphNet</b> combines two complementary "
            "graph message-passing paradigms:"
        )
    )
    story.append(
        B(
            "<b>1. Dynamic Edge-Aware Graph Attention (<font face='Courier'>GATv2Conv</font>, 2 Layers, 4 Heads, 128-D):</b> "
            "Standard GCNs weight all chemical bonds equally. <font face='Courier'>GATv2Conv</font> computes dynamic attention "
            "coefficients &alpha;<sub>ij</sub> over both atom features <i>h<sub>i</sub>, h<sub>j</sub></i> and bond attributes "
            "<i>e<sub>ij</sub></i>:"
        )
    )
    story.append(
        Preformatted(
            "e_ij = a^T * LeakyReLU( W_l * h_i + W_r * h_j + W_e * edge_attr_ij )\n"
            "alpha_ij = softmax_j( e_ij )\n"
            "h_i^(l+1) = LayerNorm( h_i^(l) + Dropout( SiLU( Concat_heads( sum_j alpha_ij * W_r * h_j ) ) ) )",
            code_style,
        )
    )
    story.append(
        B(
            "<b>2. Graph Isomorphism Network (<font face='Courier'>GINConv</font>, 1 Layer, 128-D):</b> "
            "Proven maximally expressive among 1-WL message-passing networks for distinguishing non-isomorphic ring scaffolds:"
        )
    )
    story.append(
        Preformatted(
            "h_i^(3) = LayerNorm( h_i^(2) + Dropout( SiLU( MLP( (1 + eps) * h_i^(2) + sum_{j in N(i)} h_j^(2) ) ) ) )",
            code_style,
        )
    )
    story.append(
        B(
            "<b>3. Dual Global Readout + Physicochemical Descriptor Fusion:</b> "
            "Node embeddings are aggregated via both <b>Global Mean Pooling</b> (capturing overall molecular polarity/composition) "
            "and <b>Global Max Pooling</b> (capturing localized warheads like a nitrothiazole ring), then concatenated with "
            "the 64-D encoded global descriptor vector <i>u<sub>emb</sub></i> into a shared 128-D bottleneck feeding 5 task heads."
        )
    )
    story.append(
        B(
            "<b>4. Multi-Task Class-Balanced Loss Function:</b> "
            "Trained for 55 epochs using AdamW (<font face='Courier'>lr=1.5e-3, weight_decay=1e-4</font>) and Cosine Annealing, "
            "combining positive-class-weighted Binary Cross-Entropy (<font face='Courier'>BCEWithLogitsLoss</font>) for the "
            "3 classification tasks with <font face='Courier'>SmoothL1Loss</font> (Huber loss, &beta;=0.35) for the 2 regression tasks:"
        )
    )
    story.append(
        Preformatted(
            "L_total = 1.15 * BCE_weighted(logit_abx, y_abx)\n"
            "        + 1.00 * BCE_weighted(logit_gram_neg, y_gn)\n"
            "        + 1.15 * BCE_weighted(logit_cytotox, y_tox)\n"
            "        + 0.70 * SmoothL1(pred_pMIC, y_pMIC) + 0.70 * SmoothL1(pred_pCC50, y_pCC50)",
            code_style,
        )
    )

    story.append(P("3.3 Monte Carlo Dropout for Single-Molecule Epistemic Uncertainty", h2_style))
    story.append(
        P(
            "A major flaw of standard deep learning models in drug discovery is overconfidence on unfamiliar chemical scaffolds. "
            "In <font face='Courier'>AMRGraphNet.predict_with_uncertainty(data, num_samples=20)</font>, we keep "
            "<font face='Courier'>BatchNorm1d</font> layers frozen in <font face='Courier'>eval()</font> mode (so single-molecule "
            "inference with batch size 1 never fails) while activating <font face='Courier'>Dropout(p=0.25)</font> and "
            "<font face='Courier'>GATv2Conv</font> attention dropout across <i>T = 20</i> stochastic forward passes. "
            "The sample mean &mu; gives the prediction and the standard deviation &sigma; quantifies epistemic uncertainty."
        )
    )

    story.append(P("3.4 Counterfactual Atom-Masking XAI &amp; De Novo Designer Logic", h2_style))
    story.append(
        B(
            "<b>Batched Atom-Masking Attribution (<font face='Courier'>src/xai/substructure_explainer.py</font>):</b> "
            "For a molecule with <i>N</i> atoms, the explainer creates <i>N</i> copies of the molecular graph where atom "
            "<i>i</i>'s feature vector is zeroed out (<font face='Courier'>x[i, :] = 0</font>) and runs all <i>N</i> graphs "
            "in a single PyG <font face='Courier'>Batch</font>. The drop in logit "
            "<font face='Courier'>w_i = logit_full - logit_masked[i]</font> measures each atom's exact causal contribution "
            "to antibacterial activity (rendered in <b>Emerald</b>) and human cytotoxicity (rendered in <b>Crimson</b>)."
        )
    )
    story.append(
        B(
            "<b>Constrained De Novo Lead Optimization (<font face='Courier'>src/generative/antibiotic_designer.py</font>):</b> "
            "Applies 10 medicinal-chemistry SMARTS reactions (e.g., installing primary aminomethyl groups "
            "<font face='Courier'>[c;H1:1]&gt;&gt;[c:1]CN</font> for OmpF/OmpC porin entry, pyridine nitrogen insertion to lower "
            "LogP, or nitro-to-cyano detoxification) plus 100%-valid <b>SELFIES</b> token mutations. Analogs are filtered by "
            "<font face='Courier'>140 &le; MW &le; 620</font>, <font face='Courier'>SA_Score &le; 5.8</font>, "
            "<font face='Courier'>QED &ge; 0.22</font>, and ranked by the unified multi-objective score:"
        )
    )
    story.append(
        Preformatted(
            "AMR_Score = 0.35 * P(Abx) + 0.20 * clip((pMIC - 3.5)/5.0, 0, 1)\n"
            "          + 0.20 * (1 - P(Tox)) + 0.15 * (eNTRy_Score / 3.0) + 0.10 * (1 - T_max)",
            code_style,
        )
    )

    # ==================== SECTION 4: METRICS & FIGURES ====================
    story.append(Spacer(1, 8))
    story.append(P("4. Complete Guide to All Metrics &amp; Benchmark Results", h1_style))
    story.append(
        P(
            "When presenting this project to a professor or industry interviewer, you should be able to explain every "
            "statistical and chemical metric clearly:"
        )
    )

    metric_table = [
        [
            P("<b>Metric</b>", cell_header),
            P("<b>Formula / Range</b>", cell_header),
            P("<b>What It Means &amp; Why It Matters in Drug Discovery</b>", cell_header),
        ],
        [
            P("<b>ROC-AUC</b>", cell_bold),
            P("Area Under ROC Curve<br/>[0.5 = random, 1.0 = ideal]", cell_style),
            P(
                "Measures the probability that a randomly chosen active antibiotic is ranked higher than a randomly "
                "chosen inactive molecule across all thresholds.",
                cell_style,
            ),
        ],
        [
            P("<b>PR-AUC (AP)</b>", cell_bold),
            P("Area Under Precision-Recall<br/>[Baseline = class rate, 1.0 = ideal]", cell_style),
            P(
                "<b>Most important screening metric:</b> In drug discovery, active hits are the minority class. High PR-AUC "
                "(AMRGraphNet = <b>0.9614</b>) means top-ranked hits have very few false positives.",
                cell_style,
            ),
        ],
        [
            P("<b>F1 Score</b>", cell_bold),
            P("2&times;(Prec&times;Rec)/(Prec+Rec)<br/>[0.0 to 1.0]", cell_style),
            P(
                "Harmonic mean of Precision and Recall at the 0.50 probability threshold (AMRGraphNet = <b>0.9444</b> on active_abx).",
                cell_style,
            ),
        ],
        [
            P("<b>Brier Score</b>", cell_bold),
            P("Mean((y_true - P_pred)<sup>2</sup>)<br/><b>[Lower is better, 0.0 = ideal]</b>", cell_style),
            P(
                "Measures <b>probability calibration</b>. When AMRGraphNet says P(Abx) = 90%, is it truly active 90% of the time? "
                "AMRGraphNet achieves the lowest Brier score (<b>0.0350</b>).",
                cell_style,
            ),
        ],
        [
            P("<b>RMSE &amp; MAE</b>", cell_bold),
            P("Log<sub>10</sub> Molar units<br/><b>[Lower is better]</b>", cell_style),
            P(
                "Root Mean Squared Error and Mean Absolute Error on continuous pMIC and pCC50 predictions. "
                "An MAE of ~0.25 means predictions are within ~1.8-fold of true molar potency.",
                cell_style,
            ),
        ],
        [
            P("<b>Spearman r</b>", cell_bold),
            P("Rank correlation [-1 to +1]<br/><b>[Higher is better]</b>", cell_style),
            P(
                "Measures how accurately the model orders compounds from most potent to least potent (critical when picking "
                "the top 20 compounds to synthesize).",
                cell_style,
            ),
        ],
        [
            P("<b>pMIC &amp; MIC (&mu;M)</b>", cell_bold),
            P("pMIC = -log<sub>10</sub>(MIC [M])<br/>MIC [&mu;M] = 10<sup>6 - pMIC</sup>", cell_style),
            P(
                "Minimum Inhibitory Concentration required to stop bacterial growth. Higher pMIC = stronger antibiotic: "
                "pMIC 6.0 = 1 &mu;M; pMIC 7.0 = 0.1 &mu;M.",
                cell_style,
            ),
        ],
        [
            P("<b>pCC<sub>50</sub> &amp; SI</b>", cell_bold),
            P("SI = 10<sup>pMIC - pCC50</sup><br/><b>[SI &ge; 10 = Safe Window]</b>", cell_style),
            P(
                "pCC<sub>50</sub> measures mammalian HepG2 50% cytotoxic concentration. Selectivity Index (SI) is the ratio "
                "CC<sub>50</sub> / MIC. SI = 40x means the drug is 40 times more potent against bacteria than human cells.",
                cell_style,
            ),
        ],
        [
            P("<b>T<sub>max</sub> Novelty</b>", cell_bold),
            P("Max ECFP4 Tanimoto [0 to 1]<br/><b>[T<sub>max</sub> &lt; 0.40 = Novel]</b>", cell_style),
            P(
                "Jaccard bit overlap of 2048-bit Morgan fingerprints (radius 2) against 36 clinical antibiotics. "
                "T<sub>max</sub> &lt; 0.40 indicates a novel scaffold unlikely to suffer cross-resistance.",
                cell_style,
            ),
        ],
    ]
    story.append(_make_table(metric_table, [2.8 * cm, 4.1 * cm, 10.4 * cm]))
    story.append(Spacer(1, 8))

    story.append(P("Observed Zero-Leakage Held-Out Test Split Results (n = 99, 60 Unseen Scaffolds)", h2_style))
    bench_table = [
        [
            P("<b>Task / Endpoint</b>", cell_header),
            P("<b>AMRGraphNet (GATv2+GIN)</b>", cell_header),
            P("<b>XGBoost (ECFP4+eNTRy)</b>", cell_header),
            P("<b>RandomForest (ECFP4+eNTRy)</b>", cell_header),
        ],
        [
            P("<b>active_abx</b> (ROC / PR / F1 / Brier)", cell_bold),
            P("0.9797 / <b>0.9614</b> / <b>0.9444</b> / <b>0.0350</b>", cell_style),
            P("<b>0.9806</b> / 0.9498 / 0.9143 / 0.0453", cell_style),
            P("0.9749 / 0.9393 / 0.8857 / 0.0677", cell_style),
        ],
        [
            P("<b>gram_neg_accum</b> (ROC / PR / F1 / Brier)", cell_bold),
            P("0.9500 / <b>0.9553</b> / <b>0.8974</b> / <b>0.0693</b>", cell_style),
            P("<b>0.9525</b> / 0.9488 / <b>0.8974</b> / 0.0748", cell_style),
            P("0.9512 / 0.9416 / 0.8947 / 0.0701", cell_style),
        ],
        [
            P("<b>cytotoxic_human</b> (ROC / PR / F1 / Brier)", cell_bold),
            P("0.9419 / 0.8238 / 0.8750 / 0.0741", cell_style),
            P("<b>0.9813</b> / <b>0.9440</b> / <b>0.8814</b> / <b>0.0444</b>", cell_style),
            P("0.9764 / 0.9399 / 0.8727 / 0.0812", cell_style),
        ],
        [
            P("<b>pMIC Regression</b> (RMSE / MAE / Spearman r)", cell_bold),
            P("0.4854 / 0.2681 / 0.7646", cell_style),
            P("<b>0.4032</b> / <b>0.2302</b> / <b>0.8445</b>", cell_style),
            P("0.4709 / 0.2561 / 0.8170", cell_style),
        ],
        [
            P("<b>pCC50 Regression</b> (RMSE / MAE / Spearman r)", cell_bold),
            P("0.3919 / 0.2995 / 0.7773", cell_style),
            P("<b>0.3355</b> / 0.2495 / <b>0.7961</b>", cell_style),
            P("0.3464 / <b>0.2422</b> / 0.7717", cell_style),
        ],
    ]
    story.append(_make_table(bench_table, [4.6 * cm, 4.3 * cm, 4.2 * cm, 4.2 * cm]))

    # Embed Figures
    story.append(PageBreak())
    story.append(P("Visual Benchmark &amp; Explainable AI Diagnostics", h1_style))

    fig1 = FIGURES_DIR / "roc_pr_benchmark.png"
    if fig1.exists():
        story.append(Image(str(fig1), width=17.0 * cm, height=10.1 * cm))
        story.append(
            P(
                "Figure 1: Held-out scaffold-split ROC and Precision-Recall curves across Antibacterial Inhibition, "
                "Gram-Negative Outer-Membrane Entry, and Human HepG2 Cytotoxicity.",
                caption_style,
            )
        )

    fig2 = FIGURES_DIR / "chemical_space_novelty.png"
    if fig2.exists():
        story.append(Image(str(fig2), width=17.0 * cm, height=7.0 * cm))
        story.append(
            P(
                "Figure 2: Left — 2D PCA of 2048-bit ECFP4 chemical space showing novel scaffolds vs. 36 FDA/EMA clinical "
                "antibiotics. Right — Structural novelty distribution (99.6% of library scaffolds sit below T_max < 0.40).",
                caption_style,
            )
        )

    story.append(PageBreak())
    fig3 = FIGURES_DIR / "entry_selectivity_analysis.png"
    if fig3.exists():
        story.append(Image(str(fig3), width=17.0 * cm, height=7.0 * cm))
        story.append(
            P(
                "Figure 3: Left — Empirical validation of Richter eNTRy rules on held-out scaffolds (97.1% accumulation when "
                "eNTRy = 3/3). Right — Predicted Potency (pMIC) vs. Cytotoxicity (pCC50) Selectivity Index landscape.",
                caption_style,
            )
        )

    fig4 = FIGURES_DIR / "halicin_abaucin_xai.png"
    if fig4.exists():
        story.append(Image(str(fig4), width=17.0 * cm, height=10.2 * cm))
        story.append(
            P(
                "Figure 4: Out-of-sample XAI atom attribution maps on Halicin (SU3327), Abaucin (RS102895), and Ciprofloxacin.",
                caption_style,
            )
        )

    # ==================== SECTION 5: USER MANUAL ====================
    story.append(PageBreak())
    story.append(P("5. Step-by-Step User Manual: How to Run &amp; Use the Platform", h1_style))

    story.append(P("5.1 Command-Line Execution (PowerShell / Terminal)", h2_style))
    story.append(
        P(
            "Open PowerShell and navigate to the project directory. Note: On your Windows machine, use <font face='Courier'>py</font> "
            "(Python 3.14 launcher) to invoke Python:"
        )
    )
    story.append(
        Preformatted(
            "cd C:\\Users\\GOD\\.gemini\\antigravity\\scratch\\amr-gnn-discovery\n\n"
            "# 1. Rebuild standardized dataset & zero-leakage scaffold split (989 compounds)\n"
            "py -m src.data_pipeline.prepare_amr_dataset\n\n"
            "# 2. Train RF, XGBoost & AMRGraphNet, evaluate test split, and regenerate 300 DPI figures\n"
            "py -m src.evaluation.run_benchmarks\n\n"
            "# 3. Run automated verification test suite (6 end-to-end unit tests)\n"
            "py -m unittest discover -s tests -v\n\n"
            "# 4. Launch the interactive 5-tab Streamlit web application (auto-opens http://localhost:8501)\n"
            "py -m streamlit run app/main.py",
            code_style,
        )
    )

    story.append(P("5.2 How to Use All 5 Tabs of the Streamlit Web App", h2_style))
    story.append(
        B(
            "<b>Left Sidebar (Input Controls):</b> Choose any of the 9 benchmark molecules from the dropdown "
            "(e.g., <i>Halicin, Abaucin, Ciprofloxacin, 6-Amino-2-Methyl-Quinoline, Trimethoprim, Linezolid, Nitrofurantoin, "
            "Doxorubicin-like Control, Ibuprofen</i>) OR paste <b>any custom SMILES string</b> from PubChem/ChEMBL into the "
            "text box. Adjust the <b>MC Dropout Epistemic Passes (T)</b> slider (5–50) and <b>Novelty Cutoff (T<sub>max</sub>)</b>."
        )
    )
    story.append(
        B(
            "<b>Tab 1 — 1. AMR &amp; Safety Profiler:</b> Displays 6 live metric cards (Antibacterial Prob. &plusmn; &sigma;, "
            "Predicted pMIC &amp; MIC in &mu;M, Human Cytotoxicity, Selectivity Index, Gram-Neg eNTRy score, and Scaffold Novelty), "
            "the Composite AMR Discovery Score progress bar, cross-resistance assessment, and the 12-descriptor table."
        )
    )
    story.append(
        B(
            "<b>Tab 2 — 2. Explainable AI (XAI) &amp; eNTRy Audit:</b> Computes live counterfactual atom-masking attributions "
            "and renders two side-by-side 2D RDKit molecular heatmaps: <b>Left = Antibacterial Pharmacophore</b> (Emerald = "
            "drives activity), <b>Right = Human Cytotoxicity Liability</b> (Crimson = drives toxicity). Below, inspect the "
            "3-rule Richter eNTRy PASS/FAIL table and PAINS/Brenk structural alert badges."
        )
    )
    story.append(
        B(
            "<b>Tab 3 — 3. Virtual Screening:</b> Screen either the 99 held-out test compounds or paste a custom list of SMILES "
            "(one per line). Use the 4 interactive sliders (Min P(Abx), Max T<sub>max</sub>, Min Selectivity Index, Min eNTRy "
            "Score) to filter hits, inspect the pMIC vs. P(Tox) scatter plot, and click <b>Download Prioritized Hits (CSV)</b>."
        )
    )
    story.append(
        B(
            "<b>Tab 4 — 4. De Novo Lead Optimizer:</b> Click <b>Generate &amp; Rank Optimized Analogs</b> to mutate your active "
            "molecule using SMARTS bioisosteric edits and SELFIES mutations. View the 2D chemical structures of the top 6 "
            "generated analogs, compare their improved eNTRy and Selectivity Index scores, and export them as CSV."
        )
    )
    story.append(
        B(
            "<b>Tab 5 — 5. Scaffold Benchmarks &amp; Research Fit:</b> Inspect the zero-leakage test split classification "
            "and regression tables comparing AMRGraphNet, XGBoost, and RandomForest, along with all 4 publication figures."
        )
    )

    # ==================== SECTION 6: GERMAN MASTER'S GUIDE ====================
    story.append(P("6. German Master's Application &amp; Interview Guide", h1_style))
    story.append(
        P(
            "When applying to German M.Sc. programs (such as <b>TU Munich, LMU Munich, Universität Heidelberg, Universität "
            "Tübingen, RWTH Aachen, FU Berlin / Charité, or Saarland University / HIPS Saarbrücken</b>), admissions committees "
            "look for scientific rigor, honest evaluation, and biological domain depth."
        )
    )
    story.append(P("High-Impact CV Bullet Points You Can Use Directly", h2_style))
    story.append(
        B(
            "<b>PathoGraph-AMR (PyTorch Geometric, RDKit, XGBoost, SELFIES, Streamlit):</b> Engineered an end-to-end "
            "multi-task Graph Attention &amp; Isomorphism Network (<font face='Courier'>GATv2Conv + GINConv</font>) for "
            "structurally novel antibiotic discovery, Gram-negative porin accumulation (Richter eNTRy rules), and mammalian "
            "HepG2 cytotoxicity counter-screening."
        )
    )
    story.append(
        B(
            "Implemented deterministic <b>Bemis-Murcko scaffold splitting</b> across 989 standardized compounds (638 unique "
            "scaffolds, zero scaffold leakage), achieving <b>0.9797 ROC-AUC (0.9614 PR-AUC, 0.0350 Brier score)</b> on unseen "
            "antibacterial scaffolds and recovering AI-discovered Halicin out-of-sample (SI = 23.3x, T<sub>max</sub> = 0.21)."
        )
    )
    story.append(
        B(
            "Developed batched counterfactual atom-masking Explainable AI (XAI) pharmacophore/toxicophore heatmaps and an "
            "eNTRy-guided SMARTS + SELFIES de novo analog optimizer deployed in a containerized 5-tab Streamlit workbench."
        )
    )

    story.append(P("Top 3 Interview Questions &amp; How to Answer Them", h2_style))
    story.append(
        B(
            "<b>Q1: Why did you use Bemis-Murcko scaffold splitting instead of a standard random 80/10/10 split?</b><br/>"
            "<i>Answer:</i> In medicinal chemistry, datasets contain congeneric series (molecules sharing the same core ring "
            "with minor methyl or halogen swaps). Random splitting puts analogs of the same scaffold into both train and test, "
            "causing data leakage and inflated accuracy. Bemis-Murcko splitting strips side chains and groups by core ring "
            "skeleton so all 60 scaffolds in my test set are completely unseen during training."
        )
    )
    story.append(
        B(
            "<b>Q2: Why did you combine GATv2Conv, GINConv, and 12 global eNTRy descriptors?</b><br/>"
            "<i>Answer:</i> <font face='Courier'>GATv2Conv</font> uses edge-aware dynamic attention to weight specific chemical "
            "bonds and warheads (like nitrothiazoles or keto-acids), while <font face='Courier'>GINConv</font> provides 1-WL "
            "topological power to distinguish ring scaffolds. However, message passing over 3 hops is localized, whereas "
            "Gram-negative porin entry depends on whole-molecule macroscopic properties (Rotatable Bonds &le; 5, MW, LogP, TPSA, "
            "and primary amine presence). Fusing the 12 global eNTRy descriptors into the graph readout bridges local "
            "substructures with whole-cell permeability physics."
        )
    )
    story.append(
        B(
            "<b>Q3: Why did your model recover Halicin out-of-sample (P(Abx)=0.99) but score Abaucin low (P(Abx)=0.00)?</b><br/>"
            "<i>Answer:</i> Both Halicin and Abaucin were strictly held out in the test split. Halicin shares nitro-heterocyclic "
            "and primary-amine eNTRy pharmacophore patterns present in other scaffold families, so the GNN generalized across "
            "scaffolds to recover it. Abaucin is a narrow-spectrum spiro-benzoxazinone LolE lipoprotein-transport inhibitor "
            "with a tertiary amine and no classical antibacterial warhead; because that mechanism was absent from the training "
            "scaffolds, the model conservatively predicted low activity. Documenting this out-of-distribution boundary "
            "highlights where active learning and wet-lab assay expansion should be directed next."
        )
    )

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


if __name__ == "__main__":
    out_file = DOCS_DIR / "PathoGraph_AMR_Complete_Master_Guide.pdf"
    build_pdf(out_file)
    print(f"Generated PDF guide at: {out_file} ({out_file.stat().st_size} bytes)")

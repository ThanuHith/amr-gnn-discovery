"""AMRGraphNet: Multi-Task Edge-Aware GATv2 + GIN Network with MC Dropout Uncertainty.

Architecture:
- Edge-aware GATv2Conv layers (2 layers x 4 attention heads, edge_dim=10) with
  residual connections and LayerNorm to attend over antibacterial warheads and
  toxicophore substructures.
- GINConv layer (sum-aggregated MLP message passing) for topological ring/scaffold
  discrimination.
- Dual global readout (global_mean_pool + global_max_pool) fused with a learned
  projection of the 12 Richter eNTRy & physicochemical descriptors.
- Shared 128-D representation bottleneck feeding 5 multi-task heads:
  1. active_abx (antibacterial growth inhibition)
  2. pMIC (continuous antibacterial potency)
  3. gram_neg_accum (Gram-negative outer-membrane accumulation)
  4. cytotoxic_human (mammalian HepG2 cytotoxicity)
  5. pCC50 (continuous cytotoxicity)
- Monte Carlo Dropout (T=20 stochastic passes) for epistemic uncertainty.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GINConv, global_max_pool, global_mean_pool

from src.models.featurizer import EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM


class AMRGraphNet(nn.Module):
    """Multi-task graph attention + isomorphism network for antibiotic discovery."""

    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        head_dim = hidden_dim // heads
        self.gat1 = GATv2Conv(
            hidden_dim,
            head_dim,
            heads=heads,
            edge_dim=edge_dim,
            dropout=0.15,
            add_self_loops=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATv2Conv(
            hidden_dim,
            head_dim,
            heads=heads,
            edge_dim=edge_dim,
            dropout=0.15,
            add_self_loops=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gin = GINConv(gin_mlp, train_eps=True)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.drop = nn.Dropout(p=dropout)

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

        self.head_abx = self._make_head(hidden_dim, dropout)
        self.head_pmic = self._make_head(hidden_dim, dropout)
        self.head_gn = self._make_head(hidden_dim, dropout)
        self.head_tox = self._make_head(hidden_dim, dropout)
        self.head_pcc50 = self._make_head(hidden_dim, dropout)

    @staticmethod
    def _make_head(in_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def encode_graph(
        self,
        data: Data | Batch,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute node-level and fused graph-level representations."""
        x = data.x
        if node_mask is not None:
            x = x * node_mask

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x)

        h1 = self.gat1(h, edge_index, edge_attr)
        h = self.norm1(h + self.drop(F.silu(h1)))

        h2 = self.gat2(h, edge_index, edge_attr)
        h = self.norm2(h + self.drop(F.silu(h2)))

        h3 = self.gin(h, edge_index)
        node_emb = self.norm3(h + self.drop(F.silu(h3)))

        pooled = torch.cat(
            [global_mean_pool(node_emb, batch), global_max_pool(node_emb, batch)],
            dim=-1,
        )

        u = data.u
        if u.dim() == 1:
            u = u.unsqueeze(0)
        u_emb = self.global_encoder(u)

        graph_emb = self.fusion_bottleneck(torch.cat([pooled, u_emb], dim=-1))
        return graph_emb, node_emb

    def forward(
        self,
        data: Data | Batch,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning multi-task predictions and shared bottleneck embedding."""
        graph_emb, node_emb = self.encode_graph(data, node_mask=node_mask)
        return {
            "logit_abx": self.head_abx(graph_emb).squeeze(-1),
            "pred_pMIC": self.head_pmic(graph_emb).squeeze(-1),
            "logit_gram_neg": self.head_gn(graph_emb).squeeze(-1),
            "logit_cytotox": self.head_tox(graph_emb).squeeze(-1),
            "pred_pCC50": self.head_pcc50(graph_emb).squeeze(-1),
            "graph_embedding": graph_emb,
            "node_embeddings": node_emb,
        }

    def _enable_mc_dropout(self) -> None:
        """Keep BatchNorm in eval mode while enabling Dropout layers for MC sampling."""
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
            elif isinstance(module, GATv2Conv):
                # GATv2Conv internal edge/attention dropout relies on module.training
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        data: Data | Batch,
        num_samples: int = 20,
    ) -> Dict[str, object]:
        """Run Monte Carlo Dropout inference and compute epistemic uncertainty."""
        device = next(self.parameters()).device
        batch_data = data.to(device)
        if getattr(batch_data, "batch", None) is None:
            batch_data.batch = torch.zeros(batch_data.x.size(0), dtype=torch.long, device=device)

        if num_samples <= 1:
            self.eval()
            out = self.forward(batch_data)
            p_abx = float(torch.sigmoid(out["logit_abx"]).item())
            pmic = float(out["pred_pMIC"].item())
            p_gn = float(torch.sigmoid(out["logit_gram_neg"]).item())
            p_tox = float(torch.sigmoid(out["logit_cytotox"]).item())
            pcc50 = float(out["pred_pCC50"].item())
            si = float(10.0 ** max(min(pmic - pcc50, 6.0), -6.0))
            return {
                "prob_abx": p_abx,
                "abx_std": 0.0,
                "pred_pMIC": pmic,
                "pMIC_std": 0.0,
                "mic_uM": float(10.0 ** (6.0 - pmic)),
                "prob_gram_neg": p_gn,
                "gram_neg_std": 0.0,
                "prob_cytotox": p_tox,
                "cytotox_std": 0.0,
                "pred_pCC50": pcc50,
                "pCC50_std": 0.0,
                "selectivity_index": si,
                "graph_embedding": out["graph_embedding"].squeeze(0).cpu().numpy(),
            }

        self._enable_mc_dropout()
        probs_abx, pmics, probs_gn, probs_tox, pcc50s, embs = [], [], [], [], [], []
        for _ in range(num_samples):
            out = self.forward(batch_data)
            probs_abx.append(torch.sigmoid(out["logit_abx"]).item())
            pmics.append(out["pred_pMIC"].item())
            probs_gn.append(torch.sigmoid(out["logit_gram_neg"]).item())
            probs_tox.append(torch.sigmoid(out["logit_cytotox"]).item())
            pcc50s.append(out["pred_pCC50"].item())
            embs.append(out["graph_embedding"].squeeze(0).cpu())

        self.eval()

        t_abx = torch.tensor(probs_abx)
        t_pmic = torch.tensor(pmics)
        t_gn = torch.tensor(probs_gn)
        t_tox = torch.tensor(probs_tox)
        t_pcc50 = torch.tensor(pcc50s)

        mean_pmic = float(t_pmic.mean().item())
        mean_pcc50 = float(t_pcc50.mean().item())
        si = float(10.0 ** max(min(mean_pmic - mean_pcc50, 6.0), -6.0))

        return {
            "prob_abx": float(t_abx.mean().item()),
            "abx_std": float(t_abx.std(unbiased=False).item()),
            "pred_pMIC": mean_pmic,
            "pMIC_std": float(t_pmic.std(unbiased=False).item()),
            "mic_uM": float(10.0 ** (6.0 - mean_pmic)),
            "prob_gram_neg": float(t_gn.mean().item()),
            "gram_neg_std": float(t_gn.std(unbiased=False).item()),
            "prob_cytotox": float(t_tox.mean().item()),
            "cytotox_std": float(t_tox.std(unbiased=False).item()),
            "pred_pCC50": mean_pcc50,
            "pCC50_std": float(t_pcc50.std(unbiased=False).item()),
            "selectivity_index": si,
            "graph_embedding": torch.stack(embs, dim=0).mean(dim=0).numpy(),
        }

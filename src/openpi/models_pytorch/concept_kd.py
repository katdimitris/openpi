"""Multimodal Concept KD for shallow pi0.5 distillation.

This module implements an internal-semantic-concept distillation loss that aligns
student and teacher hidden representations through a set of learned concept vectors
("concepts"). For each (modality x layer pair), independent concept banks and
optional projectors are learned. Teacher and student token distributions over the
concept bank are aligned via three losses (mirroring the original ViT ConceptKD):

  - L_soft   : CE(p_t, p_s)   student matches teacher's soft concept distribution.
  - L_teacher: CE(q_t, p_t)   teacher aligns to its Sinkhorn-balanced targets.
  - L_student: CE(q_s, p_s)   student aligns to its own Sinkhorn-balanced targets.
  - total = (L_soft + L_teacher + L_student) / 2

Gradients flow to the concept bank from all three terms.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch import nn


_MODALITIES = ("visual", "language", "action")


def _modality_num_concepts(config) -> dict[str, int]:
    return {
        "visual": config.concept_num_visual,
        "language": config.concept_num_language,
        "action": config.concept_num_action,
    }


def normalize_mean_std(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample mean/std normalization across the token/concept dimension (dim=1).

    Uses unbiased std (Bessel correction) to match the original ViT ConceptKD.
    nan_to_num guards the K=1 edge case (single token after masking → std=NaN).
    """
    std = x.std(dim=1, keepdim=True).nan_to_num(nan=0.0)
    return (x - x.mean(dim=1, keepdim=True)) / (std + eps)


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    """Balanced Sinkhorn-Knopp normalization, SwAV `distributed_sinkhorn` style.

    Args:
      scores: [1, K, C] similarity logits (cosine or negative L2).
        K = local sample count, C = number of concepts (prototypes).
      eps: temperature for the entropic regularizer.
      n_iters: number of Sinkhorn iterations.

    Returns:
      Q: [1, K, C] assignment matrix. Rows sum to ~1 (one assignment per sample).
    """
    s = scores.detach().to(torch.float32)
    assert s.dim() == 3 and s.shape[0] == 1, f"expected [1, K, C], got {tuple(s.shape)}"

    # Match SwAV variable conventions: Q is K-by-B (prototypes-by-samples).
    Q = torch.exp(s.squeeze(0) / eps).t()  # [C, B_local]

    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    B = Q.shape[1] * world_size  # global samples
    K = Q.shape[0]  # prototypes (concepts)

    sum_Q = torch.sum(Q)
    if world_size > 1:
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for _ in range(n_iters):
        # Normalize each row: total weight per prototype must be 1/K.
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # Normalize each column: total weight per sample must be 1/B.
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # columns sum to 1 so Q is an assignment
    return Q.t().unsqueeze(0)  # [1, K, C]


class _PerKeyModule(nn.Module):
    """Container with concept bank + optional projectors for one (modality, pair)."""

    def __init__(
        self,
        num_concepts: int,
        concept_dim: int,
        student_dim: int,
        teacher_dim: int,
        use_student_proj: bool,
        use_teacher_proj: bool,
        proj_bias: bool,
        freeze_concepts: bool = False,
    ):
        super().__init__()
        # Concepts are kept in float32 for stability.
        self.concepts = nn.Parameter(torch.empty(num_concepts, concept_dim).normal_(0.0, 0.02))
        if freeze_concepts:
            self.concepts.requires_grad_(False)
        self.student_proj = (
            nn.Linear(student_dim, concept_dim, bias=proj_bias) if use_student_proj else None
        )
        self.teacher_proj = (
            nn.Linear(teacher_dim, concept_dim, bias=proj_bias) if use_teacher_proj else None
        )
        if self.student_proj is not None:
            self.student_proj.to(torch.float32)
        if self.teacher_proj is not None:
            self.teacher_proj.to(torch.float32)


class ConceptKDModule(nn.Module):
    """Multimodal concept distillation module.

    Holds independent concept banks per (modality, layer pair) and computes
    teacher/student alignment losses.
    """

    def __init__(
        self,
        config,
        *,
        student_prefix_dim: int,
        teacher_prefix_dim: int,
        student_action_dim: int,
        teacher_action_dim: int,
    ):
        super().__init__()
        self.config = config
        self.modalities: tuple[str, ...] = tuple(config.concept_modalities)
        self.layer_pairs: tuple[tuple[int, int], ...] = tuple(tuple(p) for p in config.concept_layer_pairs)
        self.similarity: str = config.concept_similarity
        self.temperature: float = config.concept_temperature
        self.sample_ratio: float = config.concept_sample_ratio
        self.max_tokens: int = config.concept_max_tokens
        self.use_student_proj: bool = config.concept_use_student_projector
        self.use_teacher_proj: bool = config.concept_use_teacher_projector
        self.proj_bias: bool = config.concept_projector_bias
        self.sinkhorn_eps: float = config.concept_sinkhorn_eps
        self.sinkhorn_iters: int = config.concept_sinkhorn_iters
        self.freeze_prototypes: bool = getattr(config, "concept_freeze_prototypes", False)
        self.soft_loss_only: bool = getattr(config, "concept_soft_loss_only", False)

        for m in self.modalities:
            if m not in _MODALITIES:
                raise ValueError(f"Unknown modality: {m}. Allowed: {_MODALITIES}")

        # Dimensions per modality: prefix branch (visual/language) uses paligemma width,
        # action tokens use the gemma_expert width.
        self._modality_dims: dict[str, tuple[int, int]] = {
            "visual": (student_prefix_dim, teacher_prefix_dim),
            "language": (student_prefix_dim, teacher_prefix_dim),
            "action": (student_action_dim, teacher_action_dim),
        }

        per_modality_num = _modality_num_concepts(config)

        # Build one _PerKeyModule per (modality, layer pair).
        self.banks = nn.ModuleDict()
        for modality in self.modalities:
            s_dim, t_dim = self._modality_dims[modality]
            concept_dim = t_dim  # per spec: concept_dim defaults to teacher hidden dim
            num_concepts = per_modality_num[modality]
            for s_layer, t_layer in self.layer_pairs:
                key = self._key(modality, s_layer, t_layer)
                self.banks[key] = _PerKeyModule(
                    num_concepts=num_concepts,
                    concept_dim=concept_dim,
                    student_dim=s_dim,
                    teacher_dim=t_dim,
                    use_student_proj=self.use_student_proj,
                    use_teacher_proj=self.use_teacher_proj,
                    proj_bias=self.proj_bias,
                    freeze_concepts=self.freeze_prototypes,
                )

        # Optionally load k-means initialized concepts.
        if config.concept_init_path is not None:
            self.load_init_from_file(config.concept_init_path)

    @staticmethod
    def _key(modality: str, s_layer: int, t_layer: int) -> str:
        return f"{modality}_s{s_layer}_t{t_layer}"

    def teacher_layer_indices(self) -> set[int]:
        return {t for _, t in self.layer_pairs}

    def student_layer_indices(self) -> set[int]:
        return {s for s, _ in self.layer_pairs}

    def load_init_from_file(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        loaded = 0
        for key, mod in self.banks.items():
            if key in state:
                v = state[key]
                if v.shape == mod.concepts.shape:
                    with torch.no_grad():
                        mod.concepts.copy_(v.to(mod.concepts.dtype))
                    loaded += 1
                else:
                    logging.warning(
                        f"Concept init shape mismatch for key={key}: file={tuple(v.shape)} expected={tuple(mod.concepts.shape)}"
                    )
        logging.info(f"Loaded concept init for {loaded}/{len(self.banks)} banks from {path}")

    # ------------------------------------------------------------------ bank diagnostics
    @torch.no_grad()
    def _compute_bank_diagnostics(self) -> dict:
        """Per-bank diagnostics: Von Neumann entropy and mean inter-concept cosine.

        Von Neumann entropy treats the normalized Gram matrix as a density matrix.
        High entropy = concepts span many independent directions (diverse bank).
        Low entropy = concepts collapsed to a low-dimensional subspace.

        Mean cosine (off-diagonal) is a fast collapse proxy:
        near 0 = orthogonal/diverse; near 1 = all concepts pointing the same way.
        """
        diag: dict = {}
        for key, bank in self.banks.items():
            C_norm = F.normalize(bank.concepts.float(), p=2, dim=1)  # [C, D]
            C = C_norm.shape[0]
            G = C_norm @ C_norm.T  # [C, C] pairwise cosine similarities
            eigvals = torch.linalg.eigvalsh(G / C).clamp(min=1e-12)
            vn_entropy = -(eigvals * eigvals.log()).sum()
            mean_cos = (G.sum() - G.trace()) / max(C * (C - 1), 1)
            diag[f"concept_diagnostics/vn_entropy/{key}"] = vn_entropy
            diag[f"concept_diagnostics/mean_cos/{key}"] = mean_cos
        return diag

    # ------------------------------------------------------------------ similarity
    def _similarity(self, x: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
        """Returns logits/similarities of shape [1, K, C]."""
        if self.similarity == "cosine":
            x_n = F.normalize(x, p=2, dim=-1)
            c_n = F.normalize(concepts, p=2, dim=-1)
            return torch.bmm(x_n, c_n.transpose(1, 2))
        if self.similarity == "l2":
            dist2 = torch.cdist(x, concepts, p=2).pow(2) / x.shape[-1]
            return -dist2
        raise ValueError(f"Unknown similarity: {self.similarity}")

    def _sample_indices(self, total: int, device) -> torch.Tensor:
        K = min(int(total * self.sample_ratio), self.max_tokens)
        K = max(K, 1)
        return torch.randperm(total, device=device)[:K]

    def _flatten_with_mask(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Flatten [B, N, D] -> [M, D], optionally keeping only mask==True positions."""
        if mask is None:
            B, N, D = tokens.shape
            return tokens.reshape(B * N, D)
        # mask: [B, N] bool
        return tokens[mask.bool()]

    # ------------------------------------------------------------------ per-pair loss
    def _compute_pair_loss(
        self,
        key: str,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Compute (L_soft_kd, L_teacher_sinkhorn, L_student_sinkhorn, diagnostics) for one (modality, layer pair).

        Default (soft_loss_only=False): mirrors the three-loss formulation from the original ViT ConceptKD:
          loss1 = CE(p_t, p_s)       student matches teacher's soft concept distribution.
          loss2 = CE(q_t, p_t)       teacher aligns to its own Sinkhorn-balanced targets.
          loss3 = CE(q_s, p_s)       student aligns to its own Sinkhorn-balanced targets.
          total = (loss1 + loss2 + loss3) / 2  (combined in `forward`)

        Concepts are normalized the same way as tokens (normalize_mean_std) so all three
        live in the same statistical space before distances are computed. Gradients flow
        to the concept bank from all three loss terms.

        soft_loss_only=True: only L_soft is computed, with M_t detached so gradients flow
        only to the student. L_teacher and L_student are returned as zeros. Intended for
        the kmeans_fixed regime where prototypes are frozen.
        """
        bank = self.banks[key]

        # Flatten + drop padding before sampling.
        s_flat = self._flatten_with_mask(student_tokens, mask)  # [M, D_s]
        t_flat = self._flatten_with_mask(teacher_tokens, mask)  # [M, D_t]
        total = s_flat.shape[0]
        if total == 0:
            zero = next(self.parameters()).new_zeros(())
            return zero, zero, zero, {}

        idx = self._sample_indices(total, s_flat.device)
        s_sampled = s_flat[idx].to(torch.float32)  # [K, D_s]
        t_sampled = t_flat[idx].to(torch.float32)  # [K, D_t]

        # Project to concept dim.
        s_proj = bank.student_proj(s_sampled) if bank.student_proj is not None else s_sampled
        t_proj = bank.teacher_proj(t_sampled) if bank.teacher_proj is not None else t_sampled

        # Add batch dim so dim=1 is the sample axis used by normalize_mean_std.
        s_proj = s_proj.unsqueeze(0)  # [1, K, D]
        t_proj = t_proj.unsqueeze(0)

        # Normalize tokens and concepts into the same statistical space.
        s_proj = normalize_mean_std(s_proj)
        t_proj = normalize_mean_std(t_proj)
        concepts = normalize_mean_std(bank.concepts.unsqueeze(0))  # [1, C, D]

        # Similarities to concepts.
        M_s = self._similarity(s_proj, concepts)   # [1, K, C]
        # Detach the teacher branch in soft-only mode so gradients flow only to the student
        # (and not through p_t into the teacher).
        M_t_raw = self._similarity(t_proj, concepts)   # [1, K, C]
        M_t = M_t_raw.detach() if self.soft_loss_only else M_t_raw

        # Softmax distributions with temperature.
        log_p_t = F.log_softmax(M_t / self.temperature, dim=-1)
        log_p_s = F.log_softmax(M_s / self.temperature, dim=-1)
        p_t = log_p_t.exp()

        # loss1: student matches teacher's soft distribution (cross-entropy).
        L_soft = -(p_t * log_p_s).sum(dim=-1).mean()

        if self.soft_loss_only:
            # Skip L_teacher and L_student entirely — concepts are frozen and we want no
            # gradients flowing through L_student to the student either.
            zero = L_soft.new_zeros(())
            L_teacher = zero
            L_student = zero
            # No q_t computed; use p_t as a placeholder for the histogram so consumers don't break.
            q_t_for_hist = p_t.detach()
        else:
            # Balanced Sinkhorn targets (detached — used as fixed assignment targets).
            q_t = sinkhorn(M_t, eps=self.sinkhorn_eps, n_iters=self.sinkhorn_iters)  # [1, K, C]
            q_s = sinkhorn(M_s, eps=self.sinkhorn_eps, n_iters=self.sinkhorn_iters)  # [1, K, C]
            # loss2: teacher aligns to its Sinkhorn-balanced targets.
            L_teacher = -(q_t * log_p_t).sum(dim=-1).mean()
            # loss3: student aligns to its own Sinkhorn-balanced targets.
            L_student = -(q_s * log_p_s).sum(dim=-1).mean()
            q_t_for_hist = q_t

        # --- per-pair diagnostics (no_grad, negligible overhead) ---
        with torch.no_grad():
            p_s = log_p_s.exp()
            # Mean per-token entropy (higher = more uniform, lower = more peaked).
            H_pt = -(p_t * log_p_t).sum(dim=-1).mean()
            H_ps = -(p_s * log_p_s).sum(dim=-1).mean()
            # Effective number of concepts: exp(H(marginal)).  Range: [1, C].
            # 1 = all tokens assigned to one concept; C = all concepts equally used.
            marginal_t = p_t.mean(dim=1).clamp(min=1e-12)  # [1, C]
            marginal_s = p_s.mean(dim=1).clamp(min=1e-12)
            ENK_t = (-(marginal_t * marginal_t.log()).sum()).exp()
            ENK_s = (-(marginal_s * marginal_s.log()).sum()).exp()
            n = min(3, p_t.shape[1])
            pair_diag = {
                "H_pt": H_pt,
                "H_ps": H_ps,
                "ENK_t": ENK_t,
                "ENK_s": ENK_s,
                "pt_sample": p_t[0, :n].float(),   # [n, C]
                "qt_sample": q_t_for_hist[0, :n].float(),
                "ps_sample": p_s[0, :n].float(),
            }

        return L_soft, L_teacher, L_student, pair_diag

    # ------------------------------------------------------------------ public API
    def forward(
        self,
        *,
        student_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        teacher_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        num_visual_tokens: int,
        num_language_tokens: int,
        action_horizon: int,
        language_mask: torch.Tensor | None,
        visual_mask: torch.Tensor | None = None,
    ) -> dict:
        """Compute concept distillation losses.

        Args:
          student_states: layer_idx -> (prefix_hidden, suffix_hidden)
          teacher_states: layer_idx -> (prefix_hidden, suffix_hidden)
          num_visual_tokens: count of visual tokens at the start of the prefix.
          num_language_tokens: count of language tokens (with padding) after visual.
          action_horizon: number of action tokens at the end of the suffix.
          language_mask: [B, num_language_tokens] valid-token mask, or None.
          visual_mask: [B, num_visual_tokens] valid-token mask, or None.
            Use this to exclude tokens from padded/black cameras (e.g. right wrist in Libero).

        Returns:
          A dict with the aggregated loss and per-modality / per-pair components.
        """
        per_pair: dict[str, torch.Tensor] = {}
        per_pair_teacher: dict[str, torch.Tensor] = {}
        per_pair_student: dict[str, torch.Tensor] = {}
        per_pair_diag: dict[str, dict] = {}

        for modality in self.modalities:
            for s_layer, t_layer in self.layer_pairs:
                key = self._key(modality, s_layer, t_layer)
                if s_layer not in student_states or t_layer not in teacher_states:
                    continue
                s_pre, s_suf = student_states[s_layer]
                t_pre, t_suf = teacher_states[t_layer]

                if modality == "visual":
                    s_tok = s_pre[:, :num_visual_tokens]
                    t_tok = t_pre[:, :num_visual_tokens]
                    mask = visual_mask
                elif modality == "language":
                    s_tok = s_pre[:, num_visual_tokens : num_visual_tokens + num_language_tokens]
                    t_tok = t_pre[:, num_visual_tokens : num_visual_tokens + num_language_tokens]
                    mask = language_mask
                else:  # action
                    s_tok = s_suf[:, -action_horizon:]
                    t_tok = t_suf[:, -action_horizon:]
                    mask = None

                L_soft, L_t, L_s, pair_diag = self._compute_pair_loss(key, s_tok, t_tok, mask)
                # In soft-only mode, L_t and L_s are zero and we don't divide by 2; the
                # standard 3-loss combination keeps its (L_soft + L_t + L_s) / 2 form.
                L_pair = L_soft if self.soft_loss_only else (L_soft + L_t + L_s) / 2
                per_pair[key] = L_pair
                per_pair_teacher[key] = L_t
                per_pair_student[key] = L_s
                per_pair_diag[key] = pair_diag

        if not per_pair:
            zero = next(self.parameters()).new_zeros(())
            return {
                "loss_concept": zero,
                "per_pair": {},
                "per_pair_teacher": {},
                "per_pair_student": {},
            }

        total = torch.stack(list(per_pair.values())).mean()

        # Aggregate per-pair scalar diagnostics (cheap — entropy ops on existing tensors).
        pair_diag_out: dict = {}
        for key, pd in per_pair_diag.items():
            for metric in ("H_pt", "H_ps", "ENK_t", "ENK_s"):
                pair_diag_out[f"concept_diagnostics/{metric}/{key}"] = pd[metric]

        # Sample distributions from the first active pair (for histogram logging).
        first_key = next(iter(per_pair_diag))
        pd0 = per_pair_diag[first_key]
        hist_out = {
            "concept_hist/pt": pd0["pt_sample"],  # [n, C]
            "concept_hist/qt": pd0["qt_sample"],
            "concept_hist/ps": pd0["ps_sample"],
        }

        return {
            "loss_concept": total,
            "per_pair": per_pair,
            "per_pair_teacher": per_pair_teacher,
            "per_pair_student": per_pair_student,
            **pair_diag_out,
            **hist_out,
        }

"""Probabilistic Knowledge Transfer (PKT) for shallow pi0.5 distillation.

PKT (Passalis & Tefas, ECCV 2018) aligns student and teacher by matching the
pairwise sample-similarity distributions of their hidden representations:

  1) compute cosine similarities within student reps  ->  M_s  [N, N]
  2) compute cosine similarities within teacher reps  ->  M_t  [N, N]
  3) scale to [0, 1] and row-normalize so each row is a distribution
  4) loss = KL(M_t || M_s)

This module exposes two variants:

  - "token":  flatten tokens across the batch (mask-aware), sample K positions
              per modality (same K as concept KD), and treat them as N samples.
  - "global": mean-pool valid tokens to a single vector per (sample, modality),
              and treat the batch as N samples.

In both cases student and teacher dimensions do NOT need to match — the loss
operates on [N, N] similarity matrices, not on the representations directly.

By default PKT runs on the last student/teacher layer; layer pairs are
configurable. Modality selection (visual/language/action) mirrors concept KD.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


_MODALITIES = ("visual", "language", "action")


def pkt_cosine_kl_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """PKT loss: KL between sample-similarity distributions.

    Args:
      student: [N, D_s]
      teacher: [N, D_t]
      eps: numerical floor used in the log and the normalization.

    Both inputs are L2-normalized along the feature axis, pairwise cosines are
    computed within each side, mapped to [0, 1] and row-normalized so every row
    is a distribution over the N "neighbors". Gradient flows only through the
    student term (teacher reps are typically already detached).
    """
    s_norm = torch.sqrt(torch.sum(student * student, dim=1, keepdim=True))
    s = student / (s_norm + eps)
    s = torch.nan_to_num(s, nan=0.0)

    t_norm = torch.sqrt(torch.sum(teacher * teacher, dim=1, keepdim=True))
    t = teacher / (t_norm + eps)
    t = torch.nan_to_num(t, nan=0.0)

    M_s = s @ s.transpose(0, 1)
    M_t = t @ t.transpose(0, 1)

    M_s = (M_s + 1.0) / 2.0
    M_t = (M_t + 1.0) / 2.0

    M_s = M_s / (M_s.sum(dim=1, keepdim=True) + eps)
    M_t = M_t / (M_t.sum(dim=1, keepdim=True) + eps)

    return torch.mean(M_t * (torch.log(M_t + eps) - torch.log(M_s + eps)))


class PKTModule(nn.Module):
    """Probabilistic Knowledge Transfer module (token or global variant)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mode: str = config.pkt_mode
        if self.mode not in ("token", "global"):
            raise ValueError(f"Unknown pkt_mode: {self.mode}. Allowed: 'token', 'global'.")

        self.modalities: tuple[str, ...] = tuple(config.pkt_modalities)
        for m in self.modalities:
            if m not in _MODALITIES:
                raise ValueError(f"Unknown PKT modality: {m}. Allowed: {_MODALITIES}")

        self.layer_pairs: tuple[tuple[int, int], ...] = tuple(
            tuple(p) for p in config.pkt_layer_pairs
        )
        self.sample_ratio: float = float(config.pkt_sample_ratio)
        self.max_tokens: int = int(config.pkt_max_tokens)
        self.eps: float = float(config.pkt_eps)

        # Dummy parameter so `next(self.parameters())` works for device/dtype lookup
        # even when no learnable PKT params exist (PKT has none by design).
        self.register_buffer("_dummy", torch.zeros((), dtype=torch.float32), persistent=False)

    @staticmethod
    def _key(modality: str, s_layer: int, t_layer: int) -> str:
        return f"{modality}_s{s_layer}_t{t_layer}"

    def teacher_layer_indices(self) -> set[int]:
        return {t for _, t in self.layer_pairs}

    def student_layer_indices(self) -> set[int]:
        return {s for s, _ in self.layer_pairs}

    # ------------------------------------------------------------------ helpers
    def _flatten_with_mask(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            B, N, D = tokens.shape
            return tokens.reshape(B * N, D)
        return tokens[mask.bool()]

    def _sample_indices(self, total: int, device) -> torch.Tensor:
        # Same K policy as concept KD: floor(total * ratio), capped at max_tokens, >= 1.
        K = min(int(total * self.sample_ratio), self.max_tokens)
        K = max(K, 1)
        return torch.randperm(total, device=device)[:K]

    def _slice_modality(
        self,
        modality: str,
        s_pre: torch.Tensor,
        s_suf: torch.Tensor,
        t_pre: torch.Tensor,
        t_suf: torch.Tensor,
        num_visual_tokens: int,
        num_language_tokens: int,
        action_horizon: int,
        visual_mask: torch.Tensor | None,
        language_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        return s_tok, t_tok, mask

    # ------------------------------------------------------------------ per-pair
    def _compute_token_loss(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        s_flat = self._flatten_with_mask(student_tokens, mask)
        t_flat = self._flatten_with_mask(teacher_tokens, mask)
        total = s_flat.shape[0]
        if total == 0:
            return self._dummy.new_zeros(())

        idx = self._sample_indices(total, s_flat.device)
        s = s_flat[idx].to(torch.float32)
        t = t_flat[idx].to(torch.float32)
        return pkt_cosine_kl_loss(s, t, eps=self.eps)

    def _compute_global_loss(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean-pool valid tokens per sample (mask-aware), then PKT between samples."""
        s = student_tokens.to(torch.float32)
        t = teacher_tokens.to(torch.float32)
        if mask is not None:
            m = mask.bool().to(s.device).unsqueeze(-1).to(torch.float32)
            denom = m.sum(dim=1).clamp(min=1.0)
            s_pool = (s * m).sum(dim=1) / denom
            t_pool = (t * m).sum(dim=1) / denom
        else:
            s_pool = s.mean(dim=1)
            t_pool = t.mean(dim=1)

        if s_pool.shape[0] < 2:
            # PKT is meaningless on a single sample.
            return self._dummy.new_zeros(())
        return pkt_cosine_kl_loss(s_pool, t_pool, eps=self.eps)

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
        per_pair: dict[str, torch.Tensor] = {}

        for modality in self.modalities:
            for s_layer, t_layer in self.layer_pairs:
                if s_layer not in student_states or t_layer not in teacher_states:
                    continue
                s_pre, s_suf = student_states[s_layer]
                t_pre, t_suf = teacher_states[t_layer]

                s_tok, t_tok, mask = self._slice_modality(
                    modality,
                    s_pre, s_suf, t_pre, t_suf,
                    num_visual_tokens, num_language_tokens, action_horizon,
                    visual_mask, language_mask,
                )

                if self.mode == "token":
                    loss = self._compute_token_loss(s_tok, t_tok, mask)
                else:
                    loss = self._compute_global_loss(s_tok, t_tok, mask)

                per_pair[self._key(modality, s_layer, t_layer)] = loss

        if not per_pair:
            zero = self._dummy.new_zeros(())
            return {"loss_pkt": zero, "per_pair": {}}

        total = torch.stack(list(per_pair.values())).mean()
        return {"loss_pkt": total, "per_pair": per_pair}

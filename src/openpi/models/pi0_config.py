import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    gemma_depth: int = 18

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class DistilledPi0Config(Pi0Config):
    teacher_config: str = "pi0_libero"
    loss_weight_gt: float = 1.0
    loss_weight_teacher: float = 1.0

    # ----- Multimodal Concept KD (off by default; preserves baseline behavior) -----
    use_concept_kd: bool = False

    # Modalities to align with concepts.
    concept_modalities: tuple[str, ...] = ("visual", "language", "action")

    # (student_layer_idx, teacher_layer_idx) pairs at which to align representations.
    # Default mapping for l09 student vs l18 teacher.
    concept_layer_pairs: tuple[tuple[int, int], ...] = (
        (0, 0),
        (4, 8),
        (8, 17),
    )

    # Number of concept vectors per modality.
    concept_num_visual: int = 512
    concept_num_language: int = 128
    concept_num_action: int = 256

    # "cosine" or "l2".
    concept_similarity: str = "cosine"
    concept_temperature: float = 0.1

    # Token subsampling.
    concept_sample_ratio: float = 0.10
    concept_max_tokens: int = 2048

    # Loss combination.
    concept_loss_weight: float = 1.0
    concept_teacher_loss_weight: float = 1.0
    concept_student_loss_weight: float = 1.0

    # Projectors.
    concept_use_student_projector: bool = True
    concept_use_teacher_projector: bool = False
    concept_projector_bias: bool = True

    # K-means initialization for concept vectors.
    concept_init_path: str | None = None
    concept_init_from_kmeans: bool = True
    concept_freeze_prototypes: bool = False

    # Sinkhorn (fixed defaults).
    concept_sinkhorn_eps: float = 0.05
    concept_sinkhorn_iters: int = 3

    # ----- Probabilistic Knowledge Transfer (PKT) -----
    # Independent of concept KD: matches batch-level cosine-similarity distributions
    # between student and teacher hidden reps. Two variants:
    #   - "token":  sample K tokens per (modality, layer pair), like concept KD.
    #   - "global": mean-pool valid tokens per (sample, modality) -> [B, D] reps.
    # Defaults to the deepest student/teacher layer (l09 student vs l18 teacher).
    use_pkt: bool = False
    pkt_mode: str = "token"  # "token" or "global"
    pkt_modalities: tuple[str, ...] = ("visual", "language", "action")
    pkt_layer_pairs: tuple[tuple[int, int], ...] = ((8, 17),)
    pkt_sample_ratio: float = 1.0
    pkt_max_tokens: int = 8192
    pkt_loss_weight: float = 1.0
    pkt_eps: float = 1e-7

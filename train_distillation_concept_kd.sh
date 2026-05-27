#!/usr/bin/env bash
# End-to-end concept-KD distillation: norm stats -> k-means concept init -> train.
# Mirrors train_distillation_baseline.sh but for the concept-KD student config.
set -euo pipefail

export NCCL_NET=Socket
export WANDB_API_KEY="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"

CONFIG=pi05_libero_l09_student_kd_concept_no_sp_td_3l_vla_0.1
EXP_NAME=libero_pi05_l09_student_kd_concept_no_sp_td_3l_vla_0.1
GROUP=libero
KMEANS_PATH=assets/concept_kmeans/${CONFIG}/concepts_3l_vla.pt

# 1) Norm stats (skip if file already present).
uv run scripts/compute_norm_stats.py --config-name ${CONFIG}

# 2) K-means init for the concept banks. Reuses the teacher weights from the config to
#    collect hidden tokens over 50 batches (~12,800 timesteps), then runs spherical
#    k-means with k-means++ init and per-feature standardization to match training-time
#    normalize_mean_std + cosine similarity. Output is loaded by ConceptKDModule when
#    `concept_init_path` is set.
if [[ ! -f "${KMEANS_PATH}" ]]; then
  # batch-size=32 fits comfortably on one 80GB H100 (training uses 256 split across 4
  # GPUs; k-means runs single-process, so we shrink the per-process batch). Bump
  # num-batches proportionally so the per-modality token cap can still be hit
  # (action: 200 * 32 * 10 = 64K available > 50K cap).
  uv run scripts/init_concept_kmeans.py ${CONFIG} \
    --batch-size 32 \
    --num-batches 200 \
    --max-tokens-per-modality 50000 \
    --kmeans-iters 20 \
    --output ${KMEANS_PATH}
else
  echo "Reusing existing k-means init at ${KMEANS_PATH}"
fi

# 3) Distillation training. Override `concept_init_path` so the banks load the k-means
#    centers; everything else (loss weights, layer pairs, etc.) is set on the config.
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/distill_pytorch.py ${CONFIG} \
  --exp_name ${EXP_NAME} \
  --group ${GROUP} \
  --overwrite \
  --model.concept_init_path=${KMEANS_PATH} 2>&1 | tee ${EXP_NAME}.log

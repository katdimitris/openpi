#!/usr/bin/env bash
# Resume concept-KD distillation from the latest valid checkpoint.
# Counterpart to train_distillation_concept_kd.sh — same config/exp_name, but uses
# --resume (NOT --overwrite) so the checkpoint dir, wandb run, and optimizer state
# are all picked up. The previous run crashed at step ~15k while saving (disk OOM),
# so the resume picks up at step 10k (the last fully-written checkpoint) and the
# wandb run continues on the same id.
set -euo pipefail

export NCCL_NET=Socket
export WANDB_API_KEY="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"

CONFIG=pi05_libero_l09_student_kd_concept_td_3l_vla_0.1
EXP_NAME=libero_pi05_l09_student_kd_concept_td_3l_vla_0.1
GROUP=libero
KMEANS_PATH=assets/concept_kmeans/${CONFIG}/concepts_3l_vla.pt
CKPT_DIR=checkpoints/${CONFIG}/${EXP_NAME}

# Sanity: the resume target must exist with all three files. Bail loudly if it doesn't.
for f in model.safetensors optimizer.pt metadata.pt; do
  if [[ ! -f "${CKPT_DIR}/10000/${f}" ]]; then
    echo "ERROR: missing ${CKPT_DIR}/10000/${f} — cannot resume" >&2
    exit 1
  fi
done

# Clean up any partially-written tmp checkpoint from the previous crash. load_checkpoint
# already ignores tmp_* dirs (they fail the .isdigit() check), but they waste disk.
find "${CKPT_DIR}" -maxdepth 1 -type d -name "tmp_*" -print -exec rm -rf {} + || true

# Each checkpoint is ~14 GB. Free space should comfortably cover the remaining saves
# (steps 15k/20k/25k/30k = ~56 GB). Print and warn at <70 GB.
FREE_GB=$(df -BG --output=avail /home/jupyter | tail -1 | tr -dc '0-9')
echo "Free disk on /home/jupyter: ${FREE_GB} GB"
if [[ ${FREE_GB} -lt 70 ]]; then
  echo "WARNING: less than 70GB free; consider deleting checkpoints/${CONFIG}/${EXP_NAME}/5000 (~14GB) before continuing."
fi

# Resume training.
#  --resume: load latest valid checkpoint (10000), restore model + optimizer + global_step.
#  LR schedule is a pure function of global_step (see distill_pytorch.py:530), so no
#  scheduler state needs restoring — it just plugs 10000 in and gets the right value.
#  --model.concept_init_path is harmless on resume (the safetensors load overwrites the
#  bank), but we pass it so the run is reproducible from scratch too.
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/distill_pytorch.py ${CONFIG} \
  --exp_name ${EXP_NAME} \
  --group ${GROUP} \
  --resume \
  --model.concept_init_path=${KMEANS_PATH} 2>&1 | tee ${EXP_NAME}_resume.log

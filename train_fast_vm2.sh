#!/bin/bash
# VM 2: concept KD ablations (vla / v / l / a)

set -e

export NCCL_NET=Socket
export WANDB_API_KEY="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"

CKPT_DIR="./fast_checkpoints"
NGPU=4
GROUP="libero_fast"

# Norm stats are identical across all configs (same data). Copy from existing run.
NORM_SRC="./assets/pi05_libero_l09_student/physical-intelligence/libero"
for config in \
    pi05_libero_l06_fast_student_kd_concept_vla_3l_0.1 \
    pi05_libero_l06_fast_student_kd_concept_v_3l_0.1 \
    pi05_libero_l06_fast_student_kd_concept_l_3l_0.1 \
    pi05_libero_l06_fast_student_kd_concept_a_3l_0.1; do
    dst="./assets/${config}/physical-intelligence/libero"
    mkdir -p "${dst}"
    cp "${NORM_SRC}/norm_stats.json" "${dst}/norm_stats.json"
done
echo "Norm stats copied."

run_exp() {
    local config_name="$1"
    local exp_name="$2"
    echo "========================================"
    echo "Starting: $config_name  (exp_name=$exp_name)"
    echo "========================================"
    uv run torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} \
        scripts/distill_pytorch.py "${config_name}" \
        --exp_name "${exp_name}" \
        --group "${GROUP}" \
        --checkpoint_base_dir "${CKPT_DIR}" \
        --overwrite \
        2>&1 | tee "${exp_name}.log"
    echo "Finished: $config_name"
    echo ""
}

run_exp pi05_libero_l06_fast_student_kd_concept_vla_3l_0.1  libero_l06_fast_concept_vla
run_exp pi05_libero_l06_fast_student_kd_concept_v_3l_0.1    libero_l06_fast_concept_v
run_exp pi05_libero_l06_fast_student_kd_concept_l_3l_0.1    libero_l06_fast_concept_l
run_exp pi05_libero_l06_fast_student_kd_concept_a_3l_0.1    libero_l06_fast_concept_a

echo "VM2 experiments complete."

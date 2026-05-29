#!/bin/bash
# VM 2: kmeans_fixed concept KD — single-layer-pair (last layer only), T=1.0.

set -e
set -o pipefail   # so failures in `<cmd> | tee ...` abort the script

export NCCL_NET=Socket
export WANDB_API_KEY="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"

CKPT_DIR="./fast_checkpoints"
NGPU=4
GROUP="libero_fast"

# Norm stats are identical across all configs (same data). Copy from existing run.
NORM_SRC="./assets/pi05_libero_l09_student/physical-intelligence/libero"
for config in \
    pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_vla_1l_T1_0.1 \
    pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_v_1l_T1_0.1 \
    pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_l_1l_T1_0.1 \
    pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_a_1l_T1_0.1; do
    dst="./assets/${config}/physical-intelligence/libero"
    mkdir -p "${dst}"
    cp "${NORM_SRC}/norm_stats.json" "${dst}/norm_stats.json"
done
echo "Norm stats copied."

# K-means initialization for the concept bank. Centroids do not depend on temperature
# (k-means is on raw teacher activations), so this produces the same numerical centers
# as VM1; the file just lives under this VM's vla_1l_T1_0.1 config name to keep each
# VM self-contained. All four T=1.0 variants on this VM read it via concept_init_path.
KMEANS_FILE="./assets/concept_kmeans/pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_vla_1l_T1_0.1/concepts.pt"
if [ ! -f "${KMEANS_FILE}" ]; then
    echo "Running k-means init -> ${KMEANS_FILE}"
    uv run scripts/init_concept_kmeans.py \
        pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_vla_1l_T1_0.1 \
        --batch-size 32 \
        --num-batches 50 \
        2>&1 | tee init_concept_kmeans.log
else
    echo "K-means file already present at ${KMEANS_FILE}; skipping init."
fi

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

run_exp pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_vla_1l_T1_0.1  libero_l06_fast_concept_kmeans_fixed_vla_1l_T1
run_exp pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_v_1l_T1_0.1    libero_l06_fast_concept_kmeans_fixed_v_1l_T1
run_exp pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_l_1l_T1_0.1    libero_l06_fast_concept_kmeans_fixed_l_1l_T1
run_exp pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_a_1l_T1_0.1    libero_l06_fast_concept_kmeans_fixed_a_1l_T1

echo "VM2 experiments complete."

#!/bin/bash
# VM 2: kmeans_fixed concept KD ablations — WITH student projector (vla / v / l / a).

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
    pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_proj_vla_3l_0.1 \
    pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_proj_v_3l_0.1 \
    pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_proj_l_3l_0.1 \
    pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_proj_a_3l_0.1; do
    dst="./assets/${config}/physical-intelligence/libero"
    mkdir -p "${dst}"
    cp "${NORM_SRC}/norm_stats.json" "${dst}/norm_stats.json"
done
echo "Norm stats copied."

# K-means initialization for the concept bank. Proj and no-proj variants share the
# same kmeans file (it is built from teacher activations only — the student-side
# projector does not affect what gets clustered). One run on the VLA superset config
# produces all (visual/language/action) centroids that every kmeans_fixed_* config
# loads via concept_init_path.
KMEANS_FILE="./assets/concept_kmeans/pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_vla_3l_0.1/concepts.pt"
if [ ! -f "${KMEANS_FILE}" ]; then
    echo "Running k-means init -> ${KMEANS_FILE}"
    # Run init from the *proj* config (whose norm_stats are already copied above) but
    # write to the shared KMEANS_FILE path that every config — proj and no-proj alike —
    # points at via concept_init_path. Centroids are identical either way: kmeans
    # depends only on teacher activations, which the student-side projector flag does
    # not affect.
    uv run scripts/init_concept_kmeans.py \
        pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_proj_vla_3l_0.1 \
        --output "${KMEANS_FILE}" \
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

run_exp pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_proj_vla_3l_0.1  libero_l06_fast_concept_kmeans_fixed_proj_vla
run_exp pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_proj_v_3l_0.1    libero_l06_fast_concept_kmeans_fixed_proj_v
run_exp pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_proj_l_3l_0.1    libero_l06_fast_concept_kmeans_fixed_proj_l
run_exp pi05_libero_l06_fast_student_kd_concept_kmeans_fixed_proj_a_3l_0.1    libero_l06_fast_concept_kmeans_fixed_proj_a

echo "VM2 experiments complete."

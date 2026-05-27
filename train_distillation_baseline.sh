export NCCL_NET=Socket
export WANDB_API_KEY="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"

# uv run scripts/compute_norm_stats.py --config-name pi05_libero_l09_student

uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/distill_pytorch.py pi05_libero_l09_student \
  --exp_name libero_pi05_l09_student \
  --group libero \
  --overwrite 2>&1 | tee libero_pi05_l09_student.log

uv run scripts/compute_norm_stats.py --config-name pi05_libero_l09_student_kd

uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/distill_pytorch.py pi05_libero_l09_student_kd \
  --exp_name libero_pi05_l09_student_kd \
  --group libero \
  --overwrite
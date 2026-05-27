#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
# delete cache from docker: docker system prune -a --volumes
export WANDB_API_KEY="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"
export WANDB_DISABLE_SERVICE=true

export TASK_SUITE="libero_10"

export EXP_NAME="teacher_pi05"

# local pretrained teacher checkpoint
export CHECKPOINT_PATH="./checkpoints_converted/pi05_libero_torch"

# IMPORTANT:
# use the ORIGINAL teacher config, not pi05_libero_l06
export SERVER_ARGS="policy:checkpoint --policy.config pi05_libero --policy.dir $CHECKPOINT_PATH"

export CLIENT_ARGS="--args.task-suite-name $TASK_SUITE --args.save-name $EXP_NAME"

echo "--- Starting Evaluation ---"
echo "Model: $EXP_NAME"
echo "Task Suite: $TASK_SUITE"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "GPU: $CUDA_VISIBLE_DEVICES"

docker compose -f examples/libero/compose.yml down

docker compose -f examples/libero/compose.yml up
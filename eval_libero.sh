#!/bin/bash

# delete cache from docker: docker system prune -a --volumes
export WANDB_API_KEY="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"
export WANDB_DISABLE_SERVICE=true

export TASK_SUITE="libero_10"

export EXP_NAME="pi05_distill_baseline"
export CHECKPOINT_PATH="./checkpoints/pi05_libero_l06/$EXP_NAME/30000"

export SERVER_ARGS="policy:checkpoint --policy.config pi05_libero_l06 --policy.dir $CHECKPOINT_PATH"

export CLIENT_ARGS="--args.task-suite-name $TASK_SUITE --args.save-name $EXP_NAME"

echo "--- Starting Evaluation ---"
echo "Model: $EXP_NAME"
echo "Task Suite: $TASK_SUITE"

docker compose -f examples/libero/compose.yml down

docker compose -f examples/libero/compose.yml up


from pathlib import Path

p = Path("src/openpi/training/config.py")
s = p.read_text()

marker = '    TrainConfig(\n        name="pi05_libero",'

new_config = '''
    TrainConfig(
        name="pi05_libero_l06",
        wandb_enabled=False,
        model=pi0_config.DistilledPi0Config(
            teacher_config="pi05_libero",
            gemma_depth=6,
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        pytorch_weight_path="checkpoints_converted/pi05_libero_torch",
        pytorch_weight_path_teacher=None,
        num_train_steps=30_000,
        batch_size=64,
        save_interval=10_000,
        pytorch_training_precision="bfloat16",
    ),
'''

if 'name="pi05_libero_l06"' not in s:
    s = s.replace(marker, new_config + "\\n" + marker)

p.write_text(s)
print("Added pi05_libero_l06 config")

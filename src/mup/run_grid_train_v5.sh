#!/bin/bash

# Array of different learning rates
muP_learning_rates=(0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625)
# Array of different model widths
widths=(256 512 1024 2048)

# Project and prefix settings
wandb_project="mup-transformer-training"
wandb_run_prefix="transformer_traintest_fix"

# Loop over the learning rates
for lr in "${muP_learning_rates[@]}"; do
  # Loop over the widths
  for width in "${widths[@]}"; do
    # Run the Python script with the current configuration
    python train.py --lr $lr --width $width --wandb_project $wandb_project --wandb_run_prefix $wandb_run_prefix --apply_muP --lr_scheduler cosine --fix_layernorm --fix_embed_lr --fix_unembed --fix_weight_decay
  done
done

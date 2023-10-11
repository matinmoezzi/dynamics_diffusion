#! /bin/bash

cd ~/matin/dynamics_diffusion
conda activate dynamics_diffusion

python scripts/dynamics_train.py trainer.total_training_steps=42000
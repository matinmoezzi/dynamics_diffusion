#! /bin/bash

~/miniforge3/bin/conda init
source ~/.bashrc

cd ~/matin/dynamics_diffusion
conda activate dynamics_diffusion

python scripts/dynamics_train.py trainer.total_training_steps=42000
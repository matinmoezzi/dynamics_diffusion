#! /bin/bash

python scripts/train_diffusion.py trainer=sde dataset@trainer.dataset=kitchen-mixed model@trainer.model=MLP diffusion@trainer.diffusion=VESDE trainer.total_training_steps=5e5
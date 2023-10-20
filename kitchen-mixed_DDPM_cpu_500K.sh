#! /bin/bash

python scripts/train_diffusion.py trainer=ddpm dataset@trainer.dataset=kitchen-mixed model@trainer.model=MLP trainer.total_training_steps=5e5
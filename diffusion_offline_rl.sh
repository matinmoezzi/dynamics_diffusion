#! /bin/bash
python scripts/train_diffusion.py device=cuda:0 dataset@trainer.dataset=maze2d-umaze trainer.total_training_steps=1_000_000
python scripts/train_diffusion.py device=cuda:0 dataset@trainer.dataset=maze2d-large trainer.total_training_steps=1_000_000

python scripts/train_diffusion.py device=cuda:1 dataset@trainer.dataset=ant-medium trainer.total_training_steps=5_000_000
python scripts/train_diffusion.py device=cuda:1 dataset@trainer.dataset=hopper-medium trainer.total_training_steps=5_000_000
python scripts/train_diffusion.py device=cuda:2 dataset@trainer.dataset=halfcheetah-medium trainer.total_training_steps=5_000_000
python scripts/train_diffusion.py device=cuda:2 dataset@trainer.dataset=walker-medium trainer.total_training_steps=5_000_000

python scripts/train_diffusion.py device=cuda:3 dataset@trainer.dataset=kitchen-mixed trainer.total_training_steps=10_000_000
python scripts/train_diffusion.py device=cuda:3 dataset@trainer.dataset=kitchen-complete trainer.total_training_steps=10_000_000
python scripts/train_diffusion.py device=cuda:4 dataset@trainer.dataset=door-expert trainer.total_training_steps=10_000_000


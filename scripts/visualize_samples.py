import os
import pathlib
import gym
import numpy as np
import argparse
from dynamics_diffusion import env
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(prog="visualize_samples")
    parser.add_argument(
        "-i",
        "--samples_dir",
        action="store",
        type=str,
        required=True,
        help="Samples Path in .npz",
    )
    args = parser.parse_args()

    assert pathlib.Path(
        args.samples_dir, ".hydra"
    ).is_dir(), "Hydra configuration not found."

    cfg = OmegaConf.load(pathlib.Path(args.samples_dir, ".hydra", "config.yaml"))
    train_cfg = OmegaConf.load(pathlib.Path(cfg.model_dir, ".hydra", "config.yaml"))

    list_samples = list(pathlib.Path(args.samples_dir).glob("*.npz"))
    assert list_samples, f"No samples (.npz) found."

    samples_path = max(list_samples, key=os.path.getctime)

    samples = np.load(samples_path)
    states = samples["states"]
    true_next_states = samples["true_next_states"]
    sampled_next_states = samples["sampled_next_states"]

    print(f"{len(states)} samples loaded from {samples_path}.")

    mse = ((sampled_next_states - true_next_states) ** 2).mean(axis=0)
    print(f"MSE loss: {mse}")

    env = gym.make("eval-" + train_cfg.env.name)
    env.reset()

    qpos = np.concatenate((states[0][:2], states[0][:2]))
    qvel = np.concatenate((states[0][2:], states[0][2:]))
    env.set_state(qpos, qvel)
    while True:
        for i in range(len(true_next_states)):
            qpos = np.concatenate((true_next_states[i][:2], sampled_next_states[i][:2]))
            qvel = np.concatenate((true_next_states[i][2:], sampled_next_states[i][2:]))
            env.set_state(qpos, qvel)
            env.render()


if __name__ == "__main__":
    main()

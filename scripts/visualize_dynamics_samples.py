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
        "-f",
        "--samples_path",
        action="store",
        type=str,
        required=True,
        help="Samples Path in .npz",
    )
    parser.add_argument(
        "-e",
        "--env",
        default="maze2d-umaze-v1",
        required=True,
        type=str,
        help="Environment name",
    )
    args = parser.parse_args()

    assert args.samples_path.endswith(".npz")

    samples = np.load(args.samples_path)

    assert (
        "states" in samples
        and "true_next_states" in samples
        and "sampled_next_states" in samples
    ), "Invalid samples file."

    states = samples["states"]
    true_next_states = samples["true_next_states"]
    sampled_next_states = samples["sampled_next_states"]

    print(f"{len(states)} samples loaded from {args.samples_path}.")

    mse = ((sampled_next_states - true_next_states) ** 2).mean(axis=0)
    print(f"MSE loss: {mse}")

    env = gym.make("eval-" + args.env)
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

import pathlib
import gym
import numpy as np
import argparse
from dynamics_diffusion import env


def main():
    parser = argparse.ArgumentParser(prog="visualize_samples")
    parser.add_argument(
        "-i",
        "--samples_path",
        action="store",
        type=str,
        required=True,
        help="Samples Path in .npz",
    )
    args = parser.parse_args()

    samples = np.load(args.samples_path)
    states = samples["states"]
    actions = samples["actions"]
    true_next_states = samples["true_next_states"]
    sampled_next_states = samples["sampled_next_states"]

    env_name = pathlib.Path(args.samples_path).parent.name.split("_")[0]
    env = gym.make("eval-" + env_name)
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

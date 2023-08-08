from .maze_model import (
    MazeEnv,
    U_MAZE_EVAL,
    MEDIUM_MAZE_EVAL,
    LARGE_MAZE_EVAL,
)
from gym.envs.registration import register

register(
    id="eval-maze2d-umaze-v1",
    entry_point="dynamics_diffusion.env:MazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_spec": U_MAZE_EVAL,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 36.63,
        "ref_max_score": 141.4,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/eval-maze2d-umaze-sparse-v1.hdf5",
    },
)

register(
    id="eval-maze2d-medium-v1",
    entry_point="dynamics_diffusion.env:MazeEnv",
    max_episode_steps=600,
    kwargs={
        "maze_spec": MEDIUM_MAZE_EVAL,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 13.07,
        "ref_max_score": 204.93,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/eval-maze2d-medium-sparse-v1.hdf5",
    },
)


register(
    id="eval-maze2d-large-v1",
    entry_point="dynamics_diffusion.env:MazeEnv",
    max_episode_steps=800,
    kwargs={
        "maze_spec": LARGE_MAZE_EVAL,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 16.4,
        "ref_max_score": 302.22,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/eval-maze2d-large-sparse-v1.hdf5",
    },
)

register(
    id="eval-maze2d-umaze-dense-v1",
    entry_point="dynamics_diffusion.env:MazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_spec": U_MAZE_EVAL,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 56.95455,
        "ref_max_score": 178.21373133248397,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/eval-maze2d-umaze-dense-v1.hdf5",
    },
)

register(
    id="eval-maze2d-medium-dense-v1",
    entry_point="dynamics_diffusion.env:MazeEnv",
    max_episode_steps=600,
    kwargs={
        "maze_spec": MEDIUM_MAZE_EVAL,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 42.28578,
        "ref_max_score": 235.5658957482388,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/eval-maze2d-medium-dense-v1.hdf5",
    },
)


register(
    id="eval-maze2d-large-dense-v1",
    entry_point="dynamics_diffusion.env:MazeEnv",
    max_episode_steps=800,
    kwargs={
        "maze_spec": LARGE_MAZE_EVAL,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 56.95455,
        "ref_max_score": 326.09647655082637,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/eval-maze2d-large-dense-v1.hdf5",
    },
)

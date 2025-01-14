from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
import os

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl
    import gym


def get_env(env_name) -> gym.Env:
    env = gym.make(env_name)
    env.reset()
    return env


def get_d4rl_dataset(name, batch_size, deterministic=False, reward_tune=None):
    env = get_env(name)
    data = d4rl.qlearning_dataset(env)
    return d4rl_load_data(
        data=data,
        batch_size=batch_size,
        deterministic=deterministic,
        reward_tune=reward_tune,
    ), {
        "state_dim": data["observations"].shape[1],
        "action_dim": data["actions"].shape[1],
        "size": data["observations"].shape[0],
    }


def d4rl_load_data(
    *, name, batch_size, deterministic=False, reward_tune=None, num_workers=1
):
    """
    Create a generator over (states, actions, next_states, rewards, dones) pairs given a d4rl environment.

    :param data: D4RL data.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    :param reward_tune: reward tuning type
    """
    env = get_env(name)
    data = d4rl.qlearning_dataset(env)
    dataset = D4RLDataset(
        data,
        reward_tune,
    )
    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=DistributedSampler(dataset, drop_last=True),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=DistributedSampler(dataset, drop_last=True),
            pin_memory=True,
        )

    while True:
        yield from loader


class D4RLDataset(Dataset):
    def __init__(self, data, reward_tune):
        self.data = data

        self.state = torch.from_numpy(self.data["observations"]).float()
        self.action = torch.from_numpy(self.data["actions"]).float()
        self.next_state = torch.from_numpy(self.data["next_observations"]).float()
        reward = torch.from_numpy(self.data["rewards"]).view(-1, 1).float()
        self.not_done = (
            1.0 - torch.from_numpy(self.data["terminals"]).view(-1, 1).float()
        )

        self.len = self.state.shape[0]

        if reward_tune == "normalize":
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            reward = reward - 1.0
        elif reward_tune == "iql_locomotion":
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == "cql_antmaze":
            reward = (reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            reward = (reward - 0.25) * 2.0

        self.reward = reward

    def __getitem__(self, idx):
        ind = idx % self.len
        return self.next_state[ind], {
            "state": self.state[ind],
            "action": self.action[ind],
        }

    def __len__(self):
        return self.len


def iql_normalize(reward, not_done):
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(
        torch.tensor(trajs_rt)
    )
    reward /= rt_max - rt_min
    reward *= 1000.0
    return reward

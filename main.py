import copy
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(".").absolute().parent))
import gym
import numpy as np
import d4rl
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from diffusion import Diffusion
from model import MLP

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


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


class D4RLDataset(Dataset):
    def __init__(self, data, reward_tune=None, device="cpu"):
        self.data = data

        self.state = torch.from_numpy(self.data["observations"]).float()
        self.action = torch.from_numpy(self.data["actions"]).float()
        self.next_state = torch.from_numpy(self.data["next_observations"]).float()
        reward = torch.from_numpy(self.data["rewards"]).view(-1, 1).float()
        self.not_done = (
            1.0 - torch.from_numpy(self.data["terminals"]).view(-1, 1).float()
        )

        self.len = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

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
        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device),
        )

    def __len__(self):
        return self.len


def split_dict(dict, n):
    keys = list(dict.keys())
    train = {k: dict[k][:n] for k in keys}
    test = {k: dict[k][n:] for k in keys}
    return train, test


def main():
    update_ema_every = 10
    step_start_ema = 1000
    ema_decay = 0.995
    ema_step = 0 

    device = "cuda"
    # Initialize offline RL dataset
    env_name = "maze2d-umaze-v1"
    env = gym.make(env_name)
    env.reset()
    data = d4rl.qlearning_dataset(env)

    data_size = data["observations"].shape[0]
    state_dim = data["observations"].shape[1]
    action_dim = data["actions"].shape[1]

    train_data, test_data = split_dict(data, int(0.95 * data_size))

    train_dataset = D4RLDataset(train_data, device=device)
    test_dataset = D4RLDataset(test_data, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    model = MLP(x_dim=state_dim, cond_dim=action_dim + state_dim, device=device)

    diffusion_model = Diffusion(
        x_dim=state_dim,
        cond_dim=action_dim + state_dim,
        model=model,
        device=device,
        clip_denoised=False,
    ).to(device)

    ema = EMA(ema_decay)
    ema_model = copy.deepcopy(diffusion_model)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=3e-4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/diffusion_{env_name}_{timestamp}")

    EPOCHS = 1


    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        diffusion_model.train(True)

        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(tqdm(train_dataloader)):
            # Every data instance is an input + label pair
            states, actions, next_states, *_ = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            cond = torch.cat((actions, states), dim=1)
            # outputs = diffusion_model(cond, verbose=False, return_diffusion=False)

            # Compute the loss and its gradients
            loss = diffusion_model.loss(next_states, cond)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            if ema_step % update_ema_every == 0:
                if ema_step >= step_start_ema:
                    ema.update_model_average(ema_model, diffusion_model)

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch * len(train_dataloader) + i + 1
                writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0
            
            ema_step += 1

    torch.save({"model": diffusion_model, "ema": ema_model}, f"model_{timestamp}")

    torch.cuda.empty_cache()

    val_loss = torch.nn.MSELoss()
    ema_loss, model_loss = 0, 0
    for i, vdata in enumerate(tqdm(test_dataloader)):
        states, actions, next_states, *_ = vdata 
        cond = torch.cat((actions, states), dim=1)
        model_out = diffusion_model(cond)
        ema_out = ema_model(cond)
        model_loss += val_loss(next_states, model_out)
        ema_loss += val_loss(next_states, ema_out)

    ema_loss /= (i + 1)
    model_loss /= (i + 1)
    print(f"Avg. Model Test Loss: {model_loss}\nAvg. EMA Test Loss: {ema_loss}")

if __name__ == "__main__":
    main()

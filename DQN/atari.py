import os
import random
import time
import json

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer

class Parameters:
    PATH = os.path.join(os.getcwd(), "DQN", 'parameter.json')

    def __init__(self):
        with open(self.PATH, "r") as f:
            self.__dict__ = json.loads(f.read())

    def save(self):
        json_data = json.dumps(self, indent=4, default=lambda o: o.__dict__)
        with open(self.PATH, "w") as f:
            f.write(json_data)

def make_env(env_id, seed, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class QNetwork(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n)
        )

    def forward(self, x):
        return self.network(x / 255.0)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


param = Parameters()

writer = SummaryWriter(f"runs/{param.run_name}")

# seeding
random.seed(param.seed)
np.random.seed(param.seed)
torch.manual_seed(param.seed)
torch.backends.cudnn.deterministic = True

# check device if have cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create vectorized environment
envs = gym.vector.SyncVectorEnv([make_env(param.env_id, param.seed, param.run_name)])
assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

# build model
q_network = QNetwork(envs).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=param.learning_rate)
target_network = QNetwork(envs).to(device)
target_network.load_state_dict(q_network.state_dict())

rb = ReplayBuffer(
    param.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    optimize_memory_usage=True,
    handle_timeout_termination=True,
)
start_time = time.time()

# START GAME
observation = envs.reset()

for global_step in range(param.total_timesteps):
    epsilon = linear_schedule(param.start_e, param.end_e, param.exploration_fraction * param.total_timesteps, global_step)
    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        q_values = q_network(torch.Tensor(observation).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()

    next_observation, rewards, dones, infos = envs.step(actions)

    for info in infos:
        if "episode" in info.keys():
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            break

    real_next_observation = next_observation.copy()
    for idx, d in enumerate(dones):
        if d:
            real_next_observation[idx] = infos[idx]["terminal_observation"]

    rb.add(observation, real_next_observation, actions, rewards, dones, infos)

    observation = next_observation

    # TRAINING
    if global_step > param.learning_starts:
        if global_step % param.train_frequency == 0:
            data = rb.sample(param.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + param.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_step % param.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

envs.close()
writer.close()

torch.save(q_network.state_dict(), "dqn-atari-breakout-3.pth")
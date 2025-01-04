"""
DQN implementation for the Boxing Atari environment
"""
import datetime
import math
import random
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv,
)
from helper_cnn import DQN, AdvancedReplayMemory

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = datetime.datetime.now().strftime("%d-%H%M%S") + "-boxing"

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

LR = 1e-4
BATCH_SIZE = 32
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 310000
LEARNING_STEPS = 50000000
GAMMA = 0.99
LEARNING_START = 80000
TAU = 0.01


def optimize_model(memory: AdvancedReplayMemory, steps: int):
    data = memory.sample(BATCH_SIZE)
    with torch.no_grad():
        target_max, _ = target_net(data.next_states).max(dim=1)
        td_target = data.rewards.flatten() + GAMMA * target_max

    state_action_values = policy_net(
        data.states).gather(1, data.actions).squeeze()
    loss = F.mse_loss(td_target, state_action_values)
    if steps % 200 == 0:
        writer.add_scalar("losses/td_loss", loss, steps)
        writer.add_scalar("losses/q_values",
                          state_action_values.mean().item(), steps)
        print("SPS:", int(steps / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(steps / (time.time() - start_time)), steps)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def select_action(state: np.array, steps_done: int):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return torch.argmax(policy_net(torch.Tensor(state).to(device)), dim=1).cpu().numpy().item()
    else:
        return env.action_space.sample()


def make_env(env_id: str, run_name: str):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(2)
        return env
    return thunk


env = make_env("ALE/Boxing-v5", run_name)()

n_actions = env.action_space.n
policy_net = DQN(n_actions).to(device)  # Q(s, a)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

target_net = DQN(n_actions).to(device)  # Q(s', a)
target_net.load_state_dict(policy_net.state_dict())

memory = AdvancedReplayMemory(1000000, env.observation_space, device)

writer = SummaryWriter(f"runs/{run_name}")
start_time = time.time()
obs, _ = env.reset(seed=1)
endgame_reward = 0
for steps in range(LEARNING_STEPS):
    action = select_action(obs, steps)
    next_obs, reward, termination, truncation, infos = env.step(action)
    endgame_reward += reward
    real_next_obs = next_obs.copy()
    memory.add(obs, real_next_obs, action, reward)
    obs = next_obs

    if termination or truncation:
        next_obs, _ = env.reset()
        writer.add_scalar("charts/Endgame reward", endgame_reward, steps)
        endgame_reward = 0
        continue

    if steps > LEARNING_START:
        optimize_model(memory, steps)

        for target_network_param, q_network_param in zip(target_net.parameters(), policy_net.parameters()):
            target_network_param.data.copy_(
                TAU * q_network_param.data +
                (1.0 - TAU) * target_network_param.data
            )

        if steps % 10000 == 0:
            torch.save(target_net.state_dict(), f"models/{run_name}.pt")

env.close()
writer.close()

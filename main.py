import gym
import itertools
import math
import matplotlib
import numpy as np
import pandas as pd
import random
import sys

from gym.envs.registration import register
from matplotlib import pyplot as plt


def greedy(Q, action_space, state):
    """Choose an action greedily."""
    return np.argmax(Q[state])

def epsilon_greedy(Q, action_space, state, epsilon):
    """Choose an action epsilon-greedily."""
    if random.random() < epsilon:
        return action_space.sample()
    else:
        return np.argmax(Q[state])

def log_progress(episode, every=100):
    """Print progress in console during training."""
    if (episode + 1) % every == 0:
        print(f'\rEpisode {episode + 1}/{num_episodes}.', end="")
        sys.stdout.flush()

    if episode + 1 == num_episodes:
        print()


# hyperparameters
num_episodes = 1000000
gamma = 0.9  # discount factor
alpha = 0.5  # learning rate
epsilon_i = 1  # initial exploration rate
epsilon_f = 0.05  # final exploration rate
epsilon_n = 1000000  # number of steps to decay exploration rate

# create environment
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)
env = gym.make('FrozenLake-v0')

# initialize Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# initialize stats
episode_rewards =  np.zeros(num_episodes)
episode_lengths = np.zeros(num_episodes)
step_count = 0

# train
for episode in range(num_episodes):
    state = env.reset()

    for t in itertools.count():
        # compute exploration rate
        if step_count < epsilon_n:
            epsilon = epsilon_i - step_count * (epsilon_i - epsilon_f) / epsilon_n
        else:
            epsilon = epsilon_f

        # step in environment
        action = epsilon_greedy(Q, env.action_space, state, epsilon)
        next_state, reward, done, _ = env.step(action)
        step_count += 1

        # TD update
        target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (target - Q[state, action])

        # update stats
        episode_rewards[episode] += gamma ** t * reward
        episode_lengths[episode] += 1

        if done:
            break

        state = next_state

    log_progress(episode)
env.close()
print(Q)

# visualize results
figure_width = 10
figure_height = 5
smooth_win_size = 100
max_reward = gamma ** 5

fig1 = plt.figure(figsize=(figure_width, figure_height))
rolling_lengths = pd.DataFrame(episode_lengths).rolling(smooth_win_size, min_periods=smooth_win_size)
mean_lengths = rolling_lengths.mean()
std_lengths = rolling_lengths.std()
plt.plot(mean_lengths, linewidth=2)
plt.fill_between(x=std_lengths.index, y1=(mean_lengths - 2 * std_lengths)[0],
                 y2=(mean_lengths + 2 * std_lengths)[0], color='b', alpha=.1)
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.title(f'Episode Length over Time (Smoothed over window size {smooth_win_size})')
plt.show(fig1)

fig2 = plt.figure(figsize=(figure_width, figure_height))
rolling_rewards = pd.DataFrame(episode_rewards).rolling(smooth_win_size, min_periods=smooth_win_size)
mean_rewards = rolling_rewards.mean()
std_rewards = rolling_rewards.std()
plt.plot(mean_rewards, linewidth=2)
plt.fill_between(x=std_rewards.index, y1=(mean_rewards - 2 * std_rewards)[0],
                 y2=(mean_rewards + 2 * std_rewards)[0], color='b', alpha=.1)
plt.hlines(max_reward, 0, num_episodes, colors='r', linestyles='solid',
           label='max reward')
plt.xlabel('Episode')
plt.ylabel('Episode Reward (Smoothed)')
plt.title(f'Episode Reward over Time (Smoothed over window size {smooth_win_size})')
plt.show(fig2)

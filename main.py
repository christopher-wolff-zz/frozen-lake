import gym
import itertools
import math
import matplotlib
import numpy as np
import random
import sys

from gym.envs.registration import register
from gym import wrappers
from time import sleep


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
num_episodes = 10000
alpha = 0.5  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2  # exploration rate

# init
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)
env = gym.make('FrozenLakeNotSlippery-v0')
Q = np.zeros((env.observation_space.n, env.action_space.n))
stats = {
    'episode_rewards': np.zeros(num_episodes),
    'episode_lengths': np.zeros(num_episodes)
}

# train
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # step in environment
        action = epsilon_greedy(Q, env.action_space, state, epsilon)
        next_state, reward, done, _ = env.step(action)

        # TD update
        target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (target - Q[state, action])

        # update stats
        stats['episode_rewards'][episode] += reward
        stats['episode_lengths'][episode] += 1

        state = next_state

    log_progress(episode)

env.close()
print(Q)

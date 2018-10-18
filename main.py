import gym
import math
import numpy as np
import random

from os import system
from time import sleep

env = gym.make('FrozenLake-v0')
env.reset()

# hyperparameters
num_episodes = 10000
alpha = 0.9
epsilon = 0.05
gamma = 0.99

Q = np.zeros((env.observation_space.n, env.action_space.n))
rewards = []
for episode in range(num_episodes):
    print(f'Episode {episode}')
    # reset environment
    state = env.reset()
    done = False
    total_reward = 0
    i = 0
    while not done:
        i += 1
        # select action epsilon-greedily
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        # execute action
        state_new, reward, done, info = env.step(action)
        if done and reward == 0:
            reward = -1
        # update Q-value
        Q[state, action] += alpha * (reward + gamma * np.max(Q[state_new]) - Q[state, action])
        total_reward += reward
        state = state_new
    print(f'mean reward: {total_reward / i}')

print(Q)

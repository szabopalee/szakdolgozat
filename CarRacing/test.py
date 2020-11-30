import numpy as np
import gym
from gym.spaces import Box, Discrete

from skimage.color import rgb2gray

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import matplotlib.pyplot as plt

def imshow(img):
    plt.imshow(img)
    plt.show()

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class GaussianActor(Actor):

    def __init__(self, act_dim):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
        self.mu_net = nn.Sequential(
            nn.Conv2d(1,6,4,2),
            nn.ReLU(),
            nn.Conv2d(6,16,4,2),
            nn.ReLU(),
            nn.Conv2d(16,32,4,2),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, act_dim)
        )

    def _distribution(self, obs):
        mu = self.mu_net(obs)           # mean
        std = torch.exp(self.log_std)   # standard deviation
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.v_net = nn.Sequential(
            nn.Conv2d(1,6,4,2),
            nn.ReLU(),
            nn.Conv2d(6,16,4,2),
            nn.ReLU(),
            nn.Conv2d(16,32,4,2),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, obs):
       return torch.squeeze(self.v_net(obs), -1)

class ActorCritic(nn.Module):
    def __init__(self, action_space):
        super().__init__()

        self.pi = GaussianActor(action_space.shape[0])

        self.value  = Critic()

    def step(self, obs):
        obs = torch.unsqueeze(obs, 0)                  # obs megfelelo formara hozasa
        obs = torch.unsqueeze(obs, 0).view(1,1,96,96)
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            action = pi.sample()
            logp_action = self.pi._log_prob_from_distribution(pi, action)
            value = self.value(obs)
        return action.numpy(), value.numpy(), logp_action.numpy()

    def act(self, obs):
        return self.step(obs)[0]

env = gym.make('CarRacing-v0')
model = torch.load('./data/ppo/ppo_s0/pyt_save/model.pt')
model.eval()

render = True
n_episodes = 10

print(env.action_space)
print(env.observation_space)
rewards = []
for i_episode in range(n_episodes):
    observation = env.reset()
    observation = rgb2gray(observation)
    sum_reward = 0
    for t in range(1000):
        if render:
            #imshow(observation)
            env.render()
        # [steering, gas, brake]
        action, value, logp = model.step(torch.as_tensor(observation.copy(), dtype=torch.float32))
        # observation is 96x96x3
        observation, reward, done, _ = env.step(action[0])
        observation = rgb2gray(observation)
        # break
        sum_reward += reward
        if(t % 100 == 0):
            print(t)
        if done or t == 999:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print("Reward: {}".format(sum_reward))
            rewards.append(sum_reward)
        if done:
            break
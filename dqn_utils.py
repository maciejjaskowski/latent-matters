# -*- coding: utf-8 -*-

import gym
import math
import random

import torchvision
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, action_space_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(2240, action_space_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQN_FC(nn.Module):

    def __init__(self, n_keypoints, action_space_size):
        super(DQN_FC, self).__init__()
        self.fc1 = nn.Linear(n_keypoints*2*2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.head = nn.Linear(128, action_space_size)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)



resize = T.Compose([T.ToPILImage(),
                    T.Resize((80, 80), interpolation=Image.BICUBIC),
                    T.ToTensor()])
import cv2
def get_screen(observation, env, device):
    img = cv2.resize(np.array(observation), (80, 80))
    tensor = torchvision.transforms.ToTensor()(img)
    return tensor.unsqueeze(0).to(device)


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 50000





def select_action(state, step_count, action_space_size, policy_net, device):

    sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * step_count / EPS_DECAY)

    if step_count > 0 and sample > eps_threshold:
        with torch.no_grad():
            ac = policy_net(state)

            action = ac.max(1)[1].view(1, 1)
            print()
            print("!!  On Policy Action thr: {} act: {} act_lst: {}".format(eps_threshold, action.item(), ac))
            print()
            return action, ac.max(1)[0]
    else:
        return torch.tensor([[random.randrange(action_space_size)]], device=device, dtype=torch.long), 0.0




def plot_pan_episode_rewards(episode_durations, is_ipython, display):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_loss(episode_losses, is_ipython, display):
    plt.figure(3)
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Q Loss')
    plt.plot(np.array(episode_losses))

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_q(qs, is_ipython, display):
    plt.figure(4)
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Q')
    plt.plot(np.array(qs))

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_rewards(qs, is_ipython, display):
    plt.figure(5)
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('r')
    plt.plot(np.array(qs))

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def get_loss(transitions, policy_net, target_net, device):
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = [s for s in batch.next_state
                                                if s is not None]
    if non_final_next_states != []:
        non_final_next_states = torch.cat(non_final_next_states)
    # final_next_states = [s for s in batch.next_state if s is None]
    # if final_next_states != []
    #     final_next_states = torch.cat(final_next_states)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = target_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(len(transitions), device=device)
    if type(non_final_next_states) != list:
        # if len(transitions) > 1:
        #     print(non_final_next_states)
        #     print( policy_net(non_final_next_states).max(1)[0].detach())
        #     print(next_state_values[non_final_mask])
        next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    return F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

######################################################################
# Training loop
# ^^^^^^^^^^^^^

def optimize_model(memory, device, policy_net, target_net, optimizer):

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    loss = get_loss(transitions, policy_net, target_net, device)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



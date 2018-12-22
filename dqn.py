# -*- coding: utf-8 -*-
"""
Shamelessly copied from PyTorch tutorial & adjusted to my needs.
"""
import numpy as np
from itertools import count

import gym
from gym import wrappers
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import tqdm

import dqn_utils
from atari_wrappers import wrap_deepmind
from dqn_utils import get_screen, DQN, ReplayMemory, plot_pan_episode_rewards, optimize_model, select_action, DQN_FC, EPS_END
from latent_training_inspired2 import Net

env = gym.make('Pong-v0')
env = wrap_deepmind(wrappers.Monitor(env, "/tmp/gym-results"),
                    episode_life=True,
                    clip_rewards=True,
                    frame_stack=False,
                    scale=False)


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
else:
    display = None

plt.ion()

# if gpu is to be used
device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")




observation = env.reset()
import cv2
print(observation.shape)
cv2.imwrite("sth.png", cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))


def plot_observation(observation, keypoints=None):
    screen = get_screen(observation, env=env, device=device).cpu().squeeze(0).permute(1, 2, 0).detach().numpy()

    if keypoints is not None:
        keypoints = keypoints.cpu().detach().numpy()
        # print(keypoints.shape)
        for i_keypoint in range(keypoints.shape[1]):
            # print(type(screen))
            screen[np.int32(np.round(keypoints[0,i_keypoint,0])),
                   np.int32(np.round(keypoints[0,i_keypoint,1])), :] = [1, 0, 0]

    plt.figure(1)
    plt.imshow(screen, interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    plt.pause(0.001)  # pause a bit so that plots are updated

plot_observation(observation)


TARGET_UPDATE = 2000

# policy_net = DQN(action_space_size=env.action_space.n).to(device)
# target_net = DQN(action_space_size=env.action_space.n).to(device)
n_keypoints = 16
feature_extractor_net = Net(n_input_channels=3, n_hidden_channels=n_keypoints, device=torch.device("cuda")).to(torch.device("cuda"))
pretrained_dict = torch.load('../atari-objects-evaluations/2018-12-18-21-25-39/models/model_00141_21.832.pt')
feature_extractor_net.load_state_dict(pretrained_dict['model_state_dict'])
feature_extractor_net.eval()

# policy_net = DQN(action_space_size=env.action_space.n).to(device)
# target_net = DQN(action_space_size=env.action_space.n).to(device)
policy_net = DQN_FC(action_space_size=env.action_space.n, n_keypoints=n_keypoints).to(device)
policy_net.train()
target_net = DQN_FC(action_space_size=env.action_space.n, n_keypoints=n_keypoints).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(50000)
PREPOPULATE_MEMORY = 10000

optimization_count = 0
step_count = 0
num_episodes = 3000

import dqn_utils
def test():
    # Initialize the environment and state
    observation = env.reset()
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    last_screen = get_screen(observation, env, device)
    current_screen = get_screen(observation, env, device)
    # print(current_screen.shape)
    with torch.no_grad():
        state = feature_extractor_net.forward({'first': current_screen.to(torch.device("cuda")),
                                               'first_prev': last_screen.to(torch.device("cuda"))})
    state = torch.cat([torch.Tensor(state['keypoints1'].detach().cpu().numpy()),
                       torch.Tensor(state['keypoints1_prev'].detach().cpu().numpy())], dim=2).to(device)
    episode_reward = 0.0

    for t in tqdm.tqdm(count()):
        # Select and perform an action
        dqn_utils.EPS_END = 0.02
        action, q = select_action(state, step_count=1000000, action_space_size=env.action_space.n, policy_net=policy_net, device=device)

        observation_human, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        episode_reward += reward
        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(observation_human, env, device)
        if not done:
            with torch.no_grad():
                next_state = feature_extractor_net.forward({'first': current_screen.to(torch.device("cuda")),
                                                            'first_prev': last_screen.to(torch.device("cuda"))})
            plot_observation(observation_human, next_state['keypoints1'])
            next_state = torch.cat([torch.Tensor(next_state['keypoints1'].detach().cpu().numpy()),
                                    torch.Tensor(next_state['keypoints1_prev'].detach().cpu().numpy())], dim=2)
        else:
            next_state = None
            print("Next state\n"*10, next_state)
        # Move to the next state
        state = next_state
        if done:
            break

pan_episode_rewards = []
for i_episode in range(0, num_episodes):
    # Initialize the environment and state
    observation = env.reset()
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    last_screen = get_screen(observation, env, device)
    current_screen = get_screen(observation, env, device)
    # print(current_screen.shape)
    with torch.no_grad():
        state = feature_extractor_net.forward({'first': current_screen.to(torch.device("cuda")),
                                               'first_prev': last_screen.to(torch.device("cuda"))})
    state = torch.cat([torch.Tensor(state['keypoints1'].detach().cpu().numpy()),
                       torch.Tensor(state['keypoints1_prev'].detach().cpu().numpy())], dim=2).to(device)
    episode_reward = 0.0
    episode_losses = []
    episode_qs = []
    episode_rewards = []


    for t in tqdm.tqdm(count()):
        # Select and perform an action
        action, q = select_action(state, step_count=step_count - PREPOPULATE_MEMORY, action_space_size=env.action_space.n, policy_net=policy_net, device=device)
        episode_qs.append(q)
        observation_human, reward, done, _ = env.step(action.item())

        step_count += 1
        observation = cv2.cvtColor(observation_human, cv2.COLOR_RGB2BGR)
        reward = torch.tensor([reward], device=device)
        episode_rewards.append(reward)

        episode_reward += reward
        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(observation_human, env, device)
        if not done:
            with torch.no_grad():
                next_state = feature_extractor_net.forward({'first': current_screen.to(torch.device("cuda")),
                                                            'first_prev': last_screen.to(torch.device("cuda"))})
            if t % 125 == 0:
                plot_observation(observation_human, next_state['keypoints1'])

            next_state = torch.cat([torch.Tensor(next_state['keypoints1'].detach().cpu().numpy()),
                                    torch.Tensor(next_state['keypoints1_prev'].detach().cpu().numpy())], dim=2)
        else:
            next_state = None
            print("Next state\n"*10, next_state)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        episode_losses.append(dqn_utils.get_loss(transitions=[dqn_utils.Transition(state, action, next_state, reward)],
                                                 policy_net=policy_net,
                                                 target_net=target_net,
                                                 device=device).detach().cpu().numpy())
        if len(episode_losses) % 125 == 0:
            dqn_utils.plot_loss(episode_losses, is_ipython, display)
            dqn_utils.plot_q(episode_qs, is_ipython, display)
            dqn_utils.plot_rewards(episode_rewards, is_ipython, display)
        # Move to the next state
        state = next_state
        if step_count > PREPOPULATE_MEMORY:
            # Perform one step of the optimization (on the target network)
            optimization_count += 1
            optimize_model(memory=memory, device=device, policy_net=policy_net, target_net=target_net,
                           optimizer=optimizer)
        # Update the target network
        if optimization_count > 0 and optimization_count % TARGET_UPDATE == 0:
            print("!!!!!   TARGET UPDATE !!!!!\n" * 5 + str(optimization_count))
            target_net.load_state_dict(policy_net.state_dict())
        if done:
            pan_episode_rewards.append(episode_reward)
            plot_pan_episode_rewards(pan_episode_rewards, is_ipython=is_ipython, display=display)
            break




print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
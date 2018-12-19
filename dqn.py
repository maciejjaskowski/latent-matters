# -*- coding: utf-8 -*-
"""
Shamelessly copied from PyTorch tutorial & adjusted to my needs.
"""

from itertools import count

import gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import tqdm

from dqn_utils import get_screen, DQN, ReplayMemory, plot_durations, optimize_model, select_action

env = gym.make('Pong-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")




env.reset()
plt.figure()
plt.imshow(get_screen(env=env, device=device).cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()
print("Example")


TARGET_UPDATE = 10

policy_net = DQN(action_space_size=env.action_space.n).to(device)
target_net = DQN(action_space_size=env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in tqdm.tqdm(count()):
        # Select and perform an action
        action = select_action(state, policy_net=policy_net, device=device)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model(memory=memory,  device=device, policy_net=policy_net, target_net=target_net, optimizer=optimizer)
        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations, is_ipython=is_ipython, display=display)
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
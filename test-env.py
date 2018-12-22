import gym
env = gym.make('Pong-v0')
print("Number of actions: ", env.action_space.n)
print(dir(env.action_space), )
print(env.unwrapped.get_action_meanings())
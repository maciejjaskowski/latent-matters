import gym
env = gym.make('Pong-v0')
env = gym.make('PongNoFrameskip-v4')
print("Number of actions: ", env.action_space.n)
print(dir(env.action_space), )
print(env.unwrapped.get_action_meanings())
import time
env.reset()
for _ in range(10000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  time.sleep(0.0001)
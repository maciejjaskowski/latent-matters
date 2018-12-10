import json
import sys

import gym
import cv2
import datetime
import tqdm
env = gym.make("Pong-v0")
run_id = str(datetime.datetime.utcnow()).replace(" ", "_")
print(run_id)

expected_observations = 1000000

all_obs_count = 0
for i_game in range(expected_observations):
  if all_obs_count >= expected_observations:
    sys.exit(0)
  print("All observations count {} out of {}".format(all_obs_count, expected_observations))
  observation = env.reset()
  obs_name = "observations/obs_{}_{}_{}.png".format(run_id, i_game, 0)
  cv2.imwrite(obs_name, observation)
  obs_names = []
  for step in range(1, 1000000):
    all_obs_count += 1
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    prev_obs_name = obs_name
    obs_name = "observations/obs_{}_{}_{}.png".format(run_id, i_game, step)
    cv2.imwrite(obs_name, cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
    obs_names.append({"png": obs_name,
                      "reward": reward,
                      "action": action,
                      "i_game": i_game,
                      "step": step,
                      "prev_png": prev_obs_name})

    if done:
      with open("observations/ls_{}_{}.json".format(run_id, i_game), "w") as f:
        json.dump(obs_names, f)

      break

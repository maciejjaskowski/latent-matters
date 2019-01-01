import json
import sys

import gym
import cv2
import datetime
import argparse
import os
import tqdm


def generate_observations(dir, run_id, expected_observations):

    all_obs_count = 0
    for i_game in range(expected_observations):

        print("All observations count {} out of {}".format(all_obs_count, expected_observations))
        observation = env.reset()
        obs_name = os.path.join(dir, "obs_{}_{}_{}.png".format(run_id, i_game, 0))
        cv2.imwrite(obs_name, observation)
        obs_names = []
        step = 0
        done = False
        while not done:
            step += 1
            all_obs_count += 1
            action = env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)
            prev_obs_name = obs_name
            obs_name = os.path.join(dir, "obs_{}_{}_{}.png".format(run_id, i_game, step))
            cv2.imwrite(obs_name, cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
            obs_names.append({"png": obs_name,
                              "reward": reward,
                              "action": action,
                              "i_game": i_game,
                              "step": step,
                              "prev_png": prev_obs_name})

        with open(os.path.join(dir, "ls_{}_{}.json".format(run_id, i_game)), "w") as f:
            json.dump(obs_names, f)

        if all_obs_count >= expected_observations:
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--destination-dir", required=True)
    parser.add_argument("--number-of-observations", type=int, default=100000)
    args = parser.parse_args()

    env = gym.make(args.env)
    run_id = str(datetime.datetime.utcnow()).replace(" ", "_")

    os.makedirs(args.destination_dir, exist_ok=False)
    generate_observations(args.destination_dir, run_id=run_id, expected_observations=args.number_of_observations)

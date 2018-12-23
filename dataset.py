import cv2
import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torch.utils.data import Sampler


class Dataset(data.Dataset):

    def __init__(self, root, n_games, min_diff, max_diff, epoch_size, shuffle=True):
        self.max_diff = max_diff  # Maximum distance to be predicted
        self.min_diff = min_diff
        self.epoch_size = epoch_size

        games = [os.path.join(root, d) for d in os.listdir(root) if d.endswith(".json")]
        self.obs_count = 0
        self.shuffle = shuffle

        self.n_games = n_games

        self.game_data = []
        for game in games[:n_games]:
            with open(game, "r") as f:
                game_obs = json.load(f)
                # FIXME countdown=22 bo przez okolo 22 ramek nic sie nie dzieje i nie widac pilki
                # At some point we should find a better way to model that in the model.

                # FIXME(now) niech kazdy punkt bedzie osobna gra, zeby uniknac przeskakiwania obiektow - nie wiadomo jak to modelowac.
                # At some point we should allow such jumps
                countdown = 22
                filtered_game_obs = []
                for o in game_obs:
                    if o['reward'] != 0: # or len(filtered_game_obs) > 3:

                        countdown = 22
                        self.game_data.append(filtered_game_obs)
                        self.obs_count += len(filtered_game_obs)
                        filtered_game_obs = []
                        #break
                    if countdown > 0:
                        countdown -= 1
                    else:
                        filtered_game_obs.append(o)

        print(len(self.game_data))
        self.games_count = len(self.game_data[:n_games])
        print(len(self.game_data[0]))
        print(self.obs_count)
        assert self.games_count > 0
        self.root = root

    def transform(self, img):
        img = cv2.resize(np.array(img), (80, 80))
        tensor = torchvision.transforms.ToTensor()(img)
        return tensor

    def __len__(self):
        return self.epoch_size

    def get_sample(self, game_index, index1, index2):
        assert index1 <= index2
        prefix = "../atari-objects-observations/"
        first = prefix + self.game_data[game_index][index1]['png']
        first_prev = prefix + self.game_data[game_index][index1]['prev_png']

        second = prefix + self.game_data[game_index][index2]['png']
        second_prev = prefix + self.game_data[game_index][index2]['prev_png']

        return {"first": self.transform(Image.open(first)),
                "first_prev": self.transform(Image.open(first_prev)),
                "second": self.transform(Image.open(second)),
                "second_prev": self.transform(Image.open(second_prev)),
                }, np.float32(index2 - index1)

    def __getitem__(self, _):
        game = torch.randint(high=self.games_count, size=(1,), dtype=torch.int64).tolist()[0]
        index1 = torch.randint(high=len(self.game_data[game])-self.min_diff, size=(1,)).tolist()[0]
        index2 = torch.randint(low=max(index1+self.min_diff, index1), high=min(index1+self.max_diff, len(self.game_data[game])), size=(1,)).tolist()[0]
        sample = self.get_sample(game_index=game, index1=index1, index2=index2)
        # print("GII", game, index1, index2)

        return sample

    def iter_game(self, i_game):
        for obs in self.game_data[i_game]:
            yield obs

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of observations: {}\n'.format(self.obs_count)
        fmt_str += '    Number of games: {}\n'.format(self.games_count)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


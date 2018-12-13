from __future__ import print_function
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

from dataset import Dataset
from PIL import Image

from latent_training_inspired import Net
import torch as t

class TestNetI():

    # def test_forward(self):
    #     hw = 80
    #     net = Net(n_input_channels=2, n_hidden_channels=3)
    #     input = np.zeros((2, hw, hw), dtype=np.float32) - 10000.0
    #     input[0, 2, 3] = 10000.0
    #     input[1, 0, 1] = 10000.0
    #     input = input.reshape([-1,2, hw, hw])
    #     tensor = t.Tensor(input)
    #
    #     res = net.forward({'first': tensor, 'first_prev': tensor, 'second': tensor})
    #
    #     assert res['img_change'].mean() == 0.0, list(res['img_change'].shape) == [2, 3, 18, 18]

    def test_losses(self):
        hw = 18
        net = Net(n_input_channels=1, n_hidden_channels=1)
        map1 = np.zeros((1, 1, hw, hw), dtype=np.float32)
        map1[0, 0, 0, 0] = 1.0
        map1 = t.Tensor(map1)

        img_change= np.zeros((1, 1, hw, hw), dtype=np.float32)
        img_change[0, 0, 0, 0] = 0.5
        img_change = t.Tensor(img_change)

        keypoints1 = np.zeros((1, 1, 2), dtype=np.float32)
        keypoints1[0, 0, :] = [3,3]
        keypoints1 = t.Tensor(keypoints1)

        keypoints1_prev = np.zeros((1, 1, 2), dtype=np.float32)
        keypoints1_prev[0, 0, :] = [2, 2]
        keypoints1_prev = t.Tensor(keypoints1_prev)

        keypoints_consistency_loss, silhuette_consistency_loss = net.losses(keypoints1=keypoints1, map1=map1, keypoints1_prev=keypoints1_prev, img_change=img_change)

        print(keypoints_consistency_loss, silhuette_consistency_loss)
        assert keypoints_consistency_loss == (1 + 1)**2
        assert abs(silhuette_consistency_loss - -np.log(1.0 * 0.5)) < 0.001

    def test_softargmax(self):
        net = Net(n_input_channels=1, n_hidden_channels=1)
        input = np.zeros((1, 80, 80), dtype=np.float32) - 10000.0
        input[0, 2, 3] = 10000.0
        input[0, 2, 4] = 0.0
        input = input.reshape([-1,1, 80, 80])
        tensor = t.Tensor(input)
        res = net._softargmax(tensor)

        assert np.all(res[0].numpy() == np.array([[2]]))
        assert np.all(res[1].numpy() == np.array([[3]]))





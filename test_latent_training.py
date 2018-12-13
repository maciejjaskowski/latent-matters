from __future__ import print_function

import numpy as np
import torch as t

from latent_training import Net


class TestNet():

    def test_softargmax(self):
        net = Net(n_input_channels=2)
        input = np.zeros((2, 9, 9), dtype=np.float32) - 10000.0
        input[0, 2, 3] = 10000.0
        input[1, 0, 1] = 10000.0
        input = input.reshape([-1,2, 9, 9])
        tensor = t.Tensor(input)
        res = net._softargmax(tensor)

        assert np.all(res[0].numpy() == np.array([[2, 0]]))
        assert np.all(res[1].numpy() == np.array([[3, 1]]))


    def test_smoke(self):
        net = Net(n_input_channels=2)

        input = np.zeros((3, 2, 9, 9), dtype=np.float32)
        tensor = t.Tensor(input)
        res = net._one_pass(tensor)
        print(res[0].shape, res[1].shape)

    def test_forward(self):
        net = Net(n_input_channels=2)

        input = np.zeros((3, 2, 9, 9), dtype=np.float32)
        tensor = t.Tensor(input)
        res = net.forward({'first': tensor,
                           'first_prev': tensor,
                           'second': tensor,
                           'second_prev': tensor})
        assert list(res.shape) == [3,1]



# Assumptions to be lifted in the future:
# 1. The objects do not disappear
# 2. The maximum distance to be predicted is limited (16)
# 3. The objects do not "jump" (e.g. in Pong after a score)
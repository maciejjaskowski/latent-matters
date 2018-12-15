from __future__ import print_function

import cv2
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

class Net(nn.Module):
    def __init__(self, n_input_channels, n_hidden_channels):
        super(Net, self).__init__()
        self.n_hidden_channels = n_hidden_channels
        self.conv1 = nn.Conv2d(n_input_channels, 20, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv2d(20, 30, kernel_size=3, dilation=1, stride=1)

        self.bn = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, self.n_hidden_channels, kernel_size=3, dilation=1, stride=1)  # Modelujemy 3 obiekty


        # self.fc2_dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(1+self.n_hidden_channels, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.ran = torch.Tensor([range(80)] * 80)

    def _softargmax(self, x4, T=400.0):
        softmaxed = F.softmax(T*x4.reshape([-1, x4.shape[1], x4.shape[2]*x4.shape[3]]), dim=-1).reshape(x4.shape)  #, _stacklevel=5)
        y_v = torch.stack([self.ran] * softmaxed.shape[1], dim=0)
        y_v = torch.stack([y_v] * softmaxed.shape[0], dim=0)
        x_v = torch.stack([self.ran.t()] * softmaxed.shape[1], dim=0)
        x_v = torch.stack([x_v] * softmaxed.shape[0], dim=0)
        assert (softmaxed.shape == y_v.shape), softmaxed.shape
        y_v = (softmaxed * y_v).sum(dim=[2, 3])
        x_v = (softmaxed * x_v).sum(dim=[2, 3])
        return y_v, x_v, softmaxed

    def _encode(self, x):
        # print("SSS", x[0,0,40,:].mean())
        x2 = F.relu(self.conv1(x-0.33))
        # x3 = F.relu(self.bn(self.conv2(x2)))
        x3 = F.relu(self.conv2(x2))
        dense = self.conv3(x3)  # No activation as we _softargmax the result
        dense_upscaled = F.interpolate(dense, size=(80, 80), mode='bilinear', align_corners=False)
        x_v, y_v, softmaxed = self._softargmax(dense_upscaled)
        assert list(x_v.shape)[1:] == [self.n_hidden_channels], (list(x_v.shape)[1:], [self.n_hidden_channels])
        xy = torch.stack([x_v, y_v], dim=2)
        assert list(xy.shape[1:]) == [self.n_hidden_channels, 2], "{} != {}".format(list(xy.shape[1:]), [self.n_hidden_channels, 2])
        return xy, softmaxed, dense, dense_upscaled


    def losses(self, keypoints1, map1, keypoints1_prev, img_change):

        keypoints_consistency_loss = []
        silhuette_consistency_loss = []
        silhuette_sum_loss = []
        eps = 1e-7
        for b in range(keypoints1.shape[0]):
            for k in range(keypoints1.shape[1]):
                keypoints_consistency_loss.append(torch.sum((keypoints1[b,k,:] - keypoints1_prev[b,k,:])**2))
                # print("SHAPE", map1.shape, img_change.shape)
                if torch.mean(img_change[b]) > 0:
                    silhuette_consistency_loss.append(-torch.log(eps + torch.sum(map1[b,k,:,:] * img_change[b,:,:])))

                    silhuette_sum_loss.append(-torch.log(eps + torch.sum((torch.sum(map1[b,:,:,:], dim=0) * img_change[b,:,:]))))


        delta = 3.0
        keypoint_variety_loss = torch.Tensor([0.0])
        for b in range(keypoints1.shape[0]):
            for i in range(keypoints1.shape[1]):
                for j in range(keypoints1.shape[2]):
                    if i != j:
                        cur = torch.max(delta**2 - torch.sum((keypoints1[b,i,:] - keypoints1[b,j,:])**2), torch.Tensor([0.0]))

                        keypoint_variety_loss += cur
        keypoint_variety_loss /= keypoints1.shape[1]**2 * keypoints1.shape[0]

        # print(torch.mean(img_change, dim=[1,2]), img_change.shape)
        silhuette_sum_loss = torch.mean(torch.stack(silhuette_sum_loss))
        keypoints_consistency_loss = torch.mean(torch.stack(keypoints_consistency_loss))
        silhuette_consistency_loss = torch.mean(torch.stack(silhuette_consistency_loss))
        return keypoints_consistency_loss, silhuette_consistency_loss, keypoint_variety_loss[0], silhuette_sum_loss

    def forward(self, X):
        keypoints1, map1, _, _ = self._encode(X['first'])
        keypoints1_prev, _, _, _ = self._encode(X['first_prev'])
        keypoints2, map2, _, _ = self._encode(X['second'])

        # img_change = torch.sum(((torch.abs(X['first'] - X['second']) > 0)).float(), dim=1)
        img_change = (torch.sum(torch.abs(X['first_prev'] - X['first']), dim=1) > 0).float()

        # print("img_change", img_change.shape)
        # print("keypoints", keypoints1[0])

        keypoints_consistency_loss, silhuette_consistency_loss, keypoint_variety_loss, silhuette_sum_loss = self.losses(keypoints1=keypoints1, keypoints1_prev=keypoints1_prev, map1=map1, img_change=img_change)

        return {"keypoints1": keypoints1,
                "keypoints2": keypoints2,
                "keypoints1_prev": keypoints1_prev,
                "map1": map1,
                "map2": map2,
                "img_change": img_change,
                "keypoint_variety_loss": keypoint_variety_loss,
                "keypoints_consistency_loss": keypoints_consistency_loss,
                "silhuette_consistency_loss": silhuette_consistency_loss,
                "silhuette_sum_loss": silhuette_sum_loss}



def train(args, classification, model, device, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = {key: d.to(device) for key, d in data.items()}
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss_move = []
        for b in range(output['keypoints1'].shape[0]):
            for k in range(output['keypoints1'].shape[1]):
                loss_move.append(
                    ((output['keypoints1'][b,k] - output['keypoints1_prev'][b,k]) * target[b] - (output['keypoints2'][b,k] - output['keypoints1_prev'][b,k]))**2)
        loss_move = 0.1 * torch.mean(torch.stack(loss_move))
        keypoints_consistency_loss = 0.003*output['keypoints_consistency_loss']
        # Chce zeby keypoint_variaty loss nie bylo duzy, ale gwaltownie rosl
        keypoint_variety_loss = 0.03*output['keypoint_variety_loss']
        silhuette_sum_loss = output['silhuette_sum_loss']

        # loss = loss_move + output['silhuette_consistency_loss'] + output['keypoint_variety_loss'] + keypoints_consistency_loss
        loss = output['silhuette_consistency_loss'] + loss_move + silhuette_sum_loss #keypoint_variety_loss #keypoints_consistency_loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(output['keypoints_consistency_loss'])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} move: {:.6f} key_var: {:.6f} silh_sum: {:.6f} key_cons: {:.6f} silh_cons: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_move,
                                                                   keypoint_variety_loss,
                                                                   silhuette_sum_loss,
                                                                   keypoints_consistency_loss,
                                                                   output['silhuette_consistency_loss']))


def test(epoch, model, device, train_dataset):
    print("Test epoch {}".format(epoch))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        game = train_dataset.game_data[0]
        for i_obs in range(len(game)-1):
            res = single_image(model, i_obs)
            Image.fromarray(res).save('../atari-objects-evaluations/epoch{:05d}_game0_{:03d}.png'.format(epoch, i_obs))
    #
    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

def single_image(model, i_obs):
    # sample = iter(test_loader).next()
    sample, _ = train_dataset.get_sample(0, i_obs, i_obs + 1)
    res = model.forward({'first': sample['first'].unsqueeze(0), 'first_prev': sample['first_prev'].unsqueeze(0),
                     'second': sample['second'].unsqueeze(0)})
    png = sample['first']
    xy, softmaxed, dense, dense_upscaled = model._encode(png.unsqueeze(0))
    xy = xy.detach().numpy()
    # print("XY shape", xy.shape)
    softmaxed = softmaxed.detach().numpy()
    # xy = np.stack([np.unravel_index(np.argmax(dense_upscaled.detach().numpy()[0][i]), (80, 80)) for i in range(6)])
    # xy = xy.reshape((1, 6, 2))
    # print("XY shape2", xy.shape)
    x_r = xy[:, :, 0]
    y_r = xy[:, :, 1]
    r = np.zeros((80, 80, 3), np.uint8)
    # print(softmaxed[0, :, :, :])
    attention = (cv2.resize(softmaxed[0, :, :, :].transpose([1, 2, 0]), (80, 80)) / softmaxed[0, :, :,
                                                                                :].max() * 255).astype(np.uint8)
    r[np.round(x_r).astype(np.int32), np.round(y_r).astype(np.int32), :] = [255, 0, 0]
    # print(np.array(png).dtype, attention.dtype, r.dtype)
    repr = Image.fromarray(r).resize((320, 320))
    attention = attention[:, :, :3] + attention[:, :, 3:]
    # print(attention.shape)
    attention = Image.fromarray(attention).resize((320, 320))
    # print(np.array(repr))
    # res = np.maximum(np.array(torchvision.transforms.ToPILImage()(png).resize((320, 320))), np.array(repr))#, np.array(attention))
    res = np.maximum(np.array(repr), np.array(attention))
    # print(attention)
    # res = np.array(attention)
    return res


# Max diff ustawic na 2 lub 3 ?
# Bilinear w innym miejscu ?
# Zrobić "zwykly" attention do oszacowania img_change, po to zeby suma "map" dodawala sie do img_change ?
# [DONE] Cos ten softargmax nie dziala tak jak powinien (porownaj z argmax)
# Silhuette sum loss moglby byc lepszy, gdyby uzyc KL divergence ?
# [Fixed] Silhuette is not robust because img_change is calculated wrong ?


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_path = '../atari-objects-observations/'
    batch_size = 64
    train_dataset = Dataset(
        root=data_path,
        n_games=10000,
        min_diff=1,
        max_diff=2,
        epoch_size=batch_size * 10
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=5,shuffle=False,
    )

    def img_diff(second, first):
        return (1.0 + second - first) / 2


    classification = False
    model = Net(n_input_channels=3, n_hidden_channels=6).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        test(epoch, model, device, train_dataset)
        train(args, classification, model, device, train_loader, optimizer, epoch)


# Assumptions to be lifted in the future:
# 1. The objects do not disappear
# 2. The maximum distance to be predicted is limited (16)
# 3. The objects do not "jump" (e.g. in Pong after a score)
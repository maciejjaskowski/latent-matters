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
        self.conv2 = nn.Conv2d(20, 30, kernel_size=3, dilation=1, stride=2)

        self.bn = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, self.n_hidden_channels, kernel_size=3, dilation=1, stride=2)  # Modelujemy 3 obiekty


        # self.fc2_dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(1+self.n_hidden_channels, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.ran = torch.Tensor([range(80)] * 80)

    def _softargmax(self, x4):
        softmaxed = F.softmax(x4.reshape([-1, x4.shape[1], x4.shape[2]*x4.shape[3]]), dim=-1).reshape(x4.shape)  #, _stacklevel=5)

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
        x3 = F.relu(self.bn(self.conv2(x2)))
        dense = self.conv3(x3)  # No activation as we _softargmax the result
        dense_upscaled = F.interpolate(dense, size=(80, 80), mode='bilinear', align_corners=False)
        x_v, y_v, softmaxed = self._softargmax(dense_upscaled)

        assert list(x_v.shape)[1:] == [self.n_hidden_channels], (list(x_v.shape)[1:], [self.n_hidden_channels])
        xy = torch.stack([x_v, y_v], dim=2)
        assert list(xy.shape[1:]) == [self.n_hidden_channels, 2], "{} != {}".format(list(xy.shape[1:]), [self.n_hidden_channels, 2])
        return xy, softmaxed

    def _combine_with_one_hot(self, res, i):
        assert self.n_hidden_channels == 3, " A Temporary assumption"
        one_hot = torch.Tensor(1.0 * np.array([[i == 0, i == 1, i == 2]] * res.shape[0]))
        # print("SHAPE", one_hot.shape)
        #
        # print("SHAPE", res.shape)
        r = torch.cat([res.reshape(-1,1), one_hot], dim=1)
        # print("SHAPE", r.shape)
        return r

    def _decode(self, res, i):
        r = self._combine_with_one_hot(res, i)

        res1 = F.relu(self.fc1(r))
        res2 = F.relu(self.fc2(res1))
        return self.fc3(res2)

    def losses(self, keypoints1, map1, keypoints1_prev, img_change):

        keypoints_consistency_loss = []
        silhuette_consistency_loss = []
        eps = 1e-7
        for b in range(keypoints1.shape[0]):
            for k in range(keypoints1.shape[1]):
                keypoints_consistency_loss.append(torch.sum(keypoints1[b,k,:] - keypoints1_prev[b,k,:])**2)
                # print("SHAPE", map1.shape, img_change.shape)
                if torch.mean(img_change[b]) > 0:
                    silhuette_consistency_loss.append(-torch.log(eps + torch.sum(map1[b,k,:,:] * img_change[b,:,:])))

        keypoint_variety_loss = torch.Tensor(0.0)
        for b in range(keypoints1.shape[0]):
            for i in range(keypoints1.shape[1]):
                for j in range(keypoints1.shape[2]):
                    if i != j:
                        keypoint_variety_loss += torch.max(torch.sum((keypoints1[b,i,:] - keypoints1[b,j,:])**2), 0)
        keypoint_variety_loss /= keypoints1.shape[1]**2 * keypoints1.shape[0]

        # print(torch.mean(img_change, dim=[1,2]), img_change.shape)
        keypoints_consistency_loss = torch.mean(torch.stack(keypoints_consistency_loss))
        silhuette_consistency_loss = torch.mean(torch.stack(silhuette_consistency_loss))
        return keypoints_consistency_loss, silhuette_consistency_loss, keypoint_variety_loss

    def forward(self, X):
        keypoints1, map1 = self._encode(X['first'])
        keypoints1_prev, _ = self._encode(X['first_prev'])
        keypoints2, map2 = self._encode(X['second'])

        img_change = torch.sum(((torch.abs(X['first'] - X['second']) > 0)).float(), dim=1)

        print("keypoints", keypoints1[0])

        keypoints_consistency_loss, silhuette_consistency_loss, keypoint_variety_loss = self.losses(keypoints1=keypoints1, keypoints1_prev=keypoints1_prev, map1=map1, img_change=img_change)

        return {"keypoints1": keypoints1,
                "keypoints2": keypoints2,
                "keypoints1_prev": keypoints1_prev,
                "map1": map1,
                "map2": map2,
                "img_change": img_change,
                "keypoint_variety_loss": keypoint_variety_loss,
                "keypoints_consistency_loss": keypoints_consistency_loss,
                "silhuette_consistency_loss": silhuette_consistency_loss}



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
        loss_move = torch.mean(torch.stack(loss_move))

        loss = 0.3 * loss_move + 0.03*output['keypoints_consistency_loss'] + output['silhuette_consistency_loss'] + output['keypoint_variety_loss']
        # loss = output['silhuette_consistency_loss']
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} move: {:.6f} key_cons: {:.6f} silh_cons: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_move,
                                                                   output['keypoints_consistency_loss'],
                                                                   output['silhuette_consistency_loss']))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def play(train_dataset):
    game = train_dataset.game_data[0]
    for i_obs, obs in enumerate(game):
        # sample = iter(test_loader).next()
        png = Image.open("../atari-objects-" + obs['png']).resize((320, 320))
        # png_prev = Image.open("../atari-objects-" + obs['prev_png']).resize((320, 320))
        xy, softmaxed = model._encode(train_dataset.transform(png).unsqueeze(0))
        xy = xy.detach().numpy()
        softmaxed = softmaxed.detach().numpy()
        y_r = xy[:, :, 0]
        x_r = xy[:, :, 1]
        r = np.zeros((80, 80, 3), np.uint8)
        print(softmaxed[0, :, :, :])
        attention = (cv2.resize(softmaxed[0, :, :, :].transpose([1,2,0]), (80, 80)) / softmaxed[0, :, :, :].max() * 255).astype(np.uint8)
        r[np.round(x_r).astype(np.int32), np.round(y_r).astype(np.int32), :] = [255, 0, 0]
        print(np.array(png).dtype, attention.dtype)
        repr = Image.fromarray(r).resize((320, 320))
        attention = Image.fromarray(attention).resize((320,320))
        print(np.array(repr))
        res = np.maximum(np.array(png), np.array(repr), np.array(attention))
        Image.fromarray(res).save('game0_{:03d}.png'.format(i_obs))

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
    train_dataset = Dataset(
        root=data_path,
        n_games=100,
        min_diff=1,
        max_diff=3
    )
    train_loader = DataLoader(train_dataset, batch_size=64,num_workers=5,shuffle=False,
    )
    test_loader = DataLoader(train_dataset, batch_size=1, num_workers=5, shuffle=False)

    def img_diff(second, first):
        return (1.0 + second - first) / 2


    classification = False
    model = Net(n_input_channels=3, n_hidden_channels=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, classification, model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader)

# Assumptions to be lifted in the future:
# 1. The objects do not disappear
# 2. The maximum distance to be predicted is limited (16)
# 3. The objects do not "jump" (e.g. in Pong after a score)
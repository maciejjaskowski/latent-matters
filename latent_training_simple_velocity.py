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

class Net(nn.Module):
    def __init__(self, n_input_channels):
        super(Net, self).__init__()
        self.n_hidden_channels = 3
        self.conv1 = nn.Conv2d(n_input_channels, 20, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv2d(20, 30, kernel_size=3, dilation=1, stride=2)

        self.bn = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, self.n_hidden_channels, kernel_size=3, dilation=1, stride=2)  # Modelujemy 3 obiekty


        self.ran = torch.Tensor([range(18)] * 18) #.to(torch.current_device())
    # def to(self, device):
    #     super(Net).to(device)

    def _softargmax(self, x4):
        # print(x4)
        softmaxed = F.softmax(x4.reshape([-1, x4.shape[1], x4.shape[2]*x4.shape[3]]), dim=-1).reshape(x4.shape)  #, _stacklevel=5)
        # print("Soft", softmaxed)


        y_v = torch.stack([self.ran] * softmaxed.shape[1], dim=0)
        y_v = torch.stack([y_v] * softmaxed.shape[0], dim=0)

        x_v = torch.stack([self.ran.t()] * softmaxed.shape[1], dim=0)
        x_v = torch.stack([x_v] * softmaxed.shape[0], dim=0)
        # print("X_V shape", x_v.shape, softmaxed.shape)
        assert (softmaxed.shape == y_v.shape), softmaxed.shape
        # print((softmaxed * x_v))
        y_v = (softmaxed * y_v).sum(dim=[2, 3])
        x_v = (softmaxed * x_v).sum(dim=[2, 3])
        return x_v, y_v

    def _encode(self, x):
        # print("SSS", x[0,0,40,:].mean())

        x2 = F.relu(self.conv1(x-0.33))
        # assert x.shape[2:] == x2.shape[2:], (x.shape[2:], x2.shape[2:])
        x3 = F.relu(self.bn(self.conv2(x2)))
        # assert x2.shape[2:] == x3.shape[2:], (x2.shape[2:], x3.shape[2:])
        x4 = self.conv3(x3)  # No activation as we _softargmax the result
        # assert x3.shape[2:] == x4.shape[2:], (x3.shape[2:], x4.shape[2:])
        x_v, y_v = self._softargmax(x4)
        # print(x_v[0], y_v[0])

        assert list(x_v.shape)[1:] == [self.n_hidden_channels], (list(x_v.shape)[1:], [self.n_hidden_channels])

        return torch.stack((x_v, y_v), 2)

    def _decode(self, res):

        res1 = F.relu(self.fc1(res))
        res2 = F.relu(self.fc2(res1))
        return self.fc3(res2)

    def forward(self, X):
        n_frames = self.n_frames(X)
        # print("velocity", velocity[0])
        return n_frames

    def n_frames(self, X):
        first_positions = self._encode(X['first'])
        # print(first_positions.shape)
        first_prev_positions = self._encode(X['first_prev'])

        eps = torch.Tensor([0.0001])

        second_prev_positions = self._encode(X['second_prev'])
        nomin = (second_prev_positions - first_prev_positions)
        denom = (torch.abs(first_positions - first_prev_positions) + eps)
        print(first_positions[0])
        print(nomin[0], denom[0])
        velocity = nomin / denom
        print("velocity", velocity[0])
        speed = torch.norm(velocity, p=2, dim=-1)
        print("speed", speed[0])
        assert speed.shape == velocity.shape[:2]
        # print("speed", speed[0])

        return speed


def train(args, classification, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = {key: d.to(device) for key, d in data.items()}
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if classification:
            loss = F.nll_loss(F.log_softmax(output), target.long())
        else:
            loss = 0
            print("target", target[0], output[0])
            for b in range(output.shape[0]):

                for i in range(output.shape[1]):
                    loss += F.smooth_l1_loss(output[b, i], target[b])
            loss = loss / output.shape[0] / output.shape[1]
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


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
        xy = model._encode(train_dataset.transform(png).unsqueeze(0))
        x_r = xy[0,0,:] * 80 / 18.0
        y_r = xy[0,1,:] * 80 / 18.0
        print(x_r, y_r)
        r = np.zeros((80, 80, 3), np.uint8)
        r[np.round(x_r.detach().numpy()).astype(np.int32), np.round(y_r.detach().numpy()).astype(np.int32), :] = 255
        repr = Image.fromarray(r).resize((320, 320))
        res = np.maximum(np.array(png), np.array(repr))
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
        max_diff=2
    )
    train_loader = DataLoader(train_dataset,batch_size=64,num_workers=5,shuffle=False,
    )
    test_loader = DataLoader(train_dataset, batch_size=1, num_workers=5, shuffle=False)

    def img_diff(second, first):
        return (1.0 + second - first) / 2


    classification = False
    model = Net(n_input_channels=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, classification, model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader)

# Assumptions to be lifted in the future:
# 1. The objects do not disappear
# 2. The maximum distance to be predicted is limited (16)
# 3. The objects do not "jump" (e.g. in Pong after a score)
from __future__ import print_function

import cv2
import sys
import argparse
import datetime
import tqdm

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
import random
import os

class Net(nn.Module):
    def __init__(self, n_input_channels, n_hidden_channels):
        super(Net, self).__init__()
        self.n_hidden_channels = n_hidden_channels
        self.conv1 = nn.Conv2d(n_input_channels, 20, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv2d(20, 30, kernel_size=3, dilation=1, stride=1)

        self.bn = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, self.n_hidden_channels, kernel_size=3, dilation=1, stride=2)  # Modelujemy 3 obiekty


        # self.fc2_dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(1+self.n_hidden_channels, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.device = device

        self.ran = torch.Tensor([range(80)] * 80).to(self.device)

    def _softargmax(self, x4, T):
        softmaxed = F.softmax(T*x4.reshape([-1, x4.shape[1], x4.shape[2]*x4.shape[3]]), dim=-1).reshape(x4.shape)  #, _stacklevel=5)

        y_v = (softmaxed * torch.stack([torch.stack([self.ran] * softmaxed.shape[1], dim=0)] * softmaxed.shape[0], dim=0)).sum(dim=[2,3])
        x_v = (softmaxed * torch.stack([torch.stack([self.ran.t()] * softmaxed.shape[1], dim=0)] * softmaxed.shape[0], dim=0)).sum(dim=[2, 3])
        return x_v, y_v, softmaxed

    def _encode(self, x, T):

        x2 = F.relu(self.conv1(x-0.33))
        x3 = F.relu(self.conv2(x2))

        dense_upscaled = F.interpolate(self.conv3(x3), size=(80, 80), mode='bilinear', align_corners=False)
        x_v, y_v, softmaxed = self._softargmax(dense_upscaled, T=T)
        assert list(x_v.shape)[1:] == [self.n_hidden_channels], (list(x_v.shape)[1:], [self.n_hidden_channels])
        xy = torch.stack([x_v, y_v], dim=2)
        assert list(xy.shape[1:]) == [self.n_hidden_channels, 2], "{} != {}".format(list(xy.shape[1:]), [self.n_hidden_channels, 2])
        return xy, softmaxed, None, dense_upscaled


    def keypoints_variety_loss(self, keypoints1):
        assert len(keypoints1.shape) == 3, keypoints1.shape
        delta = 10.0
        keypoint_variety_loss = torch.Tensor([0.0]).to(self.device)
        for b in range(keypoints1.shape[0]):
            for i in range(keypoints1.shape[1]):
                for j in range(keypoints1.shape[1]):
                    if i != j:
                        cur = torch.max(delta ** 2 - torch.sum((keypoints1[b, i, :] - keypoints1[b, j, :]) ** 2),
                                        torch.Tensor([0.0]).to(self.device))
                        # print(i,j,cur)
                        keypoint_variety_loss += cur
        return keypoint_variety_loss[0] / (keypoints1.shape[0] * keypoints1.shape[1] ** 2)

    def silhuette_variance_loss(self, softmaxed, x, y):
        assert len(softmaxed.shape) == 4
        assert len(x.shape) == 2, len(x.shape)
        assert len(y.shape) == 2, len(y.shape)
        variance = []
        for b in range(softmaxed.shape[0]):
            for k in range(softmaxed.shape[1]):
                mul = softmaxed[b,k] * ((self.ran.t() - x[b,k])**2 + (self.ran - y[b,k])**2)
                v = torch.sum(mul)
                if b == 0 and k == 0 and random.randint(0,5) == 0:
                    print(b, k, v.item())
                # print(softmaxed, mul, b,k,v)
                variance.append(v)

        return torch.mean(torch.stack(variance))

    def losses(self, keypoints1, map1, img_change):
        # silhuette_consistency_loss = []
        # silhuette_sum_loss = []
        eps = 1e-7
        silhuette_consistency_loss = torch.mean(torch.stack([-torch.log(eps + torch.sum(map1[b,k,:,:] * img_change[b,:,:]))
                                                             for b in range(keypoints1.shape[0]) for k in range(keypoints1.shape[1])
                                                             if torch.mean(img_change[b]) > 0]))
        # for b in range(map1.shape[0]):
        #     img_change[b, :, :] - torch.sum((torch.sum(map1[b, :, :, :], dim=0))
            #     # print("SHAPE", map1.shape, img_change.shape)
            #     if torch.mean(img_change[b]) > 0:
            #         silhuette_consistency_loss.append()
            #
            #
            #     else:
            #         print("mean negative", torch.mean(img_change[b]))

        # print(torch.mean(img_change, dim=[1,2]), img_change.shape)
        # silhuette_sum_loss = torch.mean(torch.stack(silhuette_sum_loss))
        # silhuette_consistency_loss = torch.mean(torch.stack(silhuette_consistency_loss))
        return silhuette_consistency_loss, torch.Tensor([0.0])[0].to(self.device) #silhuette_sum_loss

    def forward(self, X):
        T = 1.0
        keypoints1, map1, _, _ = self._encode(X['first'], T=T)
        # keypoints1_prev, _, _, _ = self._encode(X['first_prev'], T=T)
        # keypoints2, map2, _, _ = self._encode(X['second'], T=T)

        # img_change = torch.sum(((torch.abs(X['first'] - X['second']) > 0)).float(), dim=1)
        img_change = (torch.sum(torch.abs(X['first_prev'] - X['first']), dim=1) > 0).float()

        # print("img_change", img_change.shape)
        # print("keypoints", keypoints1[0])

        silhuette_variance_loss = self.silhuette_variance_loss(map1, x=keypoints1[:,:,0], y=keypoints1[:,:,1])
        keypoint_variety_loss = self.keypoints_variety_loss(keypoints1)
        silhuette_consistency_loss, silhuette_sum_loss = self.losses(keypoints1=keypoints1, map1=map1, img_change=img_change)


        return {"keypoints1": keypoints1,
                # "keypoints2": keypoints2,
                # "keypoints1_prev": keypoints1_prev,
                "map1": map1,
                # "map2": map2,
                "img_change": img_change,
                "keypoint_variety_loss": keypoint_variety_loss,
                "silhuette_consistency_loss": silhuette_consistency_loss,
                "silhuette_variance_loss": silhuette_variance_loss,
                "silhuette_sum_loss": silhuette_sum_loss}


def train(args, classification, model, device, train_loader, optimizer, epoch):

    model.train()
    epoch_losses = []
    keypoint_variety_losses = []
    silhuette_variance_losses = []
    silhuette_consistency_losses = []
    silhuette_sum_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data = {key: d.to(device) for key, d in data.items()}
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # loss_move = []
        # for b in range(output['keypoints1'].shape[0]):
        #     for k in range(output['keypoints1'].shape[1]):
        #         loss_move.append(
        #             ((output['keypoints1'][b,k] - output['keypoints1_prev'][b,k]) * target[b] - (output['keypoints2'][b,k] - output['keypoints1_prev'][b,k]))**2)
        # loss_move = 0.1 * torch.mean(torch.stack(loss_move))
        # Chce zeby keypoint_variaty loss nie bylo duzy, ale gwaltownie rosl
        keypoint_variety_loss = output['keypoint_variety_loss']
        silhuette_sum_loss = output['silhuette_sum_loss']
        silhuette_variance_loss = 0.07 * output['silhuette_variance_loss']
        silhuette_consistency_loss = output['silhuette_consistency_loss']

        loss = silhuette_consistency_loss + silhuette_variance_loss + keypoint_variety_loss  # keypoints_consistency_loss
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        keypoint_variety_losses.append(keypoint_variety_loss.item())
        silhuette_sum_losses.append(silhuette_sum_loss.item())
        silhuette_consistency_losses.append(silhuette_consistency_loss.item())
        silhuette_variance_losses.append(silhuette_variance_loss.item())


    epoch_loss = np.mean(epoch_losses)
    keypoint_variety_loss = np.mean(keypoint_variety_losses)
    silhuette_sum_loss = np.mean(silhuette_sum_losses)
    silhuette_variance_loss = np.mean(silhuette_variance_losses)
    silhuette_consistency_loss = np.mean(silhuette_consistency_losses)

    print('Train Epoch: {} \tLoss: {epoch_loss:.3f} '
          'key_var: {key_var:.3f} '
          'silh_sum: {silh_sum:.3f} '
          'silh_var: {silh_var:.3f} '
          'silh_cons: {silh_cons_loss:.3}'.format(
        epoch, epoch_loss=epoch_loss, #loss_move,
               key_var=keypoint_variety_loss,
               silh_sum=silhuette_sum_loss,
               silh_var=silhuette_variance_loss,
               silh_cons_loss=silhuette_consistency_loss))

    return epoch_loss


def test(epoch, model, device, train_dataset, eval_path, T):
    print("Test epoch {}".format(epoch))
    model.eval()
    test_loss = 0
    correct = 0
    epoch_path = os.path.join(eval_path, 'epoch{:05d}'.format(epoch))
    os.makedirs(epoch_path)
    with torch.no_grad():
        game = train_dataset.game_data[0]
        for i_obs in range(len(game)-1):
            sample, _ = train_dataset.get_sample(0, i_obs, i_obs + 1)
            png = sample['first'].to(device)
            torch_xy, torch_softmaxed, _, dense_upscaled = model._encode(png.unsqueeze(0), T=T)
            attentions, png, xy_img, argmax_xy_img, xy, argmax_xy = single_image(png=png, xy=torch_xy, softmaxed=torch_softmaxed, dense_upscaled=dense_upscaled)
            if i_obs == 10:
                print("xy[10]", xy[0, :, :])
                silhuette_variance_loss = model.silhuette_variance_loss(torch_softmaxed, x=torch_xy[:, :, 0], y=torch_xy[:, :, 1])
                keypoint_variety_loss = model.keypoints_variety_loss(torch_xy[:, :, :])
                print("silh_var", silhuette_variance_loss.item())
                print("keyp_var", keypoint_variety_loss.item())
            for i_attention, attention in enumerate(attentions):
                Image.fromarray(np.uint8(np.clip(attention*255, a_min=0, a_max=255))).resize((320, 320)).save(
                    os.path.join(epoch_path, 'game0_att_{i_attention:03d}_{i_obs:03d}.png'.format(i_attention=i_attention, i_obs=i_obs)))
            attentions = np.stack([np.clip(np.sum(np.stack(attentions, axis=2), axis=2), a_min=0.0, a_max=1.0)]*3, axis=2)
            attention_png = cv2.resize(np.uint8(attentions * 127 + png * 127), (320, 320))
            Image.fromarray(attention_png).save(os.path.join(epoch_path, 'game0_attentions_png_{i_obs:03d}.png'.format(i_obs=i_obs)))
            Image.fromarray(np.uint8(xy_img*127 + png * 127)).resize((320, 320)).save(os.path.join(epoch_path, 'game0_xy_png_{i_obs:03d}.png'.format(i_obs=i_obs)))
            Image.fromarray(np.uint8(argmax_xy_img * 127 + png * 127)).resize((320, 320)).save(
                os.path.join(epoch_path, 'game0_argmaxxy_png_{i_obs:03d}.png'.format(i_obs=i_obs)))

    #
    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


def add_points(img, x, y):
    colours = [[1, 0, 0],
               [0.66, 0, 0],
               [0, 1, 0],
               [0, 0.66, 0],
               [0, 0, 1],
               [0, 0, 0.66],
               [1, 1, 0],
               [1, 0, 1],
               [0, 1, 1],
               [0.66, 1, 0],
               [0.66, 0, 1],
               [1, 0, 1],
               [0, 0.66, 1],
               [0, 0.66, 1],

               [1, 0, 0.66],
               [0, 1, 0.66],
               [0.66, 0.66, 0],
               [0.66, 0, 0.66],
               [0, 0.66, 0.66],
               ]

    for i in range(x.shape[1]):

        img[np.round(x).astype(np.int32), np.round(y).astype(np.int32), :] = colours[i]
    return img


def single_image(png, xy, softmaxed, dense_upscaled):
    png = png.detach().cpu().numpy().copy()
    xy = xy.detach().cpu().numpy().copy()
    dense_upscaled = dense_upscaled.detach().cpu().numpy().copy()
    softmaxed = softmaxed.detach().cpu().numpy().copy()
    n_keypoints = softmaxed.shape[1]
    softmaxed_normed = softmaxed[0, :, :, :].transpose([1, 2, 0])
    argmax_xy = np.stack([np.unravel_index(np.argmax(dense_upscaled[0][i]), (80, 80)) for i in range(n_keypoints)])
    argmax_xy = argmax_xy.reshape((1, n_keypoints, 2))
    eps = 1e-7
    for i in range(softmaxed_normed.shape[2]):
        softmaxed_normed[:, :, i] = softmaxed_normed[:,:,i] / (softmaxed_normed[:, :, i].max() + eps)
    attention = cv2.resize(softmaxed_normed, (80, 80))
    xy_img = add_points(np.zeros((80, 80, 3), np.uint8), xy[:, :, 0], xy[:, :, 1])
    argmax_xy_img = add_points(np.zeros((80, 80, 3), np.uint8), argmax_xy[:, :, 0], argmax_xy[:, :, 1])
    #attention = attention[:, :, 0]  # + attention[:, :, 3:]
    # print(attention.shape)
    #attention = Image.fromarray(attention).resize((320, 320))
    # print(np.array(repr))
    # res = np.maximum(np.array(torchvision.transforms.ToPILImage()(png).resize((320, 320))), np.array(repr))#, np.array(attention))
    # res = np.maximum(np.array(repr), np.array(attention))
    return [attention[:, :, i] for i in range(attention.shape[2])], png.transpose([1,2,0]), xy_img, argmax_xy_img, xy, argmax_xy


# Max diff ustawic na 2 lub 3 ?
# Bilinear w innym miejscu ?
# Zrobić "zwykly" attention do oszacowania img_change, po to zeby suma "map" dodawala sie do img_change ?
# [DONE] Cos ten softargmax nie dziala tak jak powinien (porownaj z argmax)
# Silhuette sum loss moglby byc lepszy, gdyby uzyc KL divergence ?
# [Fixed] Silhuette is not robust because img_change is calculated wrong ?
# Softargmax nadal nie dziala tak jak powinien. (wydawalo sie wczesniej ze to BatchNorm powoduje problemy)



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


    run_id = '{now:%Y-%m-%d-%H-%M-%S}'.format(now=datetime.datetime.now())

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_path = '../atari-objects-observations/'
    eval_path = os.path.join('../atari-objects-evaluations/', run_id)
    models_path = os.path.join(eval_path, "models")
    os.makedirs(eval_path)
    os.makedirs(models_path)
    os.system("cp -fr . {}".format(os.path.join(data_path, "code")))
    with open(os.path.join(data_path, "cmd.bash"), "w") as f:
        f.write(__file__)
        f.write(sys.argv)

    n_keypoints = 16

    epoch_size = 25
    batch_size = 64
    train_dataset = Dataset(
        root=data_path,
        n_games=10000,
        min_diff=1,
        max_diff=2,
        epoch_size=batch_size * epoch_size
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=False)

    def img_diff(second, first):
        return (1.0 + second - first) / 2


    classification = False
    model = Net(n_input_channels=3, n_hidden_channels=n_keypoints).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    last_loss = None
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        test(epoch, model, device, train_dataset, eval_path=eval_path, T=1)
        epoch_loss = train(args, classification, model, device, train_loader, optimizer, epoch)

        if not last_loss or last_loss > epoch_loss:
            last_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, os.path.join(models_path,
                            "model_{epoch:05d}_{epoch_loss:.3f}.pt".format(epoch=epoch, epoch_loss=epoch_loss)))



# Assumptions to be lifted in the future:
# 1. The objects do not disappear
# 2. The maximum distance to be predicted is limited (16)
# 3. The objects do not "jump" (e.g. in Pong after a score)
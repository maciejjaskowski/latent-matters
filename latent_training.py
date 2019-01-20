from __future__ import print_function

import cv2
import sys
import argparse
import datetime
import tqdm
import json

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
from PIL import Image, ImageDraw, ImageFont
import random
import os

class Net(nn.Module):
    def __init__(self, n_input_channels, n_hidden_channels, device, batch_size):
        super(Net, self).__init__()
        self._size = 84
        self.n_hidden_channels = n_hidden_channels
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(n_input_channels, 20, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv2d(20, 30, kernel_size=3, dilation=1, stride=1)

        self.bn = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, self.n_hidden_channels, kernel_size=3, dilation=1, stride=2)  # Modelujemy 3 obiekty


        # self.fc2_dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(1+self.n_hidden_channels, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.device = device

        self.ran = torch.Tensor([range(self._size)] * self._size).to(self.device)
        self.ran_y = torch.stack([torch.stack([self.ran] * n_hidden_channels, dim=0)] * self.batch_size, dim=0)
        self.ran_x = torch.stack([torch.stack([self.ran.t()] * n_hidden_channels, dim=0)] * self.batch_size, dim=0)

    def _softargmax(self, x4, T):
        softmaxed = F.softmax(T*x4.reshape([-1, x4.shape[1], x4.shape[2]*x4.shape[3]]), dim=-1).reshape(x4.shape)  #, _stacklevel=5)

        y_v = (softmaxed * self.ran_y[:softmaxed.shape[0]]).sum(dim=[2,3])
        x_v = (softmaxed * self.ran_x[:softmaxed.shape[0]]).sum(dim=[2, 3])
        return x_v, y_v, softmaxed

    def _encode(self, x, T):
        # print(x.shape)
        x2 = F.relu(self.conv1(x-0.33))
        x3 = F.relu(self.bn(self.conv2(x2)))


        dense_upscaled = F.interpolate(self.conv3(x3), size=(self._size, self._size), mode='bilinear', align_corners=False)
        x_v, y_v, softmaxed = self._softargmax(dense_upscaled, T=T)
        assert list(x_v.shape)[1:] == [self.n_hidden_channels], (list(x_v.shape)[1:], [self.n_hidden_channels])
        xy = torch.stack([x_v, y_v], dim=2)
        assert list(xy.shape[1:]) == [self.n_hidden_channels, 2], "{} != {}".format(list(xy.shape[1:]), [self.n_hidden_channels, 2])
        return xy, softmaxed, None, dense_upscaled


    def keypoints_variety_loss(self, keypoints1):
        assert len(keypoints1.shape) == 3, keypoints1.shape
        delta = 10.0
        keypoint_variety_loss = torch.Tensor([0.0]).to(self.device)[0]

        batch_size = keypoints1.shape[0]

        for i in range(keypoints1.shape[1]):
            for j in range(keypoints1.shape[1]):
                if i != j:
                    cur = torch.max(delta ** 2 - torch.sum((keypoints1[:, i, :] - keypoints1[:, j, :]) ** 2, dim=1),
                                    torch.Tensor([0.0]*batch_size).to(self.device))
                    # print(i,j,cur)
                    keypoint_variety_loss += cur.mean()
        return keypoint_variety_loss / (keypoints1.shape[1] ** 2)

    def keypoints_variety_loss_slow(self, keypoints1):
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

    def silhuette_variance_loss_slow(self, softmaxed, x, y):
        assert len(softmaxed.shape) == 4, softmaxed
        assert len(x.shape) == 2, len(x.shape)
        assert len(y.shape) == 2, len(y.shape)
        variance = []
        for b in range(softmaxed.shape[0]):
            for k in range(softmaxed.shape[1]):
                print(b,k)
                mul = softmaxed[b,k] * ((self.ran.t() - x[b,k])**2 + (self.ran - y[b,k])**2)
                v = torch.sum(mul)
                # if b == 0 and k == 0 and random.randint(0,5) == 0:
                #     print(b, k, v.item())
                # print(softmaxed, mul, b,k,v)
                variance.append(v)

        return torch.mean(v)

    def silhuette_variance_loss(self, softmaxed, x, y):
        assert len(softmaxed.shape) == 4, softmaxed
        assert len(x.shape) == 2, len(x.shape)
        assert len(y.shape) == 2, len(y.shape)
        assert x.shape[1] == softmaxed.shape[1], "{} != {}".format(x.shape[1], softmaxed.shape[1])

        mul = softmaxed * ((self.ran_x[:softmaxed.shape[0]] - x.expand([self._size, self._size, self.batch_size,softmaxed.shape[1]]).permute(2,3,0,1))**2 +
                           (self.ran_y[:softmaxed.shape[0]] - y.expand([self._size, self._size, self.batch_size,softmaxed.shape[1]]).permute(2,3,0,1))**2)
        v = torch.sum(mul, dim=[2, 3])

        return torch.mean(v)

    def silhuette_consistency_loss(self, keypoints1, map1, img_change):
        assert len(keypoints1.shape) == 3
        assert len(map1.shape) == 4
        assert len(img_change.shape) == 3
        eps = 1e-7
        # print(keypoints1)
        if torch.mean(img_change) == 0.0:
            return torch.Tensor([0.0])[0].to(self.device)

        img_change = img_change.expand([keypoints1.shape[1],] + list(img_change.shape)).permute(1,0,2,3)
        img_change_exists = (torch.mean(img_change, dim=[2, 3]) > 0).float()
        # print("XXX", map1.shape, img_change.shape)
        res = -torch.log(eps + torch.sum(map1 * img_change, dim=[2, 3])) * img_change_exists

        return torch.mean(res)

    @staticmethod
    def img_change(X):
        return (torch.sum(torch.abs(X['first_prev'] - X['first']), dim=1) > 0).float()

    def forward(self, X):
        assert list(X['first'].shape[1:]) == [3, self._size, self._size], X['first'].shape
        T = 1.0
        # print(X)
        # z  = self._encode(X['first'], T=T)
        # print("XX", len(z))
        keypoints1, map1, _, _ = self._encode(X['first'], T=T)
        keypoints1_prev, _, _, _ = self._encode(X['first_prev'], T=T)
        keypoints2, map2, _, _ = self._encode(X['second'], T=T)

        _img_change = self.img_change(X)


        # print("img_change", img_change.shape)
        # print("keypoints", keypoints1.shape)

        silhuette_variance_loss = self.silhuette_variance_loss(map1, x=keypoints1[:,:,0], y=keypoints1[:,:,1])
        keypoint_variety_loss = self.keypoints_variety_loss(keypoints1)
        silhuette_consistency_loss = self.silhuette_consistency_loss(keypoints1=keypoints1, map1=map1,
                                                                     img_change=_img_change)


        return {"keypoints1": keypoints1,
                "keypoints2": keypoints2,
                "keypoints1_prev": keypoints1_prev,
                "map1": map1,
                # "map2": map2,
                "img_change": _img_change,
                "keypoint_variety_loss": keypoint_variety_loss,
                "silhuette_consistency_loss": silhuette_consistency_loss,
                "silhuette_variance_loss": silhuette_variance_loss}


def move_loss_slow(keypoints2, target, keypoints1, keypoints1_prev):
    move_loss = []
    for b in range(keypoints1.shape[0]):
        for k in range(keypoints1.shape[1]):
            move_loss.append(
                ((keypoints1[b, k] - keypoints1_prev[b, k]) * (target[b] + 1) - (
                  keypoints2[b, k] - keypoints1_prev[b, k])) ** 2)

    return torch.mean(torch.stack(move_loss))


def move_loss(keypoints2, target, keypoints1, keypoints1_prev):
    target = target.expand([keypoints2.shape[1], 2, keypoints2.shape[0]]).permute(2, 0, 1)
    res = ((keypoints1 - keypoints1_prev) * (target + 1) - (keypoints2 - keypoints1_prev)) ** 2
    return torch.mean(res)


def log_scalars(epoch,  scalars, log_file):
    with open(log_file, "a") as f:
        for k, v in scalars.items():
            log(epoch, k, v, f)


def log(epoch, key, value, f):
    assert type(key) == str and not ":" in key
    f.write(json.dumps(dict(epoch=epoch, key=key, value=value)))
    f.write("\n")


def train(model, device, train_loader, optimizer, epoch, alpha, log_scalars, log_iter_time=False):

    model.train()
    epoch_losses = []
    keypoint_variety_losses = []
    silhuette_variance_losses = []
    silhuette_consistency_losses = []
    # silhuette_sum_losses = []

    if log_iter_time:
        log_iter_time = tqdm.tqdm
    else:
        log_iter_time = lambda x: x

    for batch_idx, (data, target) in log_iter_time(enumerate(train_loader)):
        data = {key: d.to(device) for key, d in data.items()}
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)

        a_move_loss = alpha.get('move_loss', 1.0) * move_loss(keypoints2=output['keypoints2'], target=target,
                                                            keypoints1=output['keypoints1'],
                                                            keypoints1_prev=output['keypoints1_prev'])

        # Chce zeby keypoint_variaty loss nie bylo duzy, ale gwaltownie rosl
        keypoint_variety_loss = alpha.get('keypoint_variety_loss', 1.0) * output['keypoint_variety_loss']
        # silhuette_sum_loss = alpha.get('silhuette_sum_loss', 1.0) * output['silhuette_sum_loss']
        silhuette_variance_loss = alpha.get('silhuette_variance_loss', 1.0) * output['silhuette_variance_loss']
        silhuette_consistency_loss = alpha.get('silhuette_consistency_loss', 1.0) * output['silhuette_consistency_loss']

        loss = silhuette_consistency_loss + silhuette_variance_loss + keypoint_variety_loss + a_move_loss
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        keypoint_variety_losses.append(keypoint_variety_loss.item())
        # silhuette_sum_losses.append(silhuette_sum_loss.item())
        silhuette_consistency_losses.append(silhuette_consistency_loss.item())
        silhuette_variance_losses.append(silhuette_variance_loss.item())
        del loss


    epoch_loss = np.mean(epoch_losses)
    keypoint_variety_loss = np.mean(keypoint_variety_losses)
    # silhuette_sum_loss = np.mean(silhuette_sum_losses)
    silhuette_variance_loss = np.mean(silhuette_variance_losses)
    silhuette_consistency_loss = np.mean(silhuette_consistency_losses)
    losses = dict(epoch_loss=epoch_loss,
               key_var=keypoint_variety_loss,
               # silh_sum=silhuette_sum_loss,
               silh_var=silhuette_variance_loss,
               silh_cons_loss=silhuette_consistency_loss,
               move_loss=a_move_loss.item())

    print('Train Epoch: {} \tLoss: {epoch_loss:.3f} '
          'key_var: {key_var:.3f} '
          'silh_var: {silh_var:.3f} '
          'silh_cons: {silh_cons_loss:.3f} '
          'move: {move_loss:.3f}'.format(
        epoch, **losses
    ))
    log_scalars(epoch=epoch, scalars=losses)

    return epoch_loss


def test(epoch, model, device, train_dataset, eval_path, T, n_games):
    print("Test epoch {}".format(epoch))
    model.eval()
    test_loss = 0
    correct = 0
    epoch_path = os.path.join(eval_path, 'epoch{:05d}'.format(epoch))
    os.makedirs(epoch_path)
    with torch.no_grad():
        for i_game in range(n_games):
            game = train_dataset.game_data[i_game]
            for i_obs in range(len(game)-1):
                sample, _ = train_dataset.get_sample(0, i_obs, i_obs + 1)
                png = sample['first'].to(device)
                torch_xy, torch_softmaxed, _, dense_upscaled = model._encode(png.unsqueeze(0), T=T)
                attentions, png, xy_img, argmax_xy_img, xy, argmax_xy = single_image(png=png, xy=torch_xy, softmaxed=torch_softmaxed, dense_upscaled=dense_upscaled)

                silhuette_variance_loss = model.silhuette_variance_loss(torch_softmaxed, x=torch_xy[:, :, 0], y=torch_xy[:, :, 1])
                keypoint_variety_loss = model.keypoints_variety_loss(torch_xy[:, :, :])
                # print("S", sample['first'].shape, sample['first_prev'].shape)
                # print("X",model.img_change(X=sample).shape)
                silhuette_consistency_loss = model.silhuette_consistency_loss(torch_xy, torch_softmaxed,
                                                                              model.img_change(X={
                                                                                  'first': sample['first'].unsqueeze(0).to(device),
                                                                                  'first_prev': sample[
                                                                                      'first_prev'].unsqueeze(0).to(device)}))
                if i_obs == 40:
                    print("xy[40]", xy[0, :, :])

                    print("silh_var", silhuette_variance_loss.item())
                    print("silh_con", silhuette_consistency_loss.item())
                    print("keyp_var", keypoint_variety_loss.item())

                for i_attention, attention in enumerate(attentions):
                    Image.fromarray(np.uint8(np.clip(attention*255, a_min=0, a_max=255))).resize((320, 320)).save(
                        os.path.join(epoch_path, 'game{i_game}_att_{i_attention:03d}_{i_obs:05d}.png'.format(i_game=i_game, i_attention=i_attention, i_obs=i_obs)))
                attentions = np.stack([np.clip(np.sum(np.stack(attentions, axis=2), axis=2), a_min=0.0, a_max=1.0)]*3, axis=2)
                attention_png = cv2.resize(np.uint8(attentions * 127 + png * 127), (320, 320))
                Image.fromarray(attention_png).save(os.path.join(epoch_path, 'game{i_game}_attentions_png_{i_obs:05d}.png'.format(i_game=i_game, i_obs=i_obs)))
                game0_xy = cv2.resize(np.uint8(xy_img*127 + png * 127), (320, 320))
                game0_xy_with_losses = np.zeros((320, 640, 3), dtype=np.uint8) + 255
                game0_xy_with_losses[:, :320, :] = game0_xy
                game0_xy_with_losses = Image.fromarray(game0_xy_with_losses)
                game0_xy_with_losses_draw = ImageDraw.Draw(game0_xy_with_losses)
                font = ImageFont.truetype("DejaVuSans.ttf", 15)
                pad = 20
                # print("Z", silhuette_consistency_loss)
                for i_text, text in enumerate(["silh_var {:.3f}".format(silhuette_variance_loss.item()),
                                               "silh_con {:.3f}".format(silhuette_consistency_loss.item()),
                                               "keyp_var {:.3f}".format(keypoint_variety_loss.item())]):
                    game0_xy_with_losses_draw.text((320, i_text * pad), text, font=font, fill=(0,0,0,128))



                game0_xy_with_losses.save(os.path.join(epoch_path, 'game{i_game}_xy_png_{i_obs:05d}.png'.format(i_game=i_game, i_obs=i_obs)))
                Image.fromarray(np.uint8(argmax_xy_img * 127 + png * 127)).resize((320, 320)).save(
                    os.path.join(epoch_path, 'game{i_game}_argmaxxy_png_{i_obs:05d}.png'.format(i_game=i_game, i_obs=i_obs)))

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
        img[np.round(x[0, i]).astype(np.int32), np.round(y[0, i]).astype(np.int32), :] = colours[i]
    return img


def single_image(png, xy, softmaxed, dense_upscaled):

    png = png.detach().cpu().numpy().copy()
    xy = xy.detach().cpu().numpy().copy()
    dense_upscaled = dense_upscaled.detach().cpu().numpy().copy()
    softmaxed = softmaxed.detach().cpu().numpy().copy()

    img_shape = dense_upscaled.shape[2:4]
    n_keypoints = softmaxed.shape[1]
    softmaxed_normed = softmaxed[0, :, :, :].transpose([1, 2, 0])
    argmax_xy = np.stack([np.unravel_index(np.argmax(dense_upscaled[0][i]), img_shape) for i in range(n_keypoints)])
    argmax_xy = argmax_xy.reshape((1, n_keypoints, 2))
    eps = 1e-7
    for i in range(softmaxed_normed.shape[2]):
        softmaxed_normed[:, :, i] = softmaxed_normed[:,:,i] / (softmaxed_normed[:, :, i].max() + eps)
    attention = cv2.resize(softmaxed_normed, img_shape)
    xy_img = add_points(np.zeros(img_shape + (3,), np.uint8), xy[:, :, 0], xy[:, :, 1])
    argmax_xy_img = add_points(np.zeros(img_shape + (3,), np.uint8), argmax_xy[:, :, 0], argmax_xy[:, :, 1])
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
    parser.add_argument('--n-data-loader-workers', type=int, default=5)
    parser.add_argument('--log-iter-time', action="store_true")
    parser.add_argument('--suffix', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--n-keypoints', type=int, required=True)
    parser.add_argument('--max-diff', type=int, required=True)
    parser.add_argument('--alpha', action='append')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--observations-dir', required=True)
    parser.add_argument('--load-model', default=None, help='Use this model instead of creating a model from scratch.')
    parser.add_argument('--test-only', action="store_true", help="If used, only a single test is performed.")
    parser.add_argument('--test-n-games', type=int, default=1, help="How many games are to be 'played' while testing.")
    args = parser.parse_args()



    run_id = '{now:%Y-%m-%d-%H-%M-%S}-{suffix}'.format(now=datetime.datetime.now(), suffix=args.suffix)
    print("Run id {}".format(run_id))

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    torch.set_num_threads(8)
    device = torch.device("cuda" if use_cuda else "cpu")
    alpha = dict([a.split("=") for a in args.alpha])
    alpha = {k: torch.Tensor([float(v)]).to(device)[0] for k, v in alpha.items()}

    data_path = args.observations_dir
    eval_path = os.path.join('../atari-objects-evaluations/', run_id)
    scalars_log_filename = os.path.join(eval_path, 'logs')
    models_path = os.path.join(eval_path, "models")
    os.makedirs(eval_path)
    os.makedirs(models_path)
    os.system("cp -fr . {}".format(os.path.join(eval_path, "code")))
    with open(os.path.join(data_path, "cmd.bash"), "w") as f:
        f.write(__file__)
        f.write(" ".join(sys.argv))


    n_keypoints = args.n_keypoints

    epoch_size = 25
    batch_size = args.batch_size
    train_dataset = Dataset(
        root=data_path,
        n_games=10000,
        min_diff=1,
        max_diff=args.max_diff,
        epoch_size=batch_size * epoch_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.n_data_loader_workers,
                              shuffle=False,  # It's shuffled anyway
                              pin_memory=use_cuda)

    def img_diff(second, first):
        return (1.0 + second - first) / 2


    classification = False
    model = Net(n_input_channels=3, n_hidden_channels=n_keypoints, device=device, batch_size=batch_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.load_model:
        loaded = torch.load(args.load_model)
        print("Loading model {} with last loss {}".format(args.load_model, loaded['loss']))
        model.load_state_dict(loaded['model_state_dict'])
        epoch = loaded['epoch']
        optimizer.load_state_dict(loaded['optimizer_state_dict'])

    if args.test_only:
        test(epoch, model, device, train_dataset, eval_path=eval_path, T=1, n_games=args.test_n_games)
        sys.exit(0)

    last_loss = None
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        test(epoch, model, device, train_dataset, eval_path=eval_path, T=1, n_games=args.test_n_games)
        epoch_loss = train(model, device, train_loader, optimizer, epoch, alpha=alpha,
                           log_scalars=lambda epoch, scalars: log_scalars(epoch, scalars, scalars_log_filename),
                           log_iter_time=args.log_iter_time)

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
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", required=True, type=int)
parser.add_argument("--games", required=True, type=int)
parser.add_argument('--keypoints', required=True, type=int)
args = parser.parse_args()

epoch = args.epoch


outs = []
for game in range(args.games):
    for att in range(args.keypoints):
        out = "epoch{epoch:05d}/game{game}_att_{att:03d}.mp4".format(game=game,epoch=epoch, att=att)
        os.system("ffmpeg -r 1 -i epoch{epoch:05d}/game{game}_att_{att:03d}_%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out};"
              .format(epoch=epoch, att=att, out=out, game=game))
        outs.append(out)

    out = "epoch{epoch:05d}/game{game}_attentions_png.mp4".format(game=game,epoch=epoch)
    os.system("ffmpeg -r 1 -i epoch{epoch:05d}/game{game}_attentions_png_%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out};"
          .format(epoch=epoch, out=out, game=game))
    outs.append(out)

    out = "epoch{epoch:05d}/game{game}_argmaxxy_png.mp4".format(game=game,epoch=epoch)
    os.system("ffmpeg -r 1 -i epoch{epoch:05d}/game{game}_argmaxxy_png_%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p;"
          .format(epoch=epoch, game=game))
    outs.append(out)

    out = "epoch{epoch:05d}/game{game}_xy_png.mp4".format(game=game,epoch=epoch)
    os.system("ffmpeg -r 1 -i epoch{epoch:05d}/game{game}_xy_png_%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out};"
          .format(epoch=epoch, out=out, game=game))
    outs.append(out)

    print(outs)
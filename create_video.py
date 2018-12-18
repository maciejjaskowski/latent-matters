import os
epoch = 218
outs = []
for att in range(6):
    out = "epoch{epoch:05d}_att_{att:03d}.mp4".format(epoch=epoch, att=att)
    os.system("ffmpeg -r 1 -i epoch{epoch:05d}_game0_att_{att:03d}_%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out};"
          .format(epoch=epoch, att=att, out=out))
    outs.append(out)

out = "epoch{epoch:05d}_attentions_png.mp4".format(epoch=epoch)
os.system("ffmpeg -r 1 -i epoch{epoch:05d}_game0_attentions_png_%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out};"
      .format(epoch=epoch, out=out))
outs.append(out)

out = "epoch{epoch:05d}_argmaxxy_png.mp4".format(epoch=epoch)
os.system("ffmpeg -r 1 -i epoch{epoch:05d}_game0_argmaxxy_png_%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p;"
      .format(epoch=epoch))
outs.append(out)

out = "epoch{epoch:05d}_xy_png.mp4".format(epoch=epoch)
os.system("ffmpeg -r 1 -i epoch{epoch:05d}_game0_xy_png_%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out};"
      .format(epoch=epoch, out=out))
outs.append(out)

print(outs)
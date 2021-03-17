# python 3.7
"""Semantic-Precise Image Editing."""

import sys, glob
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils

from lib.op import generate_images, bu
from lib.misc import imread
from lib.visualizer import get_label_color
from home.utils import color_mask, preprocess_image, preprocess_label, preprocess_mask
from manipulation.strategy import EditStrategy


class ImageEditing(object):
  """Functional interface for Semantic-Precise Image Editing."""

  @staticmethod
  def __sseg_se_z(P, G, z, tar, tar_mask, edit_strategy):
    """
    Precise semantic editing using semantic extractor.
    Args:
      P : The semantic extractor.
      G : The generator supporting the edit.
      z : The initial z to be edited.
      tar : The target semantic mask.
      tar_mask : Denoting user changed region. But not used currently.
    """
    edit_strategy.setup(z)
    z0 = edit_strategy.z0 # (14, 512)
    for i in (range(edit_strategy.n_iter)):
      z, wps = edit_strategy.to_std_form() # (14, 512)
      image, feature = G.synthesis(wps, generate_feature=True)
      seg = P(feature, size=tar.shape[2])[-1]
      celoss = torch.nn.functional.cross_entropy(seg, tar)
      regloss = 1e-3 * ((z-z0) ** 2).sum()
      priorloss = 1e-3 * (z ** 2).sum() / z.shape[0]
      edit_strategy.step(celoss + regloss + priorloss)
    return edit_strategy.to_std_form()

  @staticmethod
  def __sseg_pred_z(P, G, z, tar, tar_mask, edit_strategy):
    """
    Do editing on z (1, 512)
    """
    edit_strategy.setup(z)
    z0 = edit_strategy.z0
    for i in (range(edit_strategy.n_iter)):
      z, wps = edit_strategy.to_std_form()
      image = G.synthesis(wps)
      seg = P.raw_prediction(image, size=tar.shape[2])
      celoss = torch.nn.functional.cross_entropy(seg, tar)
      regloss = 1e-3 * ((z-z0) ** 2).sum()
      priorloss = 1e-3 * (z ** 2).sum() / z.shape[0]
      edit_strategy.step(celoss + regloss + priorloss)
    return edit_strategy.to_std_form()

  @staticmethod
  def __sseg_image_z(G, z, tar, tar_mask, edit_strategy, P=None):
    """
    Precise semantic editing using semantic extractor.
    Args:
      P : The semantic extractor.
      G : The generator supporting the edit.
      z : The initial z to be edited.
      tar : The target semantic mask.
      tar_mask : Denoting user changed region. But not used currently.
    """
    edit_strategy.setup(z)
    z0 = edit_strategy.z0
    for i in (range(edit_strategy.n_iter)):
      z, wps = edit_strategy.to_std_form()
      image = bu(G.synthesis(wps), tar.shape[2])
      diff = (tar - image) ** 2
      mseregloss = ((1 - tar_mask) * diff).sum() / (1 - tar_mask).sum()
      mseeditloss = (tar_mask * diff).sum() / tar_mask.sum()
      regloss = 1e-3 * ((z-z0) ** 2).sum()
      priorloss = 1e-3 * (z ** 2).sum() / z.shape[0]
      edit_strategy.step(mseregloss + mseeditloss + regloss + priorloss)
    return edit_strategy.to_std_form()

  @staticmethod
  def sseg_edit(G, z, tar, tar_mask,
                SE=None, 
                op="internal",
                latent_strategy='mixwp',
                optimizer="adam",
                n_iter=100,
                base_lr=0.01):
    """
    Args:
      G : The generator
      z : The latent z (N, 512), or mixed latent z (N, layers, 512)
      op : ``internal'' means to use LSE, ``external'' means to use predictor, ``image'' means to use color space editing loss
    """
    res = []
    edit_strategy = EditStrategy(G=G,
                                 latent_strategy=latent_strategy,
                                 optimizer=optimizer,
                                 n_iter=n_iter,
                                 base_lr=base_lr)
    func = {
      "internal" : ImageEditing.__sseg_se_z,
      "external" : ImageEditing.__sseg_pred_z,
      "image" : ImageEditing.__sseg_image_z}[op]
    for i in range(tar.shape[0]):
      res.append(func(
        P=SE, G=G, z=z[i:i+1].float(),
        tar=tar[i:i+1].long().cuda(),
        tar_mask=tar_mask[i:i+1].float().cuda(),
        edit_strategy=edit_strategy))
    agg = []
    for i in range(len(res[0])):
      agg.append(torch.cat([r[i] for r in res]))
    return agg

  @staticmethod
  def fuse_stroke(G, SE, P, wp,
      image_stroke, image_mask, label_stroke, label_mask):
    """
    Args:
      image_stroke : [3, H, W]
      image_mask : [1, H, W]
      label_stroke : [H, W]
      label_mask : [H, W]
    """
    size = label_mask.shape[1]
    image, feature = G.synthesis(wp, generate_feature=True)
    origin_image = bu(image, size=size).cpu()
    int_label = SE(feature, size=size)[-1].argmax(1).cpu() if SE else None
    ext_label = P(image, size=size) if P else None

    m = label_mask.cpu()
    fused_label = ((1 - m) * int_label + m * label_stroke.cpu()).long() \
      if label_stroke else None

    m = image_mask.cpu()
    fused_image = (1 - m) * origin_image + m * image_stroke.cpu() \
      if image_stroke else None

    return fused_image, fused_label, origin_image, int_label, ext_label

  @staticmethod
  def fuse_stroke_batch(G, SE, P, wps, image_strokes, image_masks, label_strokes, label_masks):
    """
    Args:
      zs : The
    """
    int_label = []
    ext_label = []
    fused_label = []
    fused_image = []
    origin_image = []
    with torch.no_grad():
      for i in range(zs.shape[0]):
        fused_image, fused_label, origin_image, int_label, ext_label
        res = fuse_stroke(G, SE, P, wps[i],
          image_strokes[i], image_masks[i], label_strokes[i], label_masks[i])
        int_label.append(est_label)
        ext_label.append(P(image, size=mask.shape[1])[0])
        tar = ((1 - mask) * est_label + mask * label_strokes[i:i+1]).long()
        fused_label.append(tar)
        mask = image_masks[i:i+1]
        img = (1 - mask) * image.cpu() + mask * image_strokes[i:i+1]
        fused_image.append(img)
    fused_image = torch.cat(fused_image).float()
    fused_label = torch.cat(fused_label)
    ext_label = torch.cat(ext_label)
    int_label = torch.cat(int_label)
    origin_image = torch.cat(origin_image)
    return origin_image, fused_image, fused_label, int_label, ext_label


def read_data(data_dir, name_list, n_class=15):
  z, image_stroke, label_stroke, image_mask, label_mask = [], [], [], [], []
  for name in name_list:
    name = name[:name.rfind("_")]
    files = glob.glob(f"{data_dir}/{name}*")
    files.sort()

    print(files)
    print(name)
    print(f"{data_dir}/{name}*")

    z.append(np.load(files[0])[0])

    img = imread(files[1]).transpose(2, 0, 1)
    image_stroke.append((img - 127.5) / 127.5)

    label_img = imread(files[2])
    t = np.zeros(label_img.shape[:2]).astype("uint8")
    for j in range(n_class):
      c = get_label_color(j)
      t[color_mask(label_img, c)] = j
    label_stroke.append(t)

    img = imread(files[3])
    image_mask.append((img[:, :, 0] > 127).astype("uint8"))

    img = imread(files[4])
    label_mask.append((img[:, :, 0] > 127).astype("uint8"))

  res = [z, image_stroke, image_mask, label_stroke, label_mask]
  return [torch.from_numpy(np.stack(r)) for r in res]


if __name__ == "__main__":
  import argparse
  from lib.misc import set_cuda_devices

  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', type=str, default='data/collect_ffhq',
    help='The input directory.')
  parser.add_argument('--name-list', type=str, default='figure/spie_list.txt',
    help='The list of names.')
  parser.add_argument('--out-dir', type=str, default='results/spie',
    help='The output directory.')
  parser.add_argument('--gpu-id', default='0',
    help='Which GPU(s) to use. (default: `0`)')
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)
  res = read_data(args.data_dir, args.name_list)
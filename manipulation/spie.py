# python 3.7
"""Semantic-Precise Image Editing."""

import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils

from lib.op import generate_images, bu


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
      seg = P(feature, size=tar.shape[2], last_only=True)[0][0]
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
        P=SE, G=G.net, z=z[i:i+1].float(),
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
      label_mask : [1, H, W]
    """
    print(image_stroke.shape, image_mask.shape, label_stroke.shape, label_mask.shape)
    size = label_mask.shape[1]
    image, feature = G.synthesis(wp, generate_feature=True)
    origin_image = bu(image, size=size).cpu()
    int_label = SE(feature, size=size)[-1].argmax(1).cpu() if SE else None
    ext_label = P(image, size=size) if P else None

    m = label_mask[0]
    fused_label = ((1 - m) * est_label + m * label_stroke).long()

    m = image_mask
    fused_image = (1 - m) * image.cpu() + m * image_stroke
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
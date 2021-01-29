# python 3.7
"""Semantic-Precise Image Editing."""

import sys
sys.path.insert(0, ".")
import os.path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils

import loss
from utils.op import standard_normal_logprob, generate_images, bu


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
      celoss = loss.FocalLoss()(seg, tar)
      #celoss = loss.mask_focal_loss(tar_mask, seg, tar)
      regloss = 1e-3 * ((z-z0) ** 2).sum()
      priorloss = 1e-3 * (z ** 2).sum() / z.shape[0]
      edit_strategy.step(celoss + regloss + priorloss)
      #vutils.save_image((image.clamp(-1, 1) + 1) / 2, f"{i}.png")
      #print(f"Iter={i} ce: {celoss:.3f} reg: {regloss:.3f} prior: {priorloss:.3f}")
    return edit_strategy.to_std_form()

  @staticmethod
  def __sseg_fewshotse_z(P, G, z, tar, tar_mask, edit_strategy):
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
      z, wps = edit_strategy.to_std_form()
      image, feature = G.synthesis(wps, generate_feature=True)
      seg = P(feature, size=tar.shape[2], last_only=True)[0][0]
      #celoss = loss.mask_focal_loss(tar_mask, seg, tar)
      celoss = loss.FocalLoss()(seg, tar)
      regloss = 1e-3 * ((z-z0) ** 2).sum()
      priorloss = 1e-3 * (z ** 2).sum() / z.shape[0]
      edit_strategy.step(celoss + regloss + priorloss)
      #if i % 10 == 0:
      #  vutils.save_image((image.clamp(-1, 1) + 1) / 2, f"{i}.png")
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
      celoss = loss.FocalLoss()(seg, tar)
      regloss = 1e-3 * ((z - z0) ** 2).sum() + 1e-5 * (z ** 2).sum()
      edit_strategy.step(celoss + regloss)
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
      regloss = 1e-3 * ((z-z0) ** 2).sum() + 1e-5 * (z ** 2).sum()
      edit_strategy.step(mseregloss + mseeditloss + regloss)
    return edit_strategy.to_std_form()

  @staticmethod
  def sseg_edit(G, z, tar, tar_mask,
                SE=None, 
                op="internal",
                latent_strategy='mixwp',
                optimizer="adam",
                n_iter=100,
                base_lr=0.01):
    res = []
    edit_strategy = EditStrategy(G=G,
                                 latent_strategy=latent_strategy,
                                 optimizer=optimizer,
                                 n_iter=n_iter,
                                 base_lr=base_lr)
    func = {
      "internal" : ImageEditing.__sseg_se_z,
      "external" : ImageEditing.__sseg_pred_z,
      "fewshot" : ImageEditing.__sseg_fewshotse_z,
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


class ConditionalSampling(object):
  @staticmethod
  def __sseg_se(SE, G, tar, n_init, edit_strategy, pred=False):
    """
    Conditional sampling with semantic extractor
    using random initialization.
    Args:
      SE : The semantic extractor, can be linear or nonlinear.
      G : The generator to be edited.
      tar : The target segmentation mask.
      n_init : The number of initialized z.
      n_iter : The total optimization number.
      ls : currently fixed to mixwp, which is to use a different z for each layer.
    """

    def get_seg(wp, size):
      if pred:
        image = G.synthesis(wp)
        seg = SE.raw_prediction(image, size=size)
        return seg
      image, feature = G.synthesis(wp, generate_feature=True)
      seg = SE(feature, size=size, last_only=True)[0][0]
      return seg

    BS = 4
    interval = 10
    z = torch.randn(n_init, 1, 512)
    scores = []
    for i in range(n_init):
      with torch.no_grad():
        w = G.mapping(z[i].cuda()) # (1, 512)
        wp = w.unsqueeze(1).repeat(1, G.num_layers, 1) # (1, 14, 512)
        seg = get_seg(wp, tar.shape[2])
        label = seg.argmax(1)
        scores.append((label == tar).sum())
    indice = np.argsort(scores)
    z_ = z[indice[:n_init]].clone()
    #print(f"=> Finding init. best: {scores[indice[-1]]:.3f} worst: {scores[indice[0]]:.3f}")
    best_ind = indice[-1]

    z = z[best_ind] # (1, 512)
    z0 = z.clone().detach().cuda()
    edit_strategy.setup(z)
    for i in range(edit_strategy.n_iter):
      z, wp = edit_strategy.to_std_form()
      seg = get_seg(wp, tar.shape[2])
      celoss = loss.FocalLoss()(seg, tar)
      regloss = 1e-3 * ((z-z0) ** 2).sum() / z.shape[0]
      edit_strategy.step(celoss + regloss)
    #print(f"=> Final loss: {celoss + regloss:.3f}")
    return z, wp


  @staticmethod
  def sseg_se(SE, G, tar, pred=False, n_init=10, repeat=1,
              latent_strategy='mixwp',
              optimizer="adam",
              n_iter=100,
              base_lr=0.01):
    edit_strategy = EditStrategy(G=G,
                                 latent_strategy=latent_strategy,
                                 optimizer=optimizer,
                                 n_iter=n_iter,
                                 base_lr=base_lr)

    res = []
    for i in tqdm(range(tar.shape[0])):
      for j in range(repeat):
        res.append(ConditionalSampling.__sseg_se(SE, G.net,
          tar[i:i+1].cuda(), n_init, edit_strategy, pred))
    agg = []
    for i in range(len(res[0])):
      agg.append(torch.cat([r[i] for r in res]))
    return agg

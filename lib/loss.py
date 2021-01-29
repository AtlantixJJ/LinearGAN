import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import op

def mask_cross_entropy_loss(mask, x, y): # requires more than editing need
    ce = F.cross_entropy(x, y, reduction="none")
    return (mask * ce).sum() / mask.sum()


def mask_focal_loss(mask, x, y): # requires more than editing need
    ce = FocalLoss()(x, y, reduction="none")
    return (mask * ce).sum() / mask.sum()


class FocalLoss(nn.Module):
  def __init__(self, binary=False, alpha=1, gamma=2):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma # 指数
    if binary:
      self.func = F.binary_cross_entropy_with_logits
    else:
      self.func = F.cross_entropy

  def forward(self, inputs, targets, reduction="mean"):
    loss = self.func(inputs, targets, reduction='none')
    pt = torch.exp(-loss)
    focal_loss = self.alpha * ((1 - pt).pow(self.gamma) * loss)
    if reduction == "mean":
      return focal_loss.mean()
    else:
      return focal_loss


def int2onehot(x, n):
    z = torch.zeros(x.shape[0], n, x.shape[2], x.shape[3]).to(x.device)
    #for i in range(n):
    #  z[:, i:i+1][x == n] = 1
    #return z
    return z.scatter_(1, x, 1)

def segloss_final(segs, label, loss_fn_layer, loss_fn_final):
  N = len(segs[0])
  seglosses = []
  for cat_id in range(label.shape[0]):
    segloss = []
    # layer loss
    for i in range(N):
      seg = segs[cat_id][i]
      segloss.append(loss_fn_final(seg if seg.size(2) == label.size(3) \
        else op.bu(seg, label.size(3)), label[cat_id]))
    seglosses.append(segloss)
  return seglosses


def segloss(segs, label, loss_fn_layer, loss_fn_final):
  N = len(segs[0])
  seglosses = []
  for cat_id in range(label.shape[0]):
    segloss = []
    onehot = int2onehot(label[cat_id].unsqueeze(1), segs[cat_id][0].shape[1])
    # BCE loss
    for i in range(N):
      seg = segs[cat_id][i]
      segloss.append(loss_fn_layer(seg if seg.size(2) == label.size(3) \
        else op.bu(seg, label.size(3)), onehot))
    # CE loss
    final = segs[cat_id][-1]
    segloss.append(loss_fn_final(final if final.size(2) == label.size(3) \
      else op.bu(final, label.size(3)), label[cat_id]))
    seglosses.append(segloss)
  return seglosses


def perceptual_loss_generator(generator, perceptron, x, mask, wp):
  # Reconstruction loss.
  x_rec = generator.synthesis(wp)
  loss_pix = (mask * (x - x_rec) ** 2).sum(dim=[1, 2, 3]) \
    / mask.sum(dim=[1, 2, 3])

  # Perceptual loss.
  x_feat = perceptron(x)
  x_rec_feat = perceptron(x_rec)
  smask = F.interpolate(mask, size=x_feat.shape[2:], mode="bilinear")
  loss_feat = (smask * (x_feat - x_rec_feat) ** 2).sum(dim=[1, 2, 3]) \
    / smask.sum(dim=[1, 2, 3])
  #print(loss_pix.shape, loss_feat.shape)
  return loss_pix + loss_feat * 5e-5


def perceptual_loss(perceptron, x_rec, x, mask):
  # Reconstruction loss.
  loss_pix = (mask * (x - x_rec) ** 2).sum(dim=[1, 2, 3]) \
    / mask.sum(dim=[1, 2, 3])

  # Perceptual loss.
  x_feat = perceptron(x)
  x_rec_feat = perceptron(x_rec)
  smask = F.interpolate(mask, size=x_feat.shape[2:], mode="bilinear")
  loss_feat = (smask * (x_feat - x_rec_feat) ** 2).sum(dim=[1, 2, 3]) \
    / smask.sum(dim=[1, 2, 3])
  #print(loss_pix.shape, loss_feat.shape)
  return loss_pix + loss_feat * 5e-5

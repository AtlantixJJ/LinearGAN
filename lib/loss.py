import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import op


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


def segloss(segs, label, loss_fn):
  """The final version of loss."""
  segloss = []
  size = label.size(2)
  for seg in segs:
    seg = op.bu(seg, size) if seg.size(2) != size else seg
    segloss.append(loss_fn(seg, label))
  return segloss


def segloss_bce(segs, label, loss_fn_layer, loss_fn_final):
  """Use BCE for each layer. It is slow and CPU intensive."""
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
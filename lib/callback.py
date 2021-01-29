import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from .visualizer import segviz_torch
from .op import *


class CheckpointCallback(pl.Callback):
  def __init__(self, ckpt_func, interval):
    self.func = ckpt_func
    self.interval = interval
  
  def on_batch_end(self, trainer, pl_module):
    if trainer.global_step % self.interval == 0:
      print(f"=> Saving model on {trainer.global_step}")
      self.func(pl_module)


class WeightVisualizerCallback(pl.Callback):
  def __init__(self, size=200, interval=100):
    self.history = []
    self.size = size
    self.count = 0
    self.interval = interval

  def on_batch_end(self, trainer, pl_module):
    self.count += 1
    if self.count % self.interval != 0:
      return
    weight = pl_module.model._calc_layer_weight().detach().cpu().numpy()
    self.history.append(weight)
    if len(self.history) > self.size:
      del self.history[0]
    
    N, M = len(self.history), weight.shape[1]
    data = np.stack(self.history)
    cdata = np.cumsum(data, 2)
    for i in range(weight.shape[0]):
      fig = plt.figure(figsize=(10, 5))
      for j in range(M):
        plt.bar(range(N), data[:, i, j],
          bottom=None if j == 0 else cdata[:, i, j - 1])
      plt.legend([f"layer {j}" for j in range(M)])
      trainer.logger.experiment.add_figure(f"cat{i}/layer weight",
        fig, self.count // self.interval)
      plt.close()


class ImageVisualizerCallback(pl.Callback):
  def __init__(self, z):
    super().__init__()
    self.z = z
    self.shape = self.z.shape
    self.count = 0
  
  def on_batch_end(self, trainer, pl_module):
    self.count += 1
    if self.count % 1000 != 0:
      return
    tensorboard = trainer.logger.experiment
    with torch.no_grad():
      r, _ = pl_module(self.z.view(*self.shape), reverse=True)
    if r.shape[1] == 1:
      r = r.repeat(1, 3, 1, 1)
    disp_images = vutils.make_grid(r, nrow=4)
    tensorboard.add_image('generated samples',
      disp_images.clamp(0, 1), global_step=self.count // 1000)


def get_images_SE(G, SE, P, z):
  with torch.no_grad():
    if hasattr(G, "mapping"):
      wp = G.truncation(G.mapping(z))
      image, feature = G.synthesis(wp, generate_feature=True)
    else:
      image, feature = G(z, generate_feature=True)
    label = P(image).long() # (cat, N, H, W)
  seg = SE(feature) # [[], [], []] (category groups, sery)
  return image, seg, label


def viz_SE(G, SE, P, z):
  images = []
  layer_images = []
  seg_vizs = []
  label_vizs = []
  for i in range(z.shape[0]): # batch
    image, segs, label = get_images_SE(G, SE, P, z[i:i+1])
    images.append(image)
    for j in range(label.shape[0]): # cat
      if i == 0:
        for k in range(len(segs[j])):
          seg_label = bu(segs[j][k], 256)[0].argmax(0)
          layer_images.append(segviz_torch(seg_label))
      seg_label = segs[j][-1][0].argmax(0) # only visualize final output
      seg_vizs.append(segviz_torch(seg_label))
      label_vizs.append(segviz_torch(label[j, 0]))
  images = (torch.cat(images).clamp(-1, 1) + 1) / 2
  img = torch.stack(layer_images)
  return images.cpu(), torch.stack(seg_vizs), torch.stack(label_vizs), img


class TrainingEvaluationCallback(pl.Callback):
  def __init__(self):
    super().__init__()
    self.count = 0
    self.vals = []
  
  def on_epoch_end(self, trainer, pl_module):
    self.count += 1
    tensorboard = trainer.logger.experiment
    table = pl_module.train_evaluation
    labels = pl_module.P.labels
    pixelacc = torch.Tensor([entry[0] for entry in table])
    pixelacc = pixelacc.mean()
    IoU = torch.stack([entry[1] for entry in table], 1) # IoU: (C, N)
    c_IoU = torch.zeros((IoU.shape[0]))
    for i in range(IoU.shape[0]):
      v = IoU[IoU > -0.1]
      c_IoU[i] = -1 if len(v) == 0 else v.mean()
      tensorboard.add_scalar(f'val/{i:02d}_{labels[i]}_IoU',
        c_IoU[i], self.count)
    mIoU = c_IoU[c_IoU > -1].mean()
    tensorboard.add_scalar('val/mIoU', mIoU, self.count)
    tensorboard.add_scalar('val/pixelacc', pixelacc, self.count)
    if pl_module.best_val < mIoU: # manual checkpoint
      print(f"=> Updating best model from {pl_module.best_val:3f} to {mIoU:3f}")
      pl_module.best_val = mIoU
      save_to_pth(pl_module.model, pl_module.save_dir)
    self.vals.append([mIoU, pixelacc, c_IoU])
    torch.save(self.vals, pl_module.save_dir + "/train_evaluation.pth")
    pl_module.train_evaluation = []


class SEVisualizerCallback(pl.Callback):
  def __init__(self, z, interval=1000):
    super().__init__()
    self.z = z
    self.interval = interval
    self.count = 0

  def on_batch_end(self, trainer, pl_module):
    self.count += 1
    if self.count % self.interval != 0:
      return

    tensorboard = trainer.logger.experiment
    images, seg_vizs, label_vizs, layer_images = viz_SE(pl_module.G,
      pl_module.model, pl_module.P, self.z)
    if images.size(3) > 256:
      images, seg_vizs, label_vizs, layer_images = bu([
        images, seg_vizs, label_vizs, layer_images], 256)
    disp = vutils.make_grid(torch.cat([images, seg_vizs, label_vizs]),
      nrow=images.shape[0])
    tensorboard.add_image('semantic extraction visualization',
      disp, global_step=self.count // self.interval)
    disp_layer = vutils.make_grid(torch.cat([
      layer_images, images[0:1], label_vizs[0:1]]), nrow=4)
    tensorboard.add_image('layer semantics',
      disp_layer, global_step=self.count // self.interval)    

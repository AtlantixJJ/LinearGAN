import sys, os, torch
sys.path.insert(0, ".")
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from lib import op, loss
from pytorch_lightning.metrics.functional import iou, precision_recall


class SEFewShotLearner(pl.LightningModule):
  def __init__(self, model, G,
    optim='adam-0.001', # optimizer type and initial learning rate
    loss_type='focal', # focal / normal
    latent_strategy='notrunc-mixwp',
    resolution=512, # (pseudo)training label resolution
    save_dir='expr'):
    super().__init__()
    self.model = model
    self.latent_strategy = latent_strategy
    self.resolution = resolution
    self.save_dir = save_dir
    if loss_type == 'focal':
      self.loss_fn_layer = loss.FocalLoss(True)
      self.loss_fn_final = loss.FocalLoss(False)
    else:
      self.loss_fn_layer = F.binary_cross_entropy_with_logits
      self.loss_fn_final = F.cross_entropy
    self.G = G
    self.best_val = 0
    self.optim_type, self.lr = optim.split("-")
    self.lr = float(self.lr)

  def get_wp(self, z):
    if self.latent_strategy == 'trunc-wp':
      wp = self.G.truncation(self.G.mapping(z))
    elif self.latent_strategy == 'notrunc-wp':
      wp = self.G.mapping(z).unsqueeze(1).repeat(1, self.G.num_layers, 1)
    elif self.latent_strategy == 'notrunc-mixwp':
      zs = torch.randn(z.shape[0] * self.G.num_layers, z.shape[1])
      zs = zs.to(z.device)
      wp = self.G.mapping(zs).view(z.shape[0], self.G.num_layers, -1)
    return wp
        
  def forward(self, z):
    self.G.eval()
    with torch.no_grad():
      if hasattr(self.G, "mapping"):
        wp = self.get_wp(z)
        image, feature = self.G.synthesis(wp, generate_feature=True)
      else:
        image, feature = self.G(z, generate_feature=True)
    # [[], [], []] (category groups, sery)
    seg = self.model(feature, size=self.resolution) 
    return image, seg

  def training_step(self, batch, batch_idx):
    """
    batch is dummy, batch_idx is used for gradient accumulation
    """
    idx = batch_idx % self.label.shape[0]
    feature = [f[idx:idx+1].cuda() for f in self.feature]
    _, segs = self.model(feature, size=self.resolution)
    seg = op.bu(segs[-1], self.label.size(2))
    segloss = self.loss_fn_final(seg, self.label[idx:idx+1])
    return segloss

  def configure_optimizers(self):
    self.adam_optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(self.adam_optim,
      [20], gamma=0.1)
    return [self.adam_optim], [scheduler]


class SELearner(pl.LightningModule):
  def __init__(self, model, G, P,
               lr=0.001, # optimizer type and initial learning rate
               loss_type='focal', # focal / normal
               latent_strategy='notrunc-mixwp',
               resolution=512, # (pseudo)training label resolution
               save_dir='expr'):
    super().__init__()
    self.model = model
    self.G = G
    self.P = P
    self.lr = lr
    self.latent_strategy = latent_strategy
    self.resolution = resolution
    self.save_dir = save_dir

    if loss_type == 'focal':
      self.loss_fn_layer = loss.FocalLoss(True)
      self.loss_fn_final = loss.FocalLoss(False)
    else:
      self.loss_fn_layer = F.binary_cross_entropy_with_logits
      self.loss_fn_final = F.cross_entropy
    
    self.reset()

  def reset(self):
    self.train_evaluation = []
    self.best_val = 0

  def get_wp(self, z):
    if self.latent_strategy == 'trunc-wp':
      wp = self.G.truncation(self.G.mapping(z))
    elif self.latent_strategy == 'notrunc-wp':
      wp = self.G.mapping(z).unsqueeze(1).repeat(1, self.G.num_layers, 1)
    elif self.latent_strategy == 'notrunc-mixwp':
      zs = torch.randn(z.shape[0] * self.G.num_layers, z.shape[1])
      zs = zs.to(z.device)
      wp = self.G.mapping(zs).view(z.shape[0], self.G.num_layers, -1)
    return wp

  def forward(self, z):
    self.G.eval()
    self.P.eval()
    with torch.no_grad():
      if hasattr(self.G, "mapping"):
        wp = self.get_wp(z)
        image, feature = self.G.synthesis(wp, generate_feature=True)
      else:
        image, feature = self.G(z, generate_feature=True)
      label = self.P(image, size=self.resolution).long()
    segs = self.model(feature, size=self.resolution) 
    return segs, label

  def training_step(self, batch, batch_idx):
    segs, label = self(batch)
    seg = op.bu(segs[-1], label.size(2))
    segloss = self.loss_fn_final(seg, label)

    #total_loss = 0
    #n_layers = len(segloss) - 1 # The last one is final segmentation
    #for i in range(n_layers + 1): # 0 ~ len(segloss) - 1
    #  layer = 'final' if i == n_layers else f'layer{i}'
    #  self.log(f'CE/{layer}', segloss[i])
    #  total_loss = total_loss + segloss[i]
    total_loss = segloss
    self.log("main/total", total_loss)

    dt = seg.argmax(1).detach()
    gt = label.detach()
    IoU = iou(dt, gt, num_classes=self.model.n_class,
      ignore_index=0, absent_score=-1, reduction='none')
    pixelacc = (dt == gt).sum() / float(dt.shape.numel())
    # pixelacc, mIoU, IoU
    self.train_evaluation.append([pixelacc, IoU])
    return total_loss

  def configure_optimizers(self):
    if hasattr(self.model, "layer_weight"):
      pg = [{'params': self.model.extractor.parameters(), 'lr': self.lr},
        {'params': [self.model.layer_weight], 'lr': 10 * self.lr}]
      self.adam_optim = torch.optim.Adam(pg)
    else:
      pg = [{'params' : self.model.parameters(), 'lr': self.lr}]
    self.adam_optim = torch.optim.Adam(pg)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(self.adam_optim, [20])
    return [self.adam_optim], [scheduler]


class SemanticExtractor(nn.Module):
  """
  Base class for semantic extractors
  """

  def __init__(self, n_class, dims, layers, type="", **kwargs):
    """
    Args:
      n_class : The number of semantic categories.
      dims : The dimension (depth) of each feature map.
      layers : The layer indice of the generator.
    """
    super().__init__()

    self.type = type
    self.n_class = n_class
    self.dims = dims
    self.layers = layers
    self.segments = [0] + list(np.cumsum(self.dims))
    self.build()

  def _index_feature(self, features, i):
    """"""
    l1 = len(self.layers)
    l2 = len(features)
    if l1 < l2:
      return features[self.layers[i]]
    elif l1 == l2:
      return features[i]
    else:
      print(f"!> Error: The length of layers ({l1}) != features ({l2})")

  def predict(self, stage):
    """Return a numpy array."""
    res = self.forward(stage, True)[0].argmax(1)
    return res.detach().cpu().numpy().astype("int32")

  def arch_info(self):
    """Return the architecture information dict."""
    return dict(
      n_class=self.n_class,
      dims=self.dims,
      layers=self.layers)

  def build(self):
    pass


class LSE(SemanticExtractor):
  """Extract the semantics from generator's feature maps using 1x1 convolution.
  """

  def __init__(self, lw_type="softplus", use_bias=False, **kwargs):
    """
    Args:
      lw_type : The layer weight type. Candidates are softplus, sigmoid, none.
      use_bias : default is not to use bias.
    """
    self.lw_type = lw_type
    self.use_bias = use_bias
    super().__init__(**kwargs)
    self.build()

  def build(self):
    """Build the architecture of LSE."""
    def conv_block(in_dim, out_dim):
      return nn.Conv2d(in_dim, out_dim, 1, bias=self.use_bias)

    self.extractor = nn.ModuleList([
      conv_block(dim, self.n_class) for dim in self.dims])

    self.layer_weight = nn.Parameter(torch.ones((len(self.layers),)))

  def arch_info(self):
    base_dic = SemanticExtractor.arch_info(self)
    base_dic["lw_type"]  = self.lw_type
    base_dic["use_bias"] = self.use_bias
    base_dic["type"]     = "LSE"
    return base_dic

  def _calc_layer_weight(self):
    if self.lw_type == "none" or self.lw_type == "direct":
      return self.layer_weight
    if self.lw_type == "softplus": 
      weight = torch.nn.functional.softplus(self.layer_weight)
    elif self.lw_type == "sigmoid":
      weight = torch.sigmoid(self.layer_weight)
    return weight / weight.sum()

  def forward(self, features, size=None):
    """Given a set of features, return the segmentation.

    Args:
      features : A list of feature maps. When len(features) > len(self.layers)
                 , it is assumed that the features if taken from all the layers
                 from the generator and will be selected here. Otherwise, it is
                 assumed that the features correspond to self.layers.
      size : The target output size. Bilinear resize will be used if this
             argument is specified.
    Returns:
      A list of segmentations corresponding to each layer, with the last one 
      being the final segmentation integrating all layers.
    """

    outputs = []
    for i in range(len(self.layers)):
      feat = self._index_feature(features, i)
      x = self.extractor[i](feat)
      outputs.append(x)

    # detect final output size, if not specified
    size = size if size else outputs[-1].shape[2:]
    layers = op.bu(outputs, size)

    weight = self._calc_layer_weight()
    if self.lw_type == "none":
      final = sum(layers)
    else:
      final = sum([r * w for r, w in zip(layers, weight)])
    outputs.append(final)
    return outputs


class NSE1(LSE):
  """A direct nonlinear generalization from LSE.
  """

  def __init__(self, lw_type="softplus", use_bias=True,
               ksize=1, n_layers=3, **kwargs):
    """
    Args:
      ksize : The convolution kernel size.
    """
    self.lw_type = lw_type
    self.use_bias = use_bias
    self.ksize = ksize
    self.n_layers = n_layers
    SemanticExtractor.__init__(self, **kwargs)
    self.build()
  
  def arch_info(self):
    base_dic = SemanticExtractor.arch_info(self)
    base_dic["ksize"]    = self.ksize
    base_dic["n_layers"] = self.n_layers
    base_dic["lw_type"]  = self.lw_type
    base_dic["use_bias"] = self.use_bias
    base_dic["type"]     = "NSE-1"
    return base_dic

  def build(self):
    def conv_block(in_dim, out_dim):
      midim = (in_dim + out_dim) // 2
      padding = {1 : 0, 3 : 1}[self.ksize]
      _m = []
      _m.append(nn.Conv2d(in_dim, midim, self.ksize, padding=padding))
      _m.append(nn.ReLU(inplace=True))

      for _ in range(self.n_layers - 2):
        _m.append(nn.Conv2d(midim, midim, self.ksize, padding=padding))
        _m.append(nn.ReLU(inplace=True))

      _m.append(nn.Conv2d(midim, out_dim, self.ksize, padding=padding))
      return nn.Sequential(*_m)

    self.extractor = nn.ModuleList([
      conv_block(dim, self.n_class)
        for dim in self.dims])

    # combining result from each layer
    self.layer_weight = nn.Parameter(torch.ones((len(self.dims),)))

  def forward(self, features, size=None):
    return super().forward(features, size)


class NSE2(SemanticExtractor):
  """A generator-like semantic extractor."""

  def __init__(self, ksize=3, **kwargs):
    """
      Args:
        ksize: kernel size of convolution
    """
    self.type = "NSE-2"
    self.ksize = ksize
    super().__init__(**kwargs)

  def arch_info(self):
    base_dic = SemanticExtractor.arch_info(self)
    base_dic["ksize"] = self.ksize
    base_dic["type"] = "NSE-2"

  def build(self):
    def conv_block(in_dim, out_dim):
      _m = [
        nn.Conv2d(in_dim, out_dim, self.ksize, 1, self.ksize // 2),
        nn.ReLU(inplace=True)]
      return nn.Sequential(*_m)

    # transform generative representation to semantic embedding
    self.extractor = nn.ModuleList([
      conv_block(dim, dim) for dim in self.dims])
    # learning residue between different layers
    self.reviser = nn.ModuleList([conv_block(prev, cur) \
      for prev, cur in zip(self.dims[:-1], self.dims[1:])])
    # transform semantic embedding to label
    self.visualizer = nn.Conv2d(self.dims[-1], self.n_class, self.ksize,
                                  padding=self.ksize // 2)
  
  def forward(self, features, size=None):
    for i in range(len(self.layers)):
      feat = self._index_feature(features, i)
      if i == 0:
        hidden = self.extractor[i](feat)
      else:
        if hidden.size(2) * 2 == feat.size(2):
          hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.reviser[i - 1](hidden)
        hidden = hidden + self.extractor[i](feat)
    x = self.visualizer(hidden)
    if size is not None and size != x.size(3):
      x = op.bu(x, size)
    return [x]


EXTRACTOR_POOL = {
  "LSE" : LSE,
  "NSE-1" : NSE1,
  "NSE-2" : NSE2
}

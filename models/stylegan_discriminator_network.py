# python 3.7
"""Contains the implementation of discriminator described in StyleGAN.

Different from the official tensorflow version in folder `stylegan_official`,
this is a simple pytorch version which only contains the discriminator part.
This class is specially used for inference.

For more details, please check the original paper:
https://arxiv.org/pdf/1812.04948.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['StyleGANDiscriminatorNet']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Fused-scale options allowed.
_FUSED_SCALE_OPTIONS_ALLOWED = [True, False, 'auto']

# Minimal resolution for `auto` fused-scale strategy.
_AUTO_FUSED_SCALE_MIN_RES = 128


class StyleGANDiscriminatorNet(nn.Module):
  """Defines the discriminator network in StyleGAN.

  NOTE: The discriminator takes images with `RGB` color channels and range
  [-1, 1] as inputs.
  """

  def __init__(self,
               resolution,
               image_channels=3,
               label_size=0,
               fused_scale='auto',
               minibatch_std_group_size=4,
               minibatch_std_num_channels=1,
               fmaps_base=16 << 10,
               fmaps_max=512):
    """Initializes the discriminator with basic settings.

    Args:
      resolution: The resolution of the input image.
      image_channels: Number of channels of the input image. (default: 3)
      label_size: Size of additional labels. (default: 0)
      fused_scale: Whether to fused `conv2d` and `downsample` together,
        resulting in `conv2d` with strides. (default: False)
      minibatch_std_group_size: Group size for the minibatch standard deviation
        layer. 0 means disable. (default: 4)
      minibatch_std_num_channels: Number of new channels after the minibatch
        standard deviation layer. (default: 1)
      fmaps_base: Base factor to compute number of feature maps for each layer.
        (default: 16 << 10)
      fmaps_max: Maximum number of feature maps in each layer. (default: 512)

    Raises:
      ValueError: If the input `resolution` is not supported, or `fused_scale`
        is not supported.
    """
    super().__init__()

    if resolution not in _RESOLUTIONS_ALLOWED:
      raise ValueError(f'Invalid resolution: {resolution}!\n'
                       f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
    if fused_scale not in _FUSED_SCALE_OPTIONS_ALLOWED:
      raise ValueError(f'Invalid fused-scale option: {fused_scale}!\n'
                       f'Options allowed: {_FUSED_SCALE_OPTIONS_ALLOWED}.')

    self.init_res = _INIT_RES
    self.init_res_log2 = int(np.log2(self.init_res))
    self.resolution = resolution
    self.final_res_log2 = int(np.log2(self.resolution))
    self.image_channels = image_channels
    self.label_size = label_size
    self.fused_scale = fused_scale
    self.minibatch_std_group_size = minibatch_std_group_size
    self.minibatch_std_num_channels = minibatch_std_num_channels
    self.fmaps_base = fmaps_base
    self.fmaps_max = fmaps_max

    # Level of detail (used for progressive training).
    self.lod = nn.Parameter(torch.zeros(()))
    self.pth_to_tf_var_mapping = {'lod': 'lod'}

    # pylint: disable=line-too-long
    for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
      res = 2 ** res_log2
      block_idx = self.final_res_log2 - res_log2

      # Input convolution layer for each resolution.
      self.add_module(
          f'input{block_idx}',
          ConvBlock(in_channels=self.image_channels,
                    out_channels=self.get_nf(res),
                    kernel_size=1,
                    padding=0))
      self.pth_to_tf_var_mapping[f'input{block_idx}.conv.weight'] = (
          f'FromRGB_lod{block_idx}/weight')
      self.pth_to_tf_var_mapping[f'input{block_idx}.wscale.bias'] = (
          f'FromRGB_lod{block_idx}/bias')

      # Convolution block for each resolution (except the last one).
      if res != self.init_res:
        if self.fused_scale == 'auto':
          fused_scale = (res >= _AUTO_FUSED_SCALE_MIN_RES)
        else:
          fused_scale = self.fused_scale
        self.add_module(
            f'layer{2 * block_idx}',
            ConvBlock(in_channels=self.get_nf(res),
                      out_channels=self.get_nf(res)))
        self.add_module(
            f'layer{2 * block_idx + 1}',
            ConvBlock(in_channels=self.get_nf(res),
                      out_channels=self.get_nf(res // 2),
                      downsample=True,
                      fused_scale=fused_scale))
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.conv.weight'] = (
            f'{res}x{res}/Conv0/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.wscale.bias'] = (
            f'{res}x{res}/Conv0/bias')
        if fused_scale:
          self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.weight'] = (
              f'{res}x{res}/Conv1_down/weight')
        else:
          self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.conv.weight'] = (
              f'{res}x{res}/Conv1_down/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.wscale.bias'] = (
            f'{res}x{res}/Conv1_down/bias')

      # Convolution block for last resolution.
      else:
        self.add_module(
            f'layer{2 * block_idx}',
            ConvBlock(
                in_channels=self.get_nf(res),
                out_channels=self.get_nf(res),
                minibatch_std=True,
                minibatch_std_group_size=self.minibatch_std_group_size,
                minibatch_std_num_channels=self.minibatch_std_num_channels))
        self.add_module(
            f'layer{2 * block_idx + 1}',
            DenseBlock(in_channels=self.get_nf(res) * res * res,
                       out_channels=self.get_nf(res // 2)))
        self.add_module(
            f'layer{2 * block_idx + 2}',
            DenseBlock(in_channels=self.get_nf(res // 2),
                       out_channels=max(self.label_size, 1),
                       wscale_gain=1.0,
                       activation_type='linear'))
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.conv.weight'] = (
            f'{res}x{res}/Conv/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.wscale.bias'] = (
            f'{res}x{res}/Conv/bias')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.fc.weight'] = (
            f'{res}x{res}/Dense0/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.wscale.bias'] = (
            f'{res}x{res}/Dense0/bias')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.fc.weight'] = (
            f'{res}x{res}/Dense1/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.wscale.bias'] = (
            f'{res}x{res}/Dense1/bias')
    # pylint: enable=line-too-long

    self.downsample = AveragePoolingLayer()

  def get_nf(self, res):
    """Gets number of feature maps according to current resolution."""
    return min(self.fmaps_base // res, self.fmaps_max)

  def forward(self, image, label=None):
    if image.ndim != 4 or image.shape[1:] != (
        self.image_channels, self.resolution, self.resolution):
      raise ValueError(f'The input image should be with shape [batch_size, '
                       f'channel, height, width], where '
                       f'`channel` equals to {self.image_channels}, '
                       f'`height` and `width` equal to {self.resolution}!\n'
                       f'But {image.shape} is received!')
    if self.label_size:
      if label is None:
        raise ValueError(f'Model requires an additional label (with size '
                         f'{self.label_size}) as inputs, but no label is '
                         f'received!')
      if label.ndim != 2 or label.shape != (image.shape[0], self.label_size):
        raise ValueError(f'Input labels should be with shape [batch_size, '
                         f'label_size], where `batch_size` equals to that of '
                         f'images ({image.shape[0]}) and `label_size` equals '
                         f'to {self.label_size}!\n'
                         f'But {label.shape} is received!')

    lod = self.lod.cpu().tolist()
    for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
      block_idx = self.final_res_log2 - res_log2
      if block_idx < lod:
        image = self.downsample(image)
        continue
      if block_idx == lod:
        x = self.__getattr__(f'input{block_idx}')(image)
      x = self.__getattr__(f'layer{2 * block_idx}')(x)
      x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
    x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)

    if self.label_size:
      x = torch.sum(x * label, dim=1, keepdim=True)
    return x


class MiniBatchSTDLayer(nn.Module):
  """Implements the minibatch standard deviation layer."""

  def __init__(self, group_size=16, num_channels=1, epsilon=1e-8):
    super().__init__()
    self.group_size = group_size
    self.num_channels = num_channels
    self.epsilon = epsilon

  def forward(self, x):
    if self.group_size <= 1:
      return x
    ng = min(self.group_size, x.shape[0])  # [NCHW]
    nc = self.num_channels
    y = x.view(ng, -1, nc, x.shape[1] // nc, x.shape[2], x.shape[3])  # [GMncHW]
    y = y - torch.mean(y, dim=0, keepdim=True)  # [GMncHW]
    y = torch.mean(y ** 2, dim=0)  # [MncHW]
    y = torch.sqrt(y + self.epsilon)  # [MncHW]
    y = torch.mean(y, dim=[2, 3, 4], keepdim=True)  # [Mn111]
    y = torch.mean(y, dim=2)  # [Mn11]
    y = y.repeat(ng, 1, x.shape[2], x.shape[3])  # [NnHW]
    return torch.cat([x, y], dim=1)


class AveragePoolingLayer(nn.Module):
  """Implements the average pooling layer.

  Basically, this layer can be used to downsample feature maps from spatial
  domain.
  """

  def __init__(self, scale_factor=2):
    super().__init__()
    self.scale_factor = scale_factor

  def forward(self, x):
    ksize = [self.scale_factor, self.scale_factor]
    strides = [self.scale_factor, self.scale_factor]
    return F.avg_pool2d(x, kernel_size=ksize, stride=strides, padding=0)


class BlurLayer(nn.Module):
  """Implements the blur layer."""

  def __init__(self,
               channels,
               kernel=(1, 2, 1),
               normalize=True,
               flip=False):
    super().__init__()
    kernel = np.array(kernel, dtype=np.float32).reshape(1, -1)
    kernel = kernel.T.dot(kernel)
    if normalize:
      kernel /= np.sum(kernel)
    if flip:
      kernel = kernel[::-1, ::-1]
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    kernel = np.tile(kernel, [1, 1, channels, 1])
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    self.register_buffer('kernel', torch.from_numpy(kernel))
    self.channels = channels

  def forward(self, x):
    return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.channels)


class WScaleLayer(nn.Module):
  """Implements the layer to scale weight variable and add bias.

  NOTE: The weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
  layer), and only scaled with a constant number, which is not trainable in
  this layer. However, the bias variable is trainable in this layer.
  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               gain=np.sqrt(2.0),
               lr_multiplier=1.0):
    super().__init__()
    fan_in = in_channels * kernel_size * kernel_size
    self.scale = gain / np.sqrt(fan_in) * lr_multiplier
    self.bias = nn.Parameter(torch.zeros(out_channels))
    self.lr_multiplier = lr_multiplier

  def forward(self, x):
    if x.ndim == 4:
      return x * self.scale + self.bias.view(1, -1, 1, 1) * self.lr_multiplier
    if x.ndim == 2:
      return x * self.scale + self.bias.view(1, -1) * self.lr_multiplier
    raise ValueError(f'The input tensor should be with shape [batch_size, '
                     f'channel, height, width], or [batch_size, channel]!\n'
                     f'But {x.shape} is received!')


class ConvBlock(nn.Module):
  """Implements the convolutional block.

  Basically, this block executes minibatch standard deviation layer (if needed),
  convolutional layer,  weight-scale layer, activation layer, and downsampling
  layer (if needed) in sequence.
  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               downsample=False,
               fused_scale=False,
               wscale_gain=np.sqrt(2.0),
               activation_type='lrelu',
               minibatch_std=False,
               minibatch_std_group_size=16,
               minibatch_std_num_channels=1):
    """Initializes the class with block settings.

    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels (kernels) of the output tensor.
      kernel_size: Size of the convolutional kernel.
      stride: Stride parameter for convolution operation.
      padding: Padding parameter for convolution operation.
      dilation: Dilation rate for convolution operation.
      add_bias: Whether to add bias onto the convolutional result.
      downsample: Whether to downsample the input tensor after convolution.
      fused_scale: Whether to fused `conv2d` and `downsample` together,
        resulting in `conv2d` with strides.
      wscale_gain: The gain factor for `wscale` layer.
      activation_type: Type of activation. Support `linear` and `lrelu`.
      minibatch_std: Whether to perform minibatch standard deviation.
      minibatch_std_group_size: Group size for the minibatch standard deviation
        layer.
      minibatch_std_num_channels: Number of new channels after the minibatch
        standard deviation layer.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()

    if minibatch_std:
      in_channels = in_channels + minibatch_std_num_channels
      self.preact = MiniBatchSTDLayer(group_size=minibatch_std_group_size,
                                      num_channels=minibatch_std_num_channels)
    elif downsample:
      self.preact = BlurLayer(channels=in_channels)
    else:
      self.preact = nn.Identity()

    if downsample and not fused_scale:
      self.downsample = AveragePoolingLayer()
    else:
      self.downsample = nn.Identity()

    if downsample and fused_scale:
      self.use_stride = True
      self.weight = nn.Parameter(
          torch.randn(kernel_size, kernel_size, in_channels, out_channels))
    else:
      self.use_stride = False
      self.conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            bias=add_bias)

    fan_in = in_channels * kernel_size * kernel_size
    self.scale = wscale_gain / np.sqrt(fan_in)
    self.wscale = WScaleLayer(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              gain=wscale_gain)

    if activation_type == 'linear':
      self.activate = nn.Identity()
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    x = self.preact(x)
    if self.use_stride:
      kernel = self.weight * self.scale
      kernel = F.pad(kernel, (0, 0, 0, 0, 1, 1, 1, 1), 'constant', 0.0)
      kernel = (kernel[1:, 1:] + kernel[:-1, 1:] +
                kernel[1:, :-1] + kernel[:-1, :-1]) * 0.25
      kernel = kernel.permute(3, 2, 0, 1)
      x = F.conv2d(x, kernel, stride=2, padding=1)
    else:
      x = self.conv(x) * self.scale
      x = self.downsample(x)
    x = x / self.scale
    x = self.wscale(x)
    x = self.activate(x)
    return x


class DenseBlock(nn.Module):
  """Implements the dense block.

  Basically, this block executes fully-connected layer, weight-scale layer,
  and activation layer in sequence.
  """

  def __init__(self,
               in_channels,
               out_channels,
               add_bias=False,
               wscale_gain=np.sqrt(2.0),
               activation_type='lrelu'):
    """Initializes the class with block settings.

    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels of the output tensor.
      add_bias: Whether to add bias onto the fully-connected result.
      wscale_gain: The gain factor for `wscale` layer.
      activation_type: Type of activation. Support `linear` and `lrelu`.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()
    self.fc = nn.Linear(in_features=in_channels,
                        out_features=out_channels,
                        bias=add_bias)
    self.wscale = WScaleLayer(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              gain=wscale_gain)
    if activation_type == 'linear':
      self.activate = nn.Identity()
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    if x.ndim != 2:
      x = x.view(x.shape[0], -1)
    x = self.fc(x)
    x = self.wscale(x)
    x = self.activate(x)
    return x

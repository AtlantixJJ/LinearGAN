# python 3.7
"""Contains the implementation of face attribute classifier.

This classifier is provied by StyleGAN2, and actually borrows from the network
structure of StyleGAN discriminator. The differences are as follows:
(1) No progressive growing.
(2) No minibatch standard deviation layer.
(3) Using fixed resolution, 256.

For more details, please check the original StyleGAN paper:
https://arxiv.org/pdf/1812.04948.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FaceAttributeNet']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Fused-scale options allowed.
_FUSED_SCALE_OPTIONS_ALLOWED = [True, False, 'auto']

# Minimal resolution for `auto` fused-scale strategy.
_AUTO_FUSED_SCALE_MIN_RES = 128


class FaceAttributeNet(nn.Module):
  """Defines the face attribute classifier.

  NOTE: The classifier takes images with `RGB` color channels and range [-1, 1]
  as inputs.
  """

  def __init__(self,
               resolution=256,
               image_channels=3,
               fused_scale='auto',
               fmaps_base=16 << 10,
               fmaps_max=512):
    """Initializes the classifier with basic settings.

    Args:
      resolution: The resolution of the input image. (default: 256)
      image_channels: Number of channels of the input image. (default: 3)
      fused_scale: Whether to fused `conv2d` and `downsample` together,
        resulting in `conv2d` with strides. (default: False)
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
    self.fused_scale = fused_scale
    self.fmaps_base = fmaps_base
    self.fmaps_max = fmaps_max

    # Input convolution layer.
    self.add_module(
        f'input',
        ConvBlock(in_channels=self.image_channels,
                  out_channels=self.get_nf(self.resolution),
                  kernel_size=1,
                  padding=0))

    for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
      res = 2 ** res_log2
      block_idx = self.final_res_log2 - res_log2

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

      # Convolution block for last resolution.
      else:
        self.add_module(
            f'layer{2 * block_idx}',
            ConvBlock(
                in_channels=self.get_nf(res),
                out_channels=self.get_nf(res)))
        self.add_module(
            f'layer{2 * block_idx + 1}',
            DenseBlock(in_channels=self.get_nf(res) * res * res,
                       out_channels=self.get_nf(res // 2)))
        self.add_module(
            f'layer{2 * block_idx + 2}',
            DenseBlock(in_channels=self.get_nf(res // 2),
                       out_channels=1,
                       wscale_gain=1.0,
                       activation_type='linear'))

  def get_nf(self, res):
    """Gets number of feature maps according to current resolution."""
    return min(self.fmaps_base // res, self.fmaps_max)

  def forward(self, image):
    if image.ndim != 4 or image.shape[1:] != (
        self.image_channels, self.resolution, self.resolution):
      raise ValueError(f'The input image should be with shape [batch_size, '
                       f'channel, height, width], where '
                       f'`channel` equals to {self.image_channels}, '
                       f'`height` and `width` equal to {self.resolution}!\n'
                       f'But {image.shape} is received!')

    x = self.input(image)
    for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
      block_idx = self.final_res_log2 - res_log2
      x = self.__getattr__(f'layer{2 * block_idx}')(x)
      x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
    x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)
    return x


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

  Basically, this block executes convolutional layer, weight-scale layer,
  activation layer, and downsampling layer (if needed) in sequence.
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
               activation_type='lrelu'):
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
      activation_type: Type of activation function. Support `linear`, `lrelu`
        and `tanh`.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()

    if downsample:
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
    elif activation_type == 'tanh':
      self.activate = nn.Hardtanh()
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
      activation_type: Type of activation function. Support `linear` and
        `lrelu`.

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

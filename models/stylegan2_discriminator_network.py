# python 3.7
"""Contains the implementation of discriminator described in StyleGAN2.

Different from the official tensorflow version in folder `stylegan2_official`,
this is a simple pytorch version which only contains the discriminator part.
This class is specially used for inference.

NOTE: Compared to that of StyleGAN, the discriminator in StyleGAN2 mainly adds
skip connections and disables progressive growth. This script ONLY supports
config F in the original paper.

For more details, please check the original paper:
https://arxiv.org/pdf/1912.04958.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['StyleGAN2DiscriminatorNet', 'DownsamplingLayer']

# Resolutions allowed for discriminator in StyleGAN2.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Architectures allowed.
_ARCHITECTURES_ALLOWED = ['resnet', 'skip', 'origin']


class StyleGAN2DiscriminatorNet(nn.Module):
  """Defines the discriminator network in StyleGAN2.

  NOTE: The discriminator takes images with `RGB` color channels and range
  [-1, 1] as inputs.
  """

  def __init__(self,
               resolution,
               image_channels=3,
               label_size=0,
               architecture_type='resnet',
               minibatch_std_group_size=4,
               minibatch_std_num_channels=1,
               fmaps_base=32 << 10,
               fmaps_max=512):
    """Initializes the discriminator with basic settings.

    Args:
      resolution: The resolution of the input image.
      image_channels: Number of channels of the input image. (default: 3)
      label_size: Size of additional labels. (default: 0)
      architecture_type: Defines the architecture type. (default: `resnet`)
      minibatch_std_group_size: Group size for the minibatch standard deviation
        layer. 0 means disable. (default: 4)
      minibatch_std_num_channels: Number of new channels after the minibatch
        standard deviation layer. (default: 1)
      fmaps_base: Base factor to compute number of feature maps for each layer.
        This field doubles that of StyleGAN. (default: 32 << 10)
      fmaps_max: Maximum number of feature maps in each layer. (default: 512)

    Raises:
      ValueError: If the input `resolution` is not supported, or
        `architecture_type` is not supported.
    """
    super().__init__()

    if resolution not in _RESOLUTIONS_ALLOWED:
      raise ValueError(f'Invalid resolution: {resolution}!\n'
                       f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
    if architecture_type not in _ARCHITECTURES_ALLOWED:
      raise ValueError(f'Invalid fused-scale option: {architecture_type}!\n'
                       f'Architectures allowed: {_ARCHITECTURES_ALLOWED}.')

    self.init_res = _INIT_RES
    self.init_res_log2 = int(np.log2(self.init_res))
    self.resolution = resolution
    self.final_res_log2 = int(np.log2(self.resolution))
    self.image_channels = image_channels
    self.label_size = label_size
    self.architecture_type = architecture_type
    self.minibatch_std_group_size = minibatch_std_group_size
    self.minibatch_std_num_channels = minibatch_std_num_channels
    self.fmaps_base = fmaps_base
    self.fmaps_max = fmaps_max

    self.pth_to_tf_var_mapping = {}
    for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
      res = 2 ** res_log2
      block_idx = self.final_res_log2 - res_log2

      # Input convolution layer for each resolution (if needed).
      if res_log2 == self.final_res_log2 or self.architecture_type == 'skip':
        self.add_module(
            f'input{block_idx}',
            ConvBlock(in_channels=self.image_channels,
                      out_channels=self.get_nf(res),
                      kernel_size=1))
        self.pth_to_tf_var_mapping[f'input{block_idx}.conv.weight'] = (
            f'{res}x{res}/FromRGB/weight')
        self.pth_to_tf_var_mapping[f'input{block_idx}.bias'] = (
            f'{res}x{res}/FromRGB/bias')

      # Convolution block for each resolution (except the last one).
      if res != self.init_res:
        self.add_module(
            f'layer{2 * block_idx}',
            ConvBlock(in_channels=self.get_nf(res),
                      out_channels=self.get_nf(res)))
        self.add_module(
            f'layer{2 * block_idx + 1}',
            ConvBlock(in_channels=self.get_nf(res),
                      out_channels=self.get_nf(res // 2),
                      scale_factor=2))
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.conv.weight'] = (
            f'{res}x{res}/Conv0/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = (
            f'{res}x{res}/Conv0/bias')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.conv.weight'] = (
            f'{res}x{res}/Conv1_down/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = (
            f'{res}x{res}/Conv1_down/bias')

        if self.architecture_type == 'resnet':
          self.add_module(
              f'skip_layer{block_idx}',
              ConvBlock(in_channels=self.get_nf(res),
                        out_channels=self.get_nf(res // 2),
                        kernel_size=1,
                        scale_factor=2,
                        add_bias=False,
                        activation_type='linear'))
          self.pth_to_tf_var_mapping[f'skip_layer{block_idx}.conv.weight'] = (
              f'{res}x{res}/Skip/weight')

      # Convolution block for last resolution.
      else:
        extra_channels = self.minibatch_std_num_channels
        self.add_module(
            f'layer{2 * block_idx}',
            ConvBlock(in_channels=self.get_nf(res) + extra_channels,
                      out_channels=self.get_nf(res)))
        self.add_module(
            f'layer{2 * block_idx + 1}',
            DenseBlock(in_channels=self.get_nf(res) * res * res,
                       out_channels=self.get_nf(res // 2)))
        self.add_module(
            f'layer{2 * block_idx + 2}',
            DenseBlock(in_channels=self.get_nf(res // 2),
                       out_channels=max(self.label_size, 1),
                       activation_type='linear'))
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.conv.weight'] = (
            f'{res}x{res}/Conv/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = (
            f'{res}x{res}/Conv/bias')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.fc.weight'] = (
            f'{res}x{res}/Dense0/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = (
            f'{res}x{res}/Dense0/bias')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.fc.weight'] = (
            f'Output/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.bias'] = (
            f'Output/bias')

    if self.architecture_type == 'skip':
      self.downsample = DownsamplingLayer()
    self.minibatch_std_layer = MiniBatchSTDLayer(
        group_size=self.minibatch_std_group_size,
        num_channels=self.minibatch_std_num_channels)

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

    if self.architecture_type == 'origin':
      x = self.input0(image)
      for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
        block_idx = self.final_res_log2 - res_log2
        if res_log2 == self.init_res_log2:
          x = self.minibatch_std_layer(x)
        x = self.__getattr__(f'layer{2 * block_idx}')(x)
        x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
      x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)

    elif self.architecture_type == 'skip':
      for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
        block_idx = self.final_res_log2 - res_log2
        if block_idx == 0:
          x = self.__getattr__(f'input{block_idx}')(image)
        else:
          image = self.downsample(image)
          x = x + self.__getattr__(f'input{block_idx}')(image)
        if res_log2 == self.init_res_log2:
          x = self.minibatch_std_layer(x)
        x = self.__getattr__(f'layer{2 * block_idx}')(x)
        x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
      x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)

    elif self.architecture_type == 'resnet':
      x = self.input0(image)
      for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
        block_idx = self.final_res_log2 - res_log2
        if res_log2 != self.init_res_log2:
          residual = self.__getattr__(f'skip_layer{block_idx}')(x)
          x = self.__getattr__(f'layer{2 * block_idx}')(x)
          x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
          x = (x + residual) / np.sqrt(2.0)
        else:
          x = self.minibatch_std_layer(x)
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


class DownsamplingLayer(nn.Module):
  """Implements the downsampling layer.

  This layer can also be used as filtering layer by setting `scale_factor` as 1.
  """

  def __init__(self, scale_factor=2, kernel=(1, 3, 3, 1), extra_padding=0):
    super().__init__()
    assert scale_factor >= 1
    self.scale_factor = scale_factor

    if extra_padding != 0:
      assert scale_factor == 1

    if kernel is None:
      kernel = np.ones((scale_factor), dtype=np.float32)
    else:
      kernel = np.array(kernel, dtype=np.float32)
    assert kernel.ndim == 1
    kernel = np.outer(kernel, kernel)
    kernel = kernel / np.sum(kernel)
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    self.register_buffer('kernel', torch.from_numpy(kernel))
    self.kernel = self.kernel.flip(0, 1)
    padding = kernel.shape[2] - scale_factor + extra_padding
    self.padding = ((padding + 1) // 2, padding // 2,
                    (padding + 1) // 2, padding // 2)

  def forward(self, x):
    assert x.ndim == 4
    channels = x.shape[1]
    x = x.view(-1, 1, x.shape[2], x.shape[3])
    x = F.pad(x, self.padding, mode='constant', value=0)
    x = F.conv2d(x, self.kernel, stride=self.scale_factor)
    x = x.view(-1, channels, x.shape[2], x.shape[3])
    return x


class ConvBlock(nn.Module):
  """Implements the convolutional block."""

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               scale_factor=1,
               filtering_kernel=(1, 3, 3, 1),
               weight_gain=1.0,
               lr_multiplier=1.0,
               add_bias=True,
               activation_type='lrelu'):
    """Initializes the class with block settings.

    NOTE: Wscale is used as default.

    Args:
      in_channels: Number of channels of the input tensor.
      out_channels: Number of channels (kernels) of the output tensor.
      kernel_size: Size of the convolutional kernel.
      scale_factor: Scale factor for downsampling. `1` means skip downsampling.
      filtering_kernel: Kernel used for filtering before downsampling.
      weight_gain: Gain factor for weight parameter in convolutional layer.
      lr_multiplier: Learning rate multiplier.
      add_bias: Whether to add bias after convolution.
      activation_type: Type of activation. Support `linear`, `relu`, `lrelu`.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()

    if scale_factor > 1:
      self.filter = DownsamplingLayer(scale_factor=1,
                                      kernel=filtering_kernel,
                                      extra_padding=kernel_size - scale_factor)
      padding = 0  # Padding is done in `DownsamplingLayer`.
    else:
      self.filter = nn.Identity()
      assert kernel_size % 2 == 1
      padding = kernel_size // 2

    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=scale_factor,
                          padding=padding,
                          dilation=1,
                          groups=1,
                          bias=False)

    self.add_bias = add_bias
    if add_bias:
      self.bias = nn.Parameter(torch.zeros(out_channels))

    fan_in = in_channels * kernel_size * kernel_size
    self.weight_scale = weight_gain / np.sqrt(fan_in)

    if activation_type == 'linear':
      self.activate = nn.Identity()
      self.activate_scale = 1.0 * lr_multiplier
    elif activation_type == 'relu':
      self.activate = nn.ReLU(inplace=True)
      self.activate_scale = np.sqrt(2.0) * lr_multiplier
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
      self.activate_scale = np.sqrt(2.0) * lr_multiplier
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    x = self.filter(x)
    x = self.conv(x) * self.weight_scale
    if self.add_bias:
      x = x + self.bias.view(1, -1, 1, 1)
    x = self.activate(x) * self.activate_scale
    return x


class DenseBlock(nn.Module):
  """Implements the dense block."""

  def __init__(self,
               in_channels,
               out_channels,
               weight_gain=1.0,
               lr_multiplier=1.0,
               add_bias=True,
               activation_type='lrelu'):
    """Initializes the class with block settings.

    NOTE: Wscale is used as default.

    Args:
      in_channels: Number of channels of the input tensor.
      out_channels: Number of channels of the output tensor.
      weight_gain: Gain factor for weight parameter in dense layer.
      lr_multiplier: Learning rate multiplier.
      add_bias: Whether to add bias after fully-connected operation.
      activation_type: Type of activation. Support `linear`, `relu`, `lrelu`.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()

    self.fc = nn.Linear(in_features=in_channels,
                        out_features=out_channels,
                        bias=False)

    self.add_bias = add_bias
    if add_bias:
      self.bias = nn.Parameter(torch.zeros(out_channels))

    self.weight_scale = weight_gain / np.sqrt(in_channels)

    if activation_type == 'linear':
      self.activate = nn.Identity()
      self.activate_scale = 1.0 * lr_multiplier
    elif activation_type == 'relu':
      self.activate = nn.ReLU(inplace=True)
      self.activate_scale = np.sqrt(2.0) * lr_multiplier
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
      self.activate_scale = np.sqrt(2.0) * lr_multiplier
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    if x.ndim != 2:
      x = x.view(x.shape[0], -1)
    x = self.fc(x) * self.weight_scale
    if self.add_bias:
      x = x + self.bias.view(1, -1)
    x = self.activate(x) * self.activate_scale
    return x

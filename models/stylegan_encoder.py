# python 3.7
"""Contains the encoder class of StyleGAN inversion.

This class is derived from the `BaseEncoder` class defined in `base_encoder.py`.
"""

import numpy as np

import torch

from .base_encoder import BaseEncoder
from .stylegan_encoder_network import StyleGANEncoderNet

__all__ = ['StyleGANEncoder']


class StyleGANEncoder(BaseEncoder):
  """Defines the encoder class of StyleGAN inversion."""

  def __init__(self, model_name, logger=None):
    self.gan_type = 'stylegan'
    super().__init__(model_name, logger)

  def build(self):
    self.w_space_dim = getattr(self, 'w_space_dim', 512)
    self.encoder_channels_base = getattr(self, 'encoder_channels_base', 64)
    self.encoder_channels_max = getattr(self, 'encoder_channels_max', 1024)
    self.use_wscale = getattr(self, 'use_wscale', False)
    self.use_bn = getattr(self, 'use_bn', False)
    self.net = StyleGANEncoderNet(
        resolution=self.resolution,
        w_space_dim=self.w_space_dim,
        image_channels=self.image_channels,
        encoder_channels_base=self.encoder_channels_base,
        encoder_channels_max=self.encoder_channels_max,
        use_wscale=self.use_wscale,
        use_bn=self.use_bn)
    self.num_layers = self.net.num_layers
    self.encode_dim = [self.num_layers, self.w_space_dim]

  def convert_tf_weights(self, test_num=10):
    # pylint: disable=import-outside-toplevel
    import sys
    import pickle
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # pylint: enable=import-outside-toplevel

    sess = tf.compat.v1.InteractiveSession()

    self.logger.info(f'Loading tf weights from `{self.tf_weight_path}`.')
    sys.path.insert(0, self.official_code_path)
    with open(self.tf_weight_path, 'rb') as f:
      tf_net, _, _, _ = pickle.load(f)  # E, G, D, Gs
    sys.path.pop(0)
    self.logger.info(f'Successfully loaded!')

    self.logger.info(f'Converting tf weights to pytorch version.')
    tf_vars = dict(tf_net.__getstate__()['variables'])
    state_dict = self.net.state_dict()
    for pth_var_name, tf_var_name in self.net.pth_to_tf_var_mapping.items():
      assert tf_var_name in tf_vars
      assert pth_var_name in state_dict
      self.logger.debug(f'  Converting `{tf_var_name}` to `{pth_var_name}`.')
      var = torch.from_numpy(np.array(tf_vars[tf_var_name]))
      if 'weight' in pth_var_name:
        if 'fc' in pth_var_name:
          var = var.permute(1, 0)
        elif 'conv' in pth_var_name:
          var = var.permute(3, 2, 0, 1)
      state_dict[pth_var_name] = var
    self.logger.info(f'Successfully converted!')

    self.logger.info(f'Saving pytorch weights to `{self.weight_path}`.')
    for var_name in self.model_specific_vars:
      del state_dict[var_name]
    torch.save(state_dict, self.weight_path)
    self.logger.info(f'Successfully saved!')

    self.load()

    # Start testing if needed.
    if test_num <= 0 or not tf.test.is_built_with_cuda():
      self.logger.warning(f'Skip testing the weights converted from tf model!')
      sess.close()
      return
    self.logger.info(f'Testing conversion results.')
    self.net.eval().to(self.run_device)
    total_distance = 0.0
    for i in range(test_num):
      input_shape = [1, self.image_channels, self.resolution, self.resolution]
      image = np.random.rand(*input_shape) * 2 - 1
      tf_output = tf_net.run(image, phase=False)
      pth_output = self.encode(image)['code'].reshape(1, -1)
      distance = np.average(np.abs(tf_output - pth_output))
      self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
      total_distance += distance
    self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

    sess.close()

  def _encode(self, images):
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
    if (images.ndim != 4 or images.shape[0] <= 0 or
        images.shape[0] > self.batch_size or images.shape[1:] != (
            self.image_channels, self.resolution, self.resolution)):
      raise ValueError(f'Input images should be with shape [batch_size, '
                       f'channel, height, width], where '
                       f'`batch_size` no larger than {self.batch_size}, '
                       f'`channel` equals to {self.image_channels}, '
                       f'`height` and `width` equal to {self.resolution}!\n'
                       f'But {images.shape} is received!')

    xs = self.to_tensor(images.astype(np.float32))
    codes = self.net(xs)
    assert codes.shape == (images.shape[0], np.prod(self.encode_dim))
    codes = codes.view(codes.shape[0], *self.encode_dim)
    results = {
        'image': images,
        'code': self.get_value(codes),
    }

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def encode(self, images, **kwargs):
    return self.batch_run(images, self._encode)

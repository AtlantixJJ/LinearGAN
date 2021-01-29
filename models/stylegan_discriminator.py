# python 3.7
"""Contains the discriminator class of StyleGAN.

This class is derived from the `BaseDiscriminator` class defined in
`base_discriminator.py`.
"""

import numpy as np

import torch

from .base_discriminator import BaseDiscriminator
from .stylegan_discriminator_network import StyleGANDiscriminatorNet

__all__ = ['StyleGANDiscriminator']


class StyleGANDiscriminator(BaseDiscriminator):
  """Defines the discriminator class of StyleGAN."""

  def __init__(self, model_name, logger=None):
    self.gan_type = 'stylegan'
    super().__init__(model_name, logger)
    self.lod = self.net.lod.to(self.cpu_device).tolist()
    self.logger.info(f'Current `lod` is {self.lod}.')

  def build(self):
    self.label_size = getattr(self, 'label_size', 0)
    self.fused_scale = getattr(self, 'fused_scale', 'auto')
    self.minibatch_std_group_size = getattr(self, 'minibatch_std_group_size', 4)
    self.minibatch_std_num_channels = getattr(
        self, 'minibatch_std_num_channels', 1)
    self.fmaps_base = getattr(self, 'fmaps_base', 16 << 10)
    self.fmaps_max = getattr(self, 'fmaps_max', 512)
    self.net = StyleGANDiscriminatorNet(
        resolution=self.resolution,
        image_channels=self.image_channels,
        label_size=self.label_size,
        fused_scale=self.fused_scale,
        minibatch_std_group_size=self.minibatch_std_group_size,
        minibatch_std_num_channels=self.minibatch_std_num_channels,
        fmaps_base=self.fmaps_base,
        fmaps_max=self.fmaps_max)

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
      if self.has_encoder:
        _, _, tf_net, _ = pickle.load(f)  # E, G, D, Gs
      else:
        _, tf_net, _ = pickle.load(f)  # G, D, Gs
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
      if self.label_size:
        label_id = np.random.randint(self.label_size)
        tf_label = np.zeros((1, self.label_size), np.float32)
        tf_label[0, label_id] = 1.0
      else:
        label_id = None
        tf_label = None
      tf_output = tf_net.run(image, tf_label)
      pth_output = self.rate(image, labels=label_id)['score']
      distance = np.average(np.abs(tf_output - pth_output))
      self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
      total_distance += distance
    self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

    sess.close()

  def _rate(self, images, labels=None):
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
    labels = self.get_ont_hot_labels(images.shape[0], labels)
    ls = None if labels is None else self.to_tensor(labels.astype(np.float32))
    scores = self.net(xs, ls)
    assert scores.shape == (images.shape[0], 1)
    results = {
        'image': images,
        'score': self.get_value(scores),
    }
    if self.label_size:
      results['label'] = labels

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def rate(self, images, labels=None, **kwargs):
    return self.batch_run(images, lambda x: self._rate(x, labels=labels))

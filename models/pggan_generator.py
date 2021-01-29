# python 3.7
"""Contains the generator class of PGGAN.

This class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import numpy as np

import torch

from .base_generator import BaseGenerator
from .pggan_generator_network import PGGANGeneratorNet

__all__ = ['PGGANGenerator']


class PGGANGenerator(BaseGenerator):
  """Defines the generator class of PGGAN."""

  def __init__(self, model_name, logger=None):
    self.gan_type = 'pggan'
    super().__init__(model_name, logger)
    self.lod = self.net.lod.to(self.cpu_device).tolist()
    self.logger.info(f'Current `lod` is {self.lod}.')

  def build(self):
    self.z_space_dim = getattr(self, 'z_space_dim', 512)
    self.final_tanh = getattr(self, 'final_tanh', False)
    self.label_size = getattr(self, 'label_size', 0)
    self.fused_scale = getattr(self, 'fused_scale', False)
    self.fmaps_base = getattr(self, 'fmaps_base', 16 << 10)
    self.fmaps_max = getattr(self, 'fmaps_max', 512)
    self.net = PGGANGeneratorNet(resolution=self.resolution,
                                 z_space_dim=self.z_space_dim,
                                 image_channels=self.image_channels,
                                 final_tanh=self.final_tanh,
                                 label_size=self.label_size,
                                 fused_scale=self.fused_scale,
                                 fmaps_base=self.fmaps_base,
                                 fmaps_max=self.fmaps_max)
    self.num_layers = self.net.num_layers

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
        _, _, _, tf_net = pickle.load(f)  # E, G, D, Gs
      else:
        _, _, tf_net = pickle.load(f)  # G, D, Gs
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
        if 'layer0.conv' in pth_var_name:
          var = var.view(var.shape[0], -1, self.net.init_res, self.net.init_res)
          var = var.permute(1, 0, 2, 3).flip(2, 3)
        elif 'conv' in pth_var_name:
          var = var.permute(3, 2, 0, 1)
        elif 'conv' not in pth_var_name:
          var = var.permute(0, 1, 3, 2)
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
      latent_code = self.easy_sample(1)
      tf_label = np.zeros((1, self.label_size), np.float32)
      if self.label_size:
        label_id = np.random.randint(self.label_size)
        tf_label[0, label_id] = 1.0
      else:
        label_id = None
      tf_output = tf_net.run(latent_code, tf_label)
      pth_output = self.synthesize(latent_code, labels=label_id)['image']
      distance = np.average(np.abs(tf_output - pth_output))
      self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
      total_distance += distance
    self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

    sess.close()

  def sample(self, num, **kwargs):
    assert num > 0
    return np.random.randn(num, self.z_space_dim).astype(np.float32)

  def preprocess(self, latent_codes, **kwargs):
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    latent_codes = latent_codes.reshape(-1, self.z_space_dim)
    norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
    latent_codes = latent_codes / norm * np.sqrt(self.z_space_dim)
    return latent_codes.astype(np.float32)

  def _synthesize(self, latent_codes, labels=None):
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
    if (latent_codes.ndim != 2 or latent_codes.shape[0] <= 0 or
        latent_codes.shape[0] > self.batch_size or
        latent_codes.shape[1] != self.z_space_dim):
      raise ValueError(f'Latent codes should be with shape [batch_size, '
                       f'latent_space_dim], where `batch_size` no larger than '
                       f'{self.batch_size}, and `latent_space_dim` equals to '
                       f'{self.z_space_dim}!\n'
                       f'But {latent_codes.shape} is received!')

    zs = self.to_tensor(latent_codes.astype(np.float32))
    labels = self.get_ont_hot_labels(latent_codes.shape[0], labels)
    ls = None if labels is None else self.to_tensor(labels.astype(np.float32))
    images = self.net(zs, ls)
    results = {
        'z': latent_codes,
        'image': self.get_value(images),
    }
    if self.label_size:
      results['label'] = labels

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def synthesize(self, latent_codes, labels=None, **kwargs):
    return self.batch_run(latent_codes,
                          lambda x: self._synthesize(x, labels=labels))

  def __call__(self, latent_codes, labels=None, generate_feature=True):
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
    if (latent_codes.ndim != 2 or latent_codes.shape[0] <= 0 or
        latent_codes.shape[0] > self.batch_size or
        latent_codes.shape[1] != self.z_space_dim):
      raise ValueError(f'Latent codes should be with shape [batch_size, '
                       f'latent_space_dim], where `batch_size` no larger than '
                       f'{self.batch_size}, and `latent_space_dim` equals to '
                       f'{self.z_space_dim}!\n'
                       f'But {latent_codes.shape} is received!')

    zs = self.to_tensor(latent_codes.astype(np.float32))
    labels = self.get_ont_hot_labels(latent_codes.shape[0], labels)
    ls = None if labels is None else self.to_tensor(labels.astype(np.float32))
    images, feats = self.net(zs, ls, generate_feature=True)
    results = {
      'z': latent_codes,
      'image': images,
    }
    if generate_feature:
      results['feature'] = feats
    if self.label_size:
      results['label'] = labels

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results
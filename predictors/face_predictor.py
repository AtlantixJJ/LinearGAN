# python 3.7
"""Predictor for face attribute."""

import os
import numpy as np

import torch
import torch.nn.functional as F

from .base_predictor import BasePredictor
from .face_predictor_network import FaceAttributeNet

__all__ = ['MulticlassFacePredictor', 'FacePredictor', 'CELEBA_ATTRS']

CELEBA_ATTRS = 'male,smiling,eyeglasses,young,bald,rosy_cheeks,mustache,wavy_hair,blond_hair,heavy_makeup,attractive,bags_under_eyes,big_lips,blurry'.split(",")

class MulticlassFacePredictor(torch.nn.Module):
  def __init__(self, attr_names=CELEBA_ATTRS):
    super().__init__()
    self.Ps = torch.nn.ModuleList(
      [FacePredictor("celebahq_" + name) for name in attr_names])
  
  def forward(self, x, split=True):
    return torch.cat([P(x) if not split else \
        torch.cat([P(x[i:i+1].cuda()) for i in range(x.shape[0])])
      for P in self.Ps], 1)


class FacePredictor(BasePredictor):
  """Defines the predictor class for face analysis.

  NOTE: The output score by this predictor is in inverse proportion to the
  attribute score. Taking `male` as an example, -5.0 means male while 5.0 means
  female.
  """

  def __init__(self, predictor_name):
    self.attribute_name = predictor_name[len('celebahq_'):]
    self.input_size = 256
    super().__init__(predictor_name)

  def build(self):
    self.net = FaceAttributeNet(resolution=self.input_size)

  def load(self):
    # Load pre-trained weights.
    assert os.path.isfile(self.weight_path)
    self.net.load_state_dict(torch.load(self.weight_path))

  def _predict(self, images):
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')
    if images.dtype != np.uint8:
      raise ValueError(f'Images should be with dtype `numpy.uint8`!')
    if (images.ndim != 4 or images.shape[0] <= 0 or
        images.shape[0] > self.batch_size or
        images.shape[3] != self.image_channels):
      raise ValueError(f'Images should be with shape [batch_size, height '
                       f'width, channel], where `batch_size` no larger than '
                       f'{self.batch_size}, and `channel` equals to '
                       f'{self.image_channels}!\n'
                       f'But {images.shape} is received!')

    xs = self.to_tensor(images.astype(np.float32))
    xs = xs.permute(0, 3, 1, 2)
    xs = F.interpolate(xs,
                       size=(self.input_size, self.input_size),
                       mode='bilinear',
                       align_corners=False)
    xs = xs / 127.5 - 1.0

    attribute_scores = self.net(xs)
    assert attribute_scores.shape == (images.shape[0], 1)

    results = {
        'score': self.get_value(attribute_scores),
    }

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def __call__(self, x):
    """
    Expect input in (-1, 1)
    """
    if x.size(3) != self.input_size:
      x = F.interpolate(x,
                       size=(self.input_size, self.input_size),
                       mode='bilinear',
                       align_corners=False)
    return torch.cat([self.net(x[i:i+1]) for i in range(x.shape[0])])

  def predict(self, images, **kwargs):
    return self.batch_run(images, self._predict)

  def save(self, predictions, save_dir):
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)

    assert 'score' in predictions
    scores = np.concatenate(predictions['score'], axis=0)
    assert scores.ndim == 2 and scores.shape[1] == 1
    np.save(os.path.join(save_dir, f'{self.attribute_name}.npy'), scores)

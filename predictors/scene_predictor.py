# python 3.7
"""Predictor for scene category and attribute."""

import os
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .base_predictor import BasePredictor
from .scene_wideresnet import resnet18

__all__ = ['ScenePredictor', 'SubsetScenePredictor', 'MAN_ATTRS']

NUM_CATEGORIES = 365
NUM_ATTRIBUTES = 102
FEATURE_DIM = 512

MAN_ATTRS = "carpet wood glossy dirt scary indoor_lighting cluttered_space glass natural_light".split(" ")


class SubsetScenePredictor(torch.nn.Module):
  def __init__(self, attr_names=MAN_ATTRS, label_list="data/scene_predictor_labels.list"):
    super().__init__()
    label_list = [l.strip() for l in open(label_list, "r").readlines()]
    indice = [label_list.index(n) for n in attr_names]
    self.attr_indice = indice
    self.P = ScenePredictor()
  
  def forward(self, x, split=True):
    if split:
      return torch.cat([self.P(x[i:i+1].cuda())[:, self.attr_indice]
        for i in range(x.shape[0])])
    return self.P(x)[:, self.attr_indice]


class ScenePredictor(BasePredictor):
  """Defines the predictor class for scene analysis."""

  def __init__(self, predictor_name='scene'):
    assert predictor_name == 'scene'
    self.num_categories = NUM_CATEGORIES
    self.num_attributes = NUM_ATTRIBUTES
    self.feature_dim = FEATURE_DIM

    super().__init__(predictor_name)

  def build(self):
    self.net = resnet18(num_classes=self.num_categories)

    # Transform for input images.
    self.transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    self.norm_transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

  def load(self):
    # Load category labels.
    self.check_attr('category_anno_path')
    self.category_name_to_idx = {}
    self.category_idx_to_name = {}
    with open(self.category_anno_path, 'r') as f:
      for line in f:
        name, idx = line.strip().split(' ')
        name = name[3:].replace('/', '__')
        idx = int(idx)
        self.category_name_to_idx[name] = idx
        self.category_idx_to_name[idx] = name
    assert len(self.category_name_to_idx) == self.num_categories
    assert len(self.category_idx_to_name) == self.num_categories

    # Load attribute labels.
    self.check_attr('attribute_anno_path')
    self.attribute_name_to_idx = {}
    self.attribute_idx_to_name = {}
    with open(self.attribute_anno_path, 'r') as f:
      for idx, line in enumerate(f):
        name = line.strip().replace(' ', '_')
        self.attribute_name_to_idx[name] = idx
        self.attribute_idx_to_name[idx] = name
    assert len(self.attribute_name_to_idx) == self.num_attributes
    assert len(self.attribute_idx_to_name) == self.num_attributes

    # Load pre-trained weights for category prediction.
    assert os.path.isfile(self.weight_path)
    checkpoint = torch.load(self.weight_path,
                            map_location=lambda storage, loc: storage)
    state_dict = {k.replace('module.', ''): v
                  for k, v in checkpoint['state_dict'].items()}
    self.net.load_state_dict(state_dict)
    fc_weight = list(self.net.parameters())[-2].data.numpy()
    fc_weight[fc_weight < 0] = 0

    # Load additional weights for attribute prediction.
    self.check_attr('attribute_additional_weight_path')
    assert os.path.isfile(self.attribute_additional_weight_path)
    self.attribute_weight = np.load(self.attribute_additional_weight_path)
    assert self.attribute_weight.shape == (
        self.num_attributes, self.feature_dim)
    self.aw = torch.from_numpy(self.attribute_weight).float()

  def __call__(self, images):
    """
    Expect (-1, 1) input
    """
    self.aw = self.aw.to(self.run_device)
    images = F.interpolate(images, size=(224, 224),
      mode="bilinear", align_corners=True)
    images = (images + 1) / 2
    x = torch.stack([self.norm_transform(x) for x in images])
    res = []
    for i in range(x.shape[0]):
      logits, features = self.net(x[i:i+1])
      category_scores = F.softmax(logits, dim=1)
      features = features.squeeze(2).squeeze(2)
      attribute_scores = torch.matmul(features, self.aw.permute(1, 0))
      res.append(attribute_scores)
    return torch.cat(res)

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

    xs = [self.transform(image).unsqueeze(0) for image in images]
    xs = torch.cat(xs, dim=0).to(self.run_device)

    logits, features = self.net(xs)
    category_scores = self.get_value(F.softmax(logits, dim=1))
    features = self.get_value(features).squeeze(axis=(2, 3))
    attribute_scores = features.dot(self.attribute_weight.T)

    assert category_scores.shape == (images.shape[0], self.num_categories)
    assert attribute_scores.shape == (images.shape[0], self.num_attributes)

    results = {
        'category': category_scores,
        'attribute': attribute_scores,
    }

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def predict(self, images, **kwargs):
    return self.batch_run(images, self._predict)

  def save(self, predictions, save_dir):
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)

    # Categories
    assert 'category' in predictions
    categories = np.concatenate(predictions['category'], axis=0)
    assert categories.ndim == 2 and categories.shape[1] == self.num_categories
    detailed_categories = {
        'score': categories,
        'name_to_idx': self.category_name_to_idx,
        'idx_to_name': self.category_idx_to_name,
    }
    np.save(os.path.join(save_dir, 'category.npy'), detailed_categories)
    # Attributes
    assert 'attribute' in predictions
    attributes = np.concatenate(predictions['attribute'], axis=0)
    assert attributes.shape == (categories.shape[0], self.num_attributes)
    detailed_attributes = {
        'score': attributes,
        'name_to_idx': self.attribute_name_to_idx,
        'idx_to_name': self.attribute_idx_to_name,
    }
    np.save(os.path.join(save_dir, 'attribute.npy'), detailed_attributes)

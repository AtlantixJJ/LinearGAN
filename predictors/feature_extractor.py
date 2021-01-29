# python 3.7
"""Extracts image features using VGG, ResNet, etc."""

import os.path
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T

from .base_predictor import BasePredictor

__all__ = ['FeatureExtractor']


class FeatureExtractor(BasePredictor):
  """Defines the image feature extractor."""

  def __init__(self,
               architecture,
               spatial_feature=False,
               imagenet_logits=False):
    """Initializes with basic settings.

    architecture: Name of the architecture, such as `resnet18`.
    spatial_feature: Whether to extract spatial feature (feature map). If set
                     as `False`, a feature vector will be returned. (default:
                     False)
    imagenet_logits: Whether to return the final 1000-class logits corresponding
                     to ImageNet. (default: False)
    """
    self.spatial_feature = spatial_feature
    self.imagenet_logits = imagenet_logits
    self.feature_dim = None
    super().__init__(architecture)

  def build(self):
    # Transform for input images.
    input_size = 299 if self.predictor_name == 'inception_v3' else 224
    self.transform = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if self.predictor_name == 'alexnet':
      model = M.alexnet(pretrained=True)
    elif self.predictor_name == 'vgg11':
      model = M.vgg11(pretrained=True)
    elif self.predictor_name == 'vgg13':
      model = M.vgg13(pretrained=True)
    elif self.predictor_name == 'vgg16':
      model = M.vgg16(pretrained=True)
    elif self.predictor_name == 'vgg19':
      model = M.vgg19(pretrained=True)
    elif self.predictor_name == 'vgg11_bn':
      model = M.vgg11_bn(pretrained=True)
    elif self.predictor_name == 'vgg13_bn':
      model = M.vgg13_bn(pretrained=True)
    elif self.predictor_name == 'vgg16_bn':
      model = M.vgg16_bn(pretrained=True)
    elif self.predictor_name == 'vgg19_bn':
      model = M.vgg19_bn(pretrained=True)
    elif self.predictor_name == 'googlenet':
      model = M.googlenet(pretrained=True, aux_logits=False)
    elif self.predictor_name == 'inception_v3':
      model = M.inception_v3(pretrained=True, aux_logits=False)
    elif self.predictor_name == 'resnet18':
      model = M.resnet18(pretrained=True)
    elif self.predictor_name == 'resnet34':
      model = M.resnet34(pretrained=True)
    elif self.predictor_name == 'resnet50':
      model = M.resnet50(pretrained=True)
    elif self.predictor_name == 'resnet101':
      model = M.resnet101(pretrained=True)
    elif self.predictor_name == 'resnet152':
      model = M.resnet152(pretrained=True)
    elif self.predictor_name == 'resnext50':
      model = M.resnext50_32x4d(pretrained=True)
    elif self.predictor_name == 'resnext101':
      model = M.resnext101_32x8d(pretrained=True)
    elif self.predictor_name == 'wideresnet50':
      model = M.wide_resnet50_2(pretrained=True)
    elif self.predictor_name == 'wideresnet101':
      model = M.wide_resnet101_2(pretrained=True)
    elif self.predictor_name == 'densenet121':
      model = M.densenet121(pretrained=True)
    elif self.predictor_name == 'densenet169':
      model = M.densenet169(pretrained=True)
    elif self.predictor_name == 'densenet201':
      model = M.densenet201(pretrained=True)
    elif self.predictor_name == 'densenet161':
      model = M.densenet161(pretrained=True)
    else:
      raise NotImplementedError(f'Unsupported architecture '
                                f'`{self.predictor_name}`!')

    model.eval()

    if self.imagenet_logits:
      self.net = model
      self.feature_dim = (1000,)
      return

    if self.architecture_type == 'AlexNet':
      layers = list(model.features.children())
      if not self.spatial_feature:
        layers.append(nn.Flatten())
        self.feature_dim = (256 * 6 * 6,)
      else:
        self.feature_dim = (256, 6, 6)
    elif self.architecture_type == 'VGG':
      layers = list(model.features.children())
      if not self.spatial_feature:
        layers.append(nn.Flatten())
        self.feature_dim = (512 * 7 * 7,)
      else:
        self.feature_dim = (512, 7, 7)
    elif self.architecture_type == 'Inception':
      if self.predictor_name == 'googlenet':
        final_res = 7
        num_channels = 1024
        layers = list(model.children())[:-3]
      elif self.predictor_name == 'inception_v3':
        final_res = 8
        num_channels = 2048
        layers = list(model.children())[:-1]
        layers.insert(3, nn.MaxPool2d(kernel_size=3, stride=2))
        layers.insert(6, nn.MaxPool2d(kernel_size=3, stride=2))
      else:
        raise NotImplementedError(f'Unsupported Inception architecture '
                                  f'`{self.predictor_name}`!')
      if not self.spatial_feature:
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        self.feature_dim = (num_channels,)
      else:
        self.feature_dim = (num_channels, final_res, final_res)
    elif self.architecture_type == 'ResNet':
      if self.predictor_name in ['resnet18', 'resnet34']:
        num_channels = 512
      elif self.predictor_name in ['resnet50', 'resnet101', 'resnet152',
                                   'resnext50', 'resnext101', 'wideresnet50',
                                   'wideresnet101']:
        num_channels = 2048
      else:
        raise NotImplementedError(f'Unsupported ResNet architecture '
                                  f'`{self.predictor_name}`!')
      if not self.spatial_feature:
        layers = list(model.children())[:-1]
        layers.append(nn.Flatten())
        self.feature_dim = (num_channels,)
      else:
        layers = list(model.children())[:-2]
        self.feature_dim = (num_channels, 7, 7)
    elif self.architecture_type == 'DenseNet':
      if self.predictor_name == 'densenet121':
        num_channels = 1024
      elif self.predictor_name == 'densenet169':
        num_channels = 1664
      elif self.predictor_name == 'densenet201':
        num_channels = 1920
      elif self.predictor_name == 'densenet161':
        num_channels = 2208
      else:
        raise NotImplementedError(f'Unsupported DenseNet architecture '
                                  f'`{self.predictor_name}`!')
      layers = list(model.features.children())
      if not self.spatial_feature:
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        self.feature_dim = (num_channels,)
      else:
        self.feature_dim = (num_channels, 7, 7)
    else:
      raise NotImplementedError(f'Unsupported architecture type '
                                f'`{self.architecture_type}`!')
    self.net = nn.Sequential(*layers)

  def load(self):
    assert isinstance(self.feature_dim, tuple) and np.prod(self.feature_dim) > 0
    if not self.spatial_feature:
      assert len(self.feature_dim) == 1
    else:
      assert len(self.feature_dim) == 3

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

    features = self.net(xs)
    assert features.shape[1:] == self.feature_dim
    results = {'feature': self.get_value(features)}

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def predict(self, images, **kwargs):
    return self.batch_run(images, self._predict)

  def save(self, predictions, save_dir):
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)

    assert 'feature' in predictions
    np.save(os.path.join(save_dir, f'{self.predictor_name}_feature.npy'),
            predictions['feature'])

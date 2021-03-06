# python 3.7
"""Segmenter for face."""

import os
import numpy as np

import torch
import torch.nn.functional as F

from .base_predictor import BasePredictor
from .face_segmenter_network import UNet

from lib import op

CELEBA_FULL_CATEGORY = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

CELEBAMASK_NUMCLASS = 15
CELEBA_CATEGORY = ['bg', 'skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck', 'cloth']
CELEBA_COLORS = [(0, 0, 0),(128, 0, 0),(0, 128, 0),(128, 128, 0),(0, 0, 128),(128, 0, 128),(0, 128, 128),(128, 128, 128),(64, 0, 0),(192, 0, 0),(64, 128, 0),(192, 128, 0),(64, 0, 128),(192, 0, 128),(64, 128, 128),(192, 128, 128)]

class FaceSegmenter(BasePredictor):
  def __init__(self, model_name='stylegan2_ffhq'):
    self.num_categories = CELEBAMASK_NUMCLASS
    self.labels = CELEBA_CATEGORY
    super().__init__('face_seg')

  def build(self):
    self.net = UNet(resolution=self.resolution)
  
  def load(self):
    # Load pre-trained weights.
    assert os.path.isfile(self.weight_path)
    self.net.load_state_dict(torch.load(self.weight_path))

  def raw_prediction(self, images, size=None):
    """
    Expecting torch.Tensor as input
    """
    images = op.bu(images, self.resolution)
    x = self.net(images.clamp(-1, 1)) # (N, M, H, W)
    if size:
      x = op.bu(x, size)
    return x

  def __call__(self, images, size=None):
    """
    Expecting torch.Tensor as input
    """
    images = op.bu(images, self.resolution)
    x = self.net(images.clamp(-1, 1)) # (N, M, H, W)
    if size:
      x = op.bu(x, size)
    return x.argmax(1) # (N, H, W)

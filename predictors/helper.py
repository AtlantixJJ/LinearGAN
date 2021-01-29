# python 3.7
"""Helper function to build predictor."""

from .predictor_settings import PREDICTOR_POOL
from .face_segmenter import FaceSegmenter
from .scene_segmenter import SceneSegmenter
from .scene_predictor import ScenePredictor
from .face_predictor import FacePredictor
from .feature_extractor import FeatureExtractor

__all__ = ['build_predictor', 'build_extractor']


def build_predictor(predictor_name):
  """Builds predictor by predictor name."""
  if predictor_name not in PREDICTOR_POOL:
    raise ValueError(f'Model `{predictor_name}` is not registered in '
                     f'`PREDICTOR_POOL` in `predictor_settings.py`!')

  if predictor_name == 'face_seg':
    return FaceSegmenter(predictor_name)  
  if predictor_name == 'scene_seg':
    return SceneSegmenter(predictor_name)
  if predictor_name == 'scene':
    return ScenePredictor(predictor_name)
  if predictor_name[:len('celebahq_')] == 'celebahq_':
    return FacePredictor(predictor_name)
  raise NotImplementedError(f'Unsupported predictor `{predictor_name}`!')


def build_extractor(architecture, spatial_feature=False, imagenet_logits=False):
  """Builds feature extractor by architecture name."""
  if architecture not in PREDICTOR_POOL:
    raise ValueError(f'Feature extractor with architecture `{architecture}` is '
                     f'not registered in `PREDICTOR_POOL` in '
                     f'`predictor_settings.py`!')
  return FeatureExtractor(architecture,
                          spatial_feature=spatial_feature,
                          imagenet_logits=imagenet_logits)

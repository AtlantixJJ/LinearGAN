# python 3.7
"""Helper functions."""

import torch
from .semantic_extractor import EXTRACTOR_POOL
from .model_settings import MODEL_POOL
from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan_encoder import StyleGANEncoder
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator
from .perceptual_model import PerceptualModel

__all__ = ['build_generator', 'build_discriminator', 'build_encoder',
           'build_perceptual', 'build_semantic_extractor', 'load_semantic_extractor', 'save_semantic_extractor']


def load_semantic_extractor(fpath):
  """Load semantic extractor from a pth path."""
  data = torch.load(fpath)
  try:
    SE_type = data["arch"]["type"]
  except: # a bug in code
    print(f"!> {fpath} data incomplete, use LSE data.")
    lse_data = torch.load(fpath.replace("NSE-2", "LSE"))
    data["arch"] = {
      "ksize" : 3,
      "type" : "NSE-2",
      "n_class" : lse_data["arch"]["n_class"],
      "dims" : lse_data["arch"]["dims"],
      "layers" : lse_data["arch"]["layers"]}
    torch.save(data, fpath)
    SE_type = data["arch"]["type"]
  SE = EXTRACTOR_POOL[SE_type](**data["arch"])
  SE.load_state_dict(data["param"])
  return SE

def save_semantic_extractor(SE, fpath, train_info={}):
  data = {
    "arch" : SE.arch_info(),
    "param" : SE.state_dict()}
  torch.save(data, fpath)


def build_semantic_extractor(model_name, n_class, dims, layers, **kwargs):
  """Builds semantic extractor by model name."""
  if model_name not in EXTRACTOR_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`EXTRACTOR_POOL` in `semantic_extractor.py`!')
  return EXTRACTOR_POOL[model_name](
    n_class=n_class,
    dims=dims,
    layers=layers,
    **kwargs)


def build_generator(model_name, logger=None, **kwargs):
  """Builds generator module by model name."""
  if model_name not in MODEL_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`MODEL_POOL` in `model_settings.py`!')

  gan_type = model_name.split('_')[0]
  if gan_type in ['pggan', 'pgganinv']:
    return PGGANGenerator(model_name, logger=logger, **kwargs)
  if gan_type in ['stylegan', 'styleganinv']:
    return StyleGANGenerator(model_name, logger=logger, **kwargs)
  if gan_type in ['stylegan2', 'stylegan2inv']:
    return StyleGAN2Generator(model_name, logger=logger, **kwargs)
  raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_discriminator(model_name, logger=None):
  """Builds discriminator module by model name."""
  if model_name not in MODEL_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`MODEL_POOL` in `model_settings.py`!')

  gan_type = model_name.split('_')[0]
  if gan_type in ['pggan', 'pgganinv']:
    return PGGANDiscriminator(model_name, logger=logger)
  if gan_type in ['stylegan', 'styleganinv']:
    return StyleGANDiscriminator(model_name, logger=logger)
  if gan_type in ['stylegan2', 'stylegan2inv']:
    return StyleGAN2Discriminator(model_name, logger=logger)
  raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_encoder(model_name, logger=None):
  """Builds encoder module by model name."""
  if model_name not in MODEL_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`MODEL_POOL` in `model_settings.py`!')

  gan_type = model_name.split('_')[0]
  if gan_type == 'styleganinv':
    return StyleGANEncoder(model_name, logger=logger)
  raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


build_perceptual = PerceptualModel

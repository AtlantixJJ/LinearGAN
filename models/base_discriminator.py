# python 3.7
"""Contains the base class for discriminator in a GAN model.

This class is derived from the `BaseModule` class defined in `base_module.py`.
"""

import numpy as np

from .base_module import BaseModule

__all__ = ['BaseDiscriminator']


class BaseDiscriminator(BaseModule):
  """Base class for discriminator used in GAN variants."""

  def __init__(self, model_name, logger=None):
    """Initializes the discriminator with model name."""
    super().__init__(model_name, 'discriminator', logger)

  def preprocess(self, images):
    """Preprocesses the input images if needed.

    This function assumes the input numpy array is with shape [batch_size,
    height, width, channel]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, channel, height, width].

    NOTE: The channel order of input images is always assumed as `RGB`.

    Args:
      images: The raw inputs with dtype `numpy.uint8` and range [0, 255].

    Returns:
      The preprocessed images with dtype `numpy.float32` and range
        [self.min_val, self.max_val].

    Raises:
      ValueError: If the input `images` are not with type `numpy.ndarray` or not
        with dtype `numpy.uint8` or not with shape [batch_size, height, width,
        channel].
    """
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')
    if images.dtype != np.uint8:
      raise ValueError(f'Images should be with dtype `numpy.uint8`!')

    if images.ndim != 4 or images.shape[3] not in [1, 3]:
      raise ValueError(f'Input should be with shape [batch_size, height, '
                       f'width, channel], where channel equals to 1 or 3!\n'
                       f'But {images.shape} is received!')
    if images.shape[3] == 1 and self.image_channels == 3:
      images = np.tile(images, (1, 1, 1, 3))
    if images.shape[3] != self.image_channels:
      raise ValueError(f'Number of channels of input image, which is '
                       f'{images.shape[3]}, is not supported by the current '
                       f'discriminator, which requires {self.image_channels} '
                       f'channels!')
    if self.image_channels == 3 and self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]
    images = images.astype(np.float32)
    images = images / 255.0 * (self.max_val - self.min_val) + self.min_val
    images = images.astype(np.float32).transpose(0, 3, 1, 2)

    return images

  def rate(self, images, **kwargs):
    """Rates the input images with adversarial score.

    NOTE: The images are assumed to have already been preprocessed.

    Args:
      images: Input images to rate.

    Returns:
      A dictionary whose values are raw outputs from the discriminator. Keys of
        the dictionary usually include `image` and `score`.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def easy_rate(self, images, **kwargs):
    """Wraps functions `preprocess()` and `rate()` together."""
    return self.rate(self.preprocess(images), **kwargs)

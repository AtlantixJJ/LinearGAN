"""Pytorch or Numpy operation functions.
"""
import torch
import torch.nn.functional as F
import numpy as np
import math


def gaussian_func(x, mean, std):
  coef = 1 / math.sqrt(2 * math.pi) / std
  vals = np.exp(- (x - mean) ** 2 / 2 / std / std)
  return coef * vals


def get_mixwp(G, N=1):
  """Sample a mixed W+ latent vector. Available for StyleGANs."""
  L = G.num_layers
  zs = torch.randn(N * L, 512).cuda()
  return G.mapping(zs).view(N, L, -1)


def mixwp_sample(G, N):
  """If G is a StyleGAN, return a mixed wp. 
  If G is PGGAN, return a normal latent vector.
  
  Args:
    G : The generator.
    N : The number of samples.
  Returns:
    N sampled mixed wp or normal latent code.
  """
  if hasattr(G, "mapping"):
    return get_mixwp(G, N)
  return torch.randn(N, 512).cuda() # hardcode


def _generate_image(G, wp):
  if hasattr(G, "synthesis"):
    return G.synthesis(wp)
  return G(wp)


def generate_images(G, wp, size=256, split=True):
  """Divide the input to have batch size 1 and resize the output.

  Args:
    G : The generator.
    wp : The mixed latent for StyleGAN, or normal latent code for PGGAN.
    size : The final output size.
    split : When set to True, the output is cpu. When set to False, the output is gpu. Usually, set to False if the wp is small.
  Returns:
    The generated image resized.
  """
  if split:
    images = []
    for i in range(wp.shape[0]):
      img = _generate_image(G, wp[i:i+1])
      if img.size(3) != size:
        img = bu(img, size)
      images.append(img.detach().cpu())
    return torch.cat(images, 0)
  else:
    img = _generate_image(G, wp[i:i+1])
    if images.size(3) != size:
      images = bu(images, size)
    return images


def bu_numpy(img, size):
  """Bilinearly resize a numpy array, using Pytorch backend.

  Args:
    img: A (N, H, W, C) numpy array
  Returns:
    An image with size (N, size[0], size[1], C) numpy array,
    scaled using bilinear interpolation with PyTorch
  """
  t = torch.from_numpy(img).permute(0, 3, 1, 2)
  return bu(t, size).numpy().transpose(0, 2, 3, 1)


def bu(img, size, align_corners=True):
  """Bilinear interpolation with Pytorch.

  Args:
    img : a list of tensors or a tensor.
  """
  if type(img) is list:
    return [F.interpolate(i,
      size=size, mode='bilinear', align_corners=align_corners)
      for i in img]
  return F.interpolate(img,
    size=size, mode='bilinear', align_corners=align_corners)


def torch2numpy(x):
  return x.detach().cpu().numpy()


def torch2image(x):
  """Convert torch tensor to be numpy array format
     image in (N, H, W, C) in [0, 255] scale
  """
  x = ((x.detach().clamp(-1, 1) + 1) * 127.5).cpu().numpy()
  return x.transpose(0, 2, 3, 1).astype("uint8")


def image2torch(x):
  """Process [0, 255] (N, H, W, C) numpy array format 
     image into [0, 1] scale (N, C, H, W) torch tensor.
  """
  y = torch.from_numpy(x).float() / 127.5 - 1
  if len(x.shape) == 3 and x.shape[2] == 3:
    return y.permute(2, 0, 1).unsqueeze(0)
  if len(x.shape) == 4:
    return y.permute(0, 3, 1, 2)

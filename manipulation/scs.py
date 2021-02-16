# python 3.7
"""Semantic-Precise Image Editing."""

import sys, os
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib.op import generate_images, bu


class SCS(object):
  """Functional interface for Semantic-Conditional Sampling."""
  
  @staticmethod
  def __sseg_se(SE, G, tar, n_init, edit_strategy, pred=False):
    """
    Conditional sampling with semantic extractor
    using random initialization.
    Args:
      SE : The semantic extractor, can be linear or nonlinear.
      G : The generator to be edited.
      tar : The target segmentation mask.
      n_init : The number of initialized z.
      n_iter : The total optimization number.
      ls : currently fixed to mixwp, which is to use a different z for each layer.
    """

    def get_seg(wp, size):
      if pred:
        image = G.synthesis(wp)
        seg = SE.raw_prediction(image, size=size)
        return seg
      image, feature = G.synthesis(wp, generate_feature=True)
      seg = SE(feature, size=size, last_only=True)[0][0]
      return seg

    BS = 4
    interval = 10
    z = torch.randn(n_init, 1, 512)
    scores = []
    for i in range(n_init):
      with torch.no_grad():
        w = G.mapping(z[i].cuda()) # (1, 512)
        wp = w.unsqueeze(1).repeat(1, G.num_layers, 1) # (1, 14, 512)
        seg = get_seg(wp, tar.shape[2])
        label = seg.argmax(1)
        scores.append((label == tar).sum())
    indice = np.argsort(scores)
    z_ = z[indice[:n_init]].clone()
    best_ind = indice[-1]

    z = z[best_ind] # (1, 512)
    z0 = z.clone().detach().cuda()
    edit_strategy.setup(z)
    for i in range(edit_strategy.n_iter):
      z, wp = edit_strategy.to_std_form()
      seg = get_seg(wp, tar.shape[2])
      celoss = torch.nn.functional.cross_entropy(seg, tar)
      regloss = 1e-3 * ((z-z0) ** 2).sum() / z.shape[0]
      edit_strategy.step(celoss + regloss)
    #print(f"=> Final loss: {celoss + regloss:.3f}")
    return z, wp


  @staticmethod
  def sseg_se(SE, G, tar, pred=False, n_init=10, repeat=1,
              latent_strategy='mixwp',
              optimizer="adam",
              n_iter=100,
              base_lr=0.01):
    edit_strategy = EditStrategy(G=G,
                                 latent_strategy=latent_strategy,
                                 optimizer=optimizer,
                                 n_iter=n_iter,
                                 base_lr=base_lr)

    res = []
    for i in tqdm(range(tar.shape[0])):
      for j in range(repeat):
        res.append(ConditionalSampling.__sseg_se(SE, G.net,
          tar[i:i+1].cuda(), n_init, edit_strategy, pred))
    agg = []
    for i in range(len(res[0])):
      agg.append(torch.cat([r[i] for r in res]))
    return agg


if __name__ == "__main__":
  import argparse, os
  from lib.misc import imread, listkey_convert, set_cuda_devices

  parser = argparse.ArgumentParser()
  parser.add_argument('--SE', type=str, default='expr',
    help='The path to the experiment directories.')
  parser.add_argument('--out-dir', type=str, default='results/scs',
    help='The output directory.')
  parser.add_argument('--repeat', type=int, default=10,
    help='The output directory.')
  parser.add_argument('--repeat-ind', type=int, default=0,
    help='The output directory.')
  parser.add_argument('--gpu-id', default='0',
    help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--optimizer', type=str, default="adam",
    help='The optimizer type.')
  parser.add_argument('--base-lr', type=float, default=0.001,
    help='The base learning rate of optimizer.')
  parser.add_argument('--n-iter', default=50, type=int,
    help='The number of iteration in editing.')
  parser.add_argument('--n-init', default=10, type=int,
    help='The number of initializations.')
  parser.add_argument('--latent-strategy', default='z',
    help='The latent space strategy.')
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)

  from models.helper import build_generator, load_semantic_extractor
  from predictors.face_segmenter import FaceSegmenter
  from predictors.scene_segmenter import SceneSegmenter

  print(f"=> Loading from {args.SE}")
  if "baseline" in args.SE:
    SE_name = args.SE
  else:
    SE = load_semantic_extractor(args.SE)
    SE.cuda().eval()
    SE_name = args.SE[args.SE.rfind("/") + 1: args.SE.rfind("LSE")]
  print(SE_name)
  G_name = listkey_convert(args.SE,
    ["stylegan2_ffhq", "stylegan2_bedroom", "stylegan2_church"])
  G = build_generator(G_name)
  if "ffhq" in G_name:
    image_ids = np.random.RandomState(1116).choice(list(range(2000)), (100,))
    labels = np.stack([imread(f"../datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15/{i}.png")
      for i in image_ids])[:, :, :, 0] # (N, H, W, 3)
    labels = torch.from_numpy(labels).long()
    P = FaceSegmenter()
  else:
    P = SceneSegmenter(model_name=G_name)
    wp = np.load(f"data/trunc_{G_name}/wp.npy")
    wp = torch.from_numpy(wp).float().cuda()
    with torch.no_grad():
      images = generate_images(G.net, wp)
      labels = torch.cat([P(img.unsqueeze(0).cuda())[0] for img in images], 0)
  print(labels.shape)
  pred = "baseline" in SE_name
  P_ = P if pred else SE
  out_name = f"{G_name}_{SE_name}_repeat{args.repeat_ind}"
  if pred:
    out_name = SE_name
  z, wp = ConditionalSampling.sseg_se(P_, G, labels,
    n_iter=args.n_iter, n_init=args.n_init,
    pred=pred, repeat=args.repeat,
    latent_strategy=args.latent_strategy)
  torch.save([z, wp], f"{args.out_dir}/{out_name}.pth")
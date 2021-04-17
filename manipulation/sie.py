# python 3.7
"""Semantic Image Editing."""

import sys, glob
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils

from lib.op import generate_images, bu, torch2image
from lib.misc import imread
from lib.visualizer import get_label_color, segviz_torch
from home.utils import color_mask, preprocess_label, preprocess_mask, preprocess_image
from models.helper import load_semantic_extractor, build_generator
from predictors.helper import build_predictor
from manipulation.strategy import EditStrategy


class ImageEditing(object):
  """Functional interface for Semantic-Precise Image Editing."""

  @staticmethod
  def __sseg_se_z(P, G, z, tar, tar_mask, edit_strategy):
    """
    Precise semantic editing using semantic extractor.
    Args:
      P : The semantic extractor.
      G : The generator supporting the edit.
      z : The initial z to be edited.
      tar : The target semantic mask.
      tar_mask : Denoting user changed region. But not used currently.
    """
    edit_strategy.setup(z)
    z0 = edit_strategy.z0 # (14, 512)
    for i in (range(edit_strategy.n_iter)):
      z, wps = edit_strategy.to_std_form() # (14, 512)
      image, feature = G.synthesis(wps, generate_feature=True)
      seg = P(feature, size=tar.shape[2])[-1]
      celoss = torch.nn.functional.cross_entropy(seg, tar)
      regloss = 1e-3 * ((z-z0) ** 2).sum()
      priorloss = 1e-3 * (z ** 2).sum() / z.shape[0]
      edit_strategy.step(celoss + regloss + priorloss)
    return edit_strategy.to_std_form()

  @staticmethod
  def __sseg_pred_z(P, G, z, tar, tar_mask, edit_strategy):
    """
    Do editing on z (1, 512)
    """
    edit_strategy.setup(z)
    z0 = edit_strategy.z0
    for i in (range(edit_strategy.n_iter)):
      z, wps = edit_strategy.to_std_form()
      image = G.synthesis(wps)
      seg = P.raw_prediction(image, size=tar.shape[2])
      celoss = torch.nn.functional.cross_entropy(seg, tar)
      regloss = 1e-3 * ((z-z0) ** 2).sum()
      priorloss = 1e-3 * (z ** 2).sum() / z.shape[0]
      edit_strategy.step(celoss + regloss + priorloss)
    return edit_strategy.to_std_form()

  @staticmethod
  def __sseg_image_z(G, z, tar, tar_mask, edit_strategy, P=None):
    """
    Precise semantic editing using semantic extractor.
    Args:
      P : The semantic extractor.
      G : The generator supporting the edit.
      z : The initial z to be edited.
      tar : The target semantic mask.
      tar_mask : Denoting user changed region. But not used currently.
    """
    edit_strategy.setup(z)
    z0 = edit_strategy.z0
    for i in (range(edit_strategy.n_iter)):
      z, wps = edit_strategy.to_std_form()
      image = bu(G.synthesis(wps), tar.shape[2])
      diff = (tar - image) ** 2
      mseregloss = ((1 - tar_mask) * diff).sum() / (1 - tar_mask).sum()
      mseeditloss = (tar_mask * diff).sum() / tar_mask.sum()
      regloss = 1e-3 * ((z-z0) ** 2).sum()
      priorloss = 1e-3 * (z ** 2).sum() / z.shape[0]
      edit_strategy.step(mseregloss + mseeditloss + regloss + priorloss)
    return edit_strategy.to_std_form()

  @staticmethod
  def sseg_edit(G, z, tar, tar_mask,
                SE=None, 
                op="internal",
                latent_strategy='mixwp',
                optimizer="adam",
                n_iter=100,
                base_lr=0.01):
    """
    Args:
      G : The generator
      z : The latent z (N, 512), or mixed latent z (N, layers, 512)
      op : ``internal'' means to use LSE, ``external'' means to use predictor, ``image'' means to use color space editing loss
    """
    res = []
    edit_strategy = EditStrategy(G=G,
                                 latent_strategy=latent_strategy,
                                 optimizer=optimizer,
                                 n_iter=n_iter,
                                 base_lr=base_lr)
    func = {
      "internal" : ImageEditing.__sseg_se_z,
      "external" : ImageEditing.__sseg_pred_z,
      "image" : ImageEditing.__sseg_image_z}[op]
    for i in range(tar.shape[0]):
      res.append(func(
        P=SE, G=G, z=z[i:i+1].float(),
        tar=tar[i:i+1].long().cuda(),
        tar_mask=tar_mask[i:i+1].float().cuda(),
        edit_strategy=edit_strategy))
    agg = []
    for i in range(len(res[0])):
      agg.append(torch.cat([r[i] for r in res]))
    return agg

  @staticmethod
  def fuse_stroke(G, SE, P, wp,
      image_stroke, image_mask, label_stroke, label_mask):
    """
    Args:
      image_stroke : [3, H, W]
      image_mask : [1, H, W]
      label_stroke : [H, W]
      label_mask : [H, W]
    """
    size = label_mask.shape[1]
    image, feature = G.synthesis(wp, generate_feature=True)
    origin_image = bu(image, size=size).cpu()
    int_label = SE(feature, size=size)[-1].argmax(1).cpu() if SE else None
    ext_label = P(image, size=size).cpu() if P else None

    m = label_mask.cpu()
    fused_int_label = None if label_stroke is None or int_label is None else \
      ((1 - m) * int_label + m * label_stroke.cpu()).long() 
    fused_ext_label = None if label_stroke is None or ext_label is None else \
      ((1 - m) * ext_label + m * label_stroke.cpu()).long()

    m = image_mask.cpu()
    fused_image = None if image_stroke is None else \
      (1 - m) * origin_image + m * image_stroke.cpu()

    return {
        "fused_image" : fused_image,
        "fused_int_label" : fused_int_label,
        "fused_ext_label" : fused_ext_label,
        "origin_image" : origin_image,
        "int_label" : int_label,
        "ext_label" : ext_label}


def read_data(data_dir, name_list, n_class=15):
  z, image_stroke, label_stroke, image_mask, label_mask = [], [], [], [], []
  for name in name_list:
    name = name[:name.rfind("_")]
    files = glob.glob(f"{data_dir}/{name}*")
    files.sort() # img_m, img_s, lbl_m, lbl_s

    image_mask.append(preprocess_mask(imread(files[0])[:, :, 0]))
    image_stroke.append(preprocess_image(imread(files[1])))
    label_mask.append(preprocess_mask(imread(files[2])[:, :, 0]))
    label_stroke.append(preprocess_label(imread(files[3]), n_class))
    zs = torch.from_numpy(np.load(files[-1]))
    z.append(zs.float().unsqueeze(0))
  res = [z, image_stroke, image_mask, label_stroke, label_mask]
  return [torch.cat(r) for r in res]


if __name__ == "__main__":
  import argparse
  from lib.misc import set_cuda_devices

  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', type=str, default='data/collect_ffhq',
    help='The input directory.')
  parser.add_argument('--name-list', type=str, default='figure/sie_list.txt',
    help='The list of names.')
  parser.add_argument('--out-dir', type=str, default='results/sie',
    help='The output directory.')
  parser.add_argument('--gpu-id', default='0',
    help='Which GPU(s) to use. (default: `0`)')
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)

  name_list = open(args.name_list, "r").readlines()
  z, image_stroke, image_mask, label_stroke, label_mask = \
    read_data(args.data_dir, name_list)
  param_dict = dict(latent_strategy="mixwp",
                    optimizer='adam',
                    n_iter=50,
                    base_lr=0.002)
  print(z.shape, image_stroke.shape, image_mask.shape, label_stroke.shape, label_mask.shape)
  G_name = "stylegan2_ffhq"
  DIR = "predictors/pretrain/"
  G = build_generator(G_name).net
  P = build_predictor("face_seg")
  SE_full = load_semantic_extractor(f"{DIR}/{G_name}_LSE.pth").cuda()
  SE_fewshot = load_semantic_extractor(f"{DIR}/{G_name}_8shot_LSE.pth").cuda()

  proc = lambda x : ((bu(x.detach().cpu(), 256) + 1) / 2)
  vizproc = lambda x : (bu(segviz_torch(
    x.detach().cpu()).unsqueeze(0), 256))
  origins, labels, baselines, fewshots, fulls = [], [], [], [], []
  for i in tqdm(range(z.shape[0])):
    zs = z[i].cuda()
    with torch.no_grad():
      wp = EditStrategy.z_to_wp(
        G, zs, in_type="zs", out_type="notrunc-wp")

    res = ImageEditing.fuse_stroke(
      G, SE_full, P, wp,
      image_stroke[i], image_mask[i],
      label_stroke[i], label_mask[i])
    origins.append(proc(res["origin_image"]))
    labels.append(vizproc(res["ext_label"]))
    labels.append(vizproc(res["fused_ext_label"]))

    fused_label_fewshot = ImageEditing.fuse_stroke(
      G, SE_fewshot, None, wp,
      image_stroke[i], image_mask[i],
      label_stroke[i], label_mask[i])["fused_int_label"]

    wp_full = ImageEditing.sseg_edit(
      G, zs, res["fused_int_label"], label_mask, SE_full,
      op="internal", **param_dict)[1]

    wp_fewshot = ImageEditing.sseg_edit(
      G, zs, fused_label_fewshot, label_mask, SE_full,
      op="internal", **param_dict)[1]

    wp_baseline = ImageEditing.sseg_edit(
      G, zs, res["fused_ext_label"], label_mask, P,
      op="external", **param_dict)[1]

    with torch.no_grad():
      image = G.synthesis(wp_baseline)
      baselines.extend([proc(image), vizproc(P(image, size=256))])

      image, feature = G.synthesis(wp_fewshot, generate_feature=True)
      label = SE_fewshot(feature, size=256)[-1].argmax(1)
      fewshots.extend([proc(image), vizproc(label)])

      image, feature = G.synthesis(wp_full, generate_feature=True)
      label = SE_fewshot(feature, size=256)[-1].argmax(1)
      fulls.extend([proc(image), vizproc(label)])

  origins = vutils.make_grid(torch.cat(origins),
    nrow=1, padding=10, pad_value=255)
  labels = vutils.make_grid(torch.cat(labels),
    nrow=2, padding=10, pad_value=255)
  baselines = vutils.make_grid(torch.cat(baselines),
    nrow=2, padding=10, pad_value=255)
  fewshots = vutils.make_grid(torch.cat(fewshots),
    nrow=2, padding=10, pad_value=255)
  fulls = vutils.make_grid(torch.cat(fulls),
    nrow=2, padding=10, pad_value=255)
  pads = torch.zeros((3, baselines.size(1), 40))
  pads.fill_(255)
  vutils.save_image(torch.cat([origins, pads, labels, pads,
    baselines, pads, fewshots, pads, fulls], 2),
    f"{args.out_dir}/ffhq.png")

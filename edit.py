import sys, argparse, glob, os
sys.path.insert(0, ".")
import torch
from tqdm import tqdm
import numpy as np
from utils.visualizer import segviz_torch, segviz_numpy, get_label_color
from utils.op import bu, torch2numpy
from utils.misc import imread, imwrite, listkey_convert, set_cuda_devices
from home.utils import color_mask
import torchvision.utils as vutils
import cv2



def fuse_stroke(G, SE, P, zs, image_strokes, image_masks, label_strokes, label_masks):
  int_label = []
  ext_label = []
  fused_label = []
  fused_image = []
  origin_image = []
  with torch.no_grad():
    for i in range(zs.shape[0]):
      mask = label_masks[i:i+1]
      w = G.mapping(zs[i:i+1].cuda())
      wp = w.unsqueeze(1).repeat(1, G.num_layers, 1)
      image, feature = G.synthesis(wp, generate_feature=True)
      image = bu(image, size=mask.shape[1])
      origin_image.append(image)
      seg = SE(feature, last_only=True, size=mask.shape[1])
      est_label = seg[0][0].argmax(1).cpu() # (1, H, W)
      int_label.append(est_label)
      ext_label.append(P(image, size=mask.shape[1])[0])
      tar = ((1 - mask) * est_label + mask * label_strokes[i:i+1]).long()
      fused_label.append(tar)
      mask = image_masks[i:i+1]
      img = (1 - mask) * image.cpu() + mask * image_strokes[i:i+1]
      fused_image.append(img)
      tar_viz = vutils.save_image(
        segviz_torch(tar).unsqueeze(0),
        f"labelrec_{i}.png")
  fused_image = torch.cat(fused_image).float()
  fused_label = torch.cat(fused_label)
  ext_label = torch.cat(ext_label)
  int_label = torch.cat(int_label)
  origin_image = torch.cat(origin_image)
  return origin_image, fused_image, fused_label, int_label, ext_label


if __name__ == "__main__":
  G2SE = {
    "stylegan2_ffhq" : "predictors/pretrain/LSE_15_512,512,512,512,512,512,512,512,512,256,256,128,128,64,64,32,32_0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16.pth"}

  parser = argparse.ArgumentParser()
  parser.add_argument('--G', type=str, default='stylegan2_ffhq',
    help='The output directory.')
  parser.add_argument('--out-dir', type=str, default='results/edit/',
    help='The output directory.')
  parser.add_argument('--op', type=str, default='internal',
    help='internal, external')
  parser.add_argument('--data-dir', type=str, default="data/collect_data_ffhq",
    help='The output directory.')
  parser.add_argument('--optimizer', type=str, default="adam",
    help='The optimizer type.')
  parser.add_argument('--base-lr', type=float, default=0.01,
    help='The base learning rate of optimizer.')
  parser.add_argument('--gpu-id', default='0',
    help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--n-iter', default=100, type=int,
    help='The number of iteration in editing.')
  parser.add_argument('--latent-strategy', default='mixwp',
    help='The latent space strategy.')
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)

  import metric
  from dataset import NoiseDataModule
  from models.semantic_extractor import load_from_pth_LSE, save_to_pth, SELearner
  from utils.editor import ImageEditing
  from models.helper import build_generator, build_semantic_extractor
  from predictors.face_segmenter import FaceSegmenter
  from predictors.scene_segmenter import SceneSegmenter

  res = read_data(args.data_dir)
  zs, image_strokes, image_masks, label_strokes, label_masks = res
  
  SE = load_from_pth_LSE(G2SE[args.G])
  SE.cuda().eval()
  G = build_generator(args.G)
  if "ffhq" in args.G:
    P = FaceSegmenter()
  else:
    P = SceneSegmenter(model_name=args.G)
  
  _, fused_image, fused_label, _, _ = fuse_stroke(G.net, SE, P, *res)

  if args.op == "":
    for op, P_ in zip(["image", "fewshot", "external"], [None, SE, P]):
      tar, mask = fused_label, label_masks
      if "image" == op:
        tar, mask = fused_image, image_masks
      z, wp = ImageEditing.sseg_edit(
        G, zs, tar, mask, P_,
        op=op,
        latent_strategy=args.latent_strategy,
        optimizer=args.optimizer,
        n_iter=args.n_iter,
        base_lr=args.base_lr)
      torch.save([z, wp], f"{args.out_dir}/{args.G}_{op}_{args.n_iter}.pth")
  elif args.op == "fewshot-face":
    #bedroom_path = "expr/fewshot/stylegan2_bedroom_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r0_n{num_sample}LSE_16_512,512,512,512,512,512,512,512,512,256,256,128,128_0,1,2,3,4,5,6,7,8,9,10,11,12.pth"
    ffhq_path = "expr/fewshot/stylegan2_ffhq_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r0_n{num_sample}LSE_15_512,512,512,512,512,512,512,512,512,256,256,128,128,64,64,32,32_0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16.pth"
    for num_sample in [1, 4, 8, 16]:
      print(f"=> Num sample: {num_sample}")
      SE = load_from_pth_LSE(ffhq_path.format(num_sample=num_sample))
      SE.cuda().eval()
      _, fused_image, fused_label, _, _ = fuse_stroke(G.net, SE, P, *res)
      z, wp = ImageEditing.sseg_edit(
        G, zs, fused_label, label_masks, SE,
        op='fewshot',
        latent_strategy=args.latent_strategy,
        optimizer=args.optimizer,
        n_iter=args.n_iter,
        base_lr=args.base_lr)
      torch.save([z, wp], f"{args.out_dir}/fewshot{num_sample}-{args.G}_internal_{args.n_iter}.pth")
  else:
    tar, mask = fused_label, label_masks
    if "image" == args.op:
      tar, mask = fused_image, image_masks
    P_ = None
    if "internal" in args.op:
      P_ = SE
    if "external" in args.op:
      P_ = P
    z, wp = ImageEditing.sseg_edit(
      G, zs, tar, mask, P_,
      op=args.op,
      latent_strategy=args.latent_strategy,
      optimizer=args.optimizer,
      n_iter=args.n_iter,
      base_lr=args.base_lr)
    torch.save([z, wp], f"{args.out_dir}/{args.G}_{args.op}_{args.n_iter}.pth")
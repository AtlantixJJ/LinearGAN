import sys, argparse, glob, os
sys.path.insert(0, ".")
import torch
from tqdm import tqdm
import numpy as np
from lib.dataset import NoiseDataModule
from models.semantic_extractor import SELearner
from models.helper import *
from predictors.face_segmenter import FaceSegmenter
from predictors.scene_segmenter import SceneSegmenter


def G_from_SE(fpath):
  name = fpath.split("/")[-2]
  model_name = "_".join(name.split("_")[:2])
  return model_name, build_generator(model_name).net


def eval_SE(SE_path, num, save_path, latent_strategy):
  SE_args = SE_path.split("/")[-2] + f"_els{latent_strategy}"
  if os.path.exists(f"{save_path}/{SE_args}_evaluation.pth"):
    print("=> evaluation result exists, skip.")
    return
  G_name, G = G_from_SE(SE_path)
  is_face = "celebahq" in SE_path or "ffhq" in SE_path
  is_feature_norm = "LSE-F" in SE_path or "LSE-WF" in SE_path
  P = FaceSegmenter() if is_face else SceneSegmenter(model_name=G_name)

  resolution = 512 if is_face and not is_feature_norm else 256
  SE = load_from_pth(SE_path)
  SE.cuda().eval()
  res = evaluate_SE(SE, G, P, resolution, num, latent_strategy)
  fpath = f"{save_path}/{SE_args}_evaluation.pth"
  res_path = fpath.replace(".pth", ".txt")
  torch.save(res, fpath)
  mIoU, c_iou = aggregate_iou(res)

  with open(res_path, "w") as f:
    c_ious = [float(i) for i in c_iou]
    s = [str(c) for c in c_ious]
    f.write(str(float(mIoU)) + "\n")
    f.write(" ".join(s))
  return mIoU, c_ious


def aggregate_iou(res):
  # r[0] is pixelacc, r[1] is IoU
  ic_iou = torch.stack([r[1] for r in res])
  c_iou = torch.zeros(ic_iou.shape[1])
  #print(ic_iou.shape, c_iou.shape)
  for c in range(ic_iou.shape[1]):
    val = ic_iou[:, c]
    val = val[val > -0.1]
    c_iou[c] = -1 if val.shape[0] == 0 else val.mean()
  mIoU = c_iou.mean()
  return mIoU, c_iou


def evaluate_SE(SE, G, P, resolution, num, ls='trunc-wp'):
  learner = SELearner(SE, G, P,
    resolution=resolution, latent_strategy=ls).cuda()
  res = []
  for i in tqdm(range(num)):
    with torch.no_grad():
      seg, label = learner(torch.randn(1, 512).cuda())

    for cat_id in range(len(SE.category_groups)):
      dt = seg[cat_id][-1].argmax(1)
      gt = label[cat_id]
      IoU = iou(dt, gt, num_classes=SE.n_class,
        ignore_index=0, absent_score=-1, reduction='none')
      pixelacc = (dt == gt).sum() / float(dt.shape.numel())
      res.append([pixelacc, IoU])
  return res


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--SE', type=str, default='expr',
    help='The path to the experiment directories.')
  parser.add_argument('--out-dir', type=str, default='results/semantics/',
    help='The output directory.')
  parser.add_argument('--num', type=int, default=10000,
    help='The evaluation sample.')
  parser.add_argument('--gpu-id', default='0',
    help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--latent-strategy', default='trunc-wp',
    help='trunc-wp, notrunc-mixwp')
  args = parser.parse_args()

  # These import will affect cuda device setting
  from utils.misc import set_cuda_devices
  n_gpus = set_cuda_devices(args.gpu_id)

  if os.path.isdir(args.SE):
    dirs = glob.glob(args.SE + "/*SE*")
    dirs.sort()
    model_paths = []
    for d in dirs:
      models = glob.glob(d + "/*.pth")
      if len(models) == 0:
        continue
      models.sort()
      model_path = models[0]
      if ".pth" not in model_path or "SE" not in model_path:
        continue
      model_paths.append(model_path)

    gpus = args.gpu_id.split(',')
    slots = [[] for _ in gpus]
    for i, m in enumerate(model_paths):
      gpu = gpus[i % len(gpus)]
      slots[i % len(gpus)].append(f"CUDA_VISIBLE_DEVICES={gpu} python3 script/semantics/evaluate_models.py --gpu-id {gpu} --SE {m} --num {args.num} --out-dir {args.out_dir} --latent-strategy {args.latent_strategy}")
    for s in slots:
      cmd = " && ".join(s) + " &"
      print(cmd)
      os.system(cmd)

  if ".pth" in args.SE:
    eval_SE(args.SE, args.num, args.out_dir, args.latent_strategy)

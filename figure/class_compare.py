import torch, sys, os, argparse, glob
sys.path.insert(0, ".")
from figure.methods_compare import *
from utils.misc import listkey_convert, str_latex_table, read_selected_labels
from utils.op import torch2numpy
import numpy as np
from predictors.face_segmenter import CELEBA_CATEGORY


def formal_name(name):
  if type(name) is list:
    return [formal_name(n) for n in name]
  finds = ["stylegan", "pggan", "bedroom", "Church", "celebahq", "ffhq"]
  subs = ["StyleGAN", "PGGAN", "LSUN-Bedroom", "LSUN-Church", "CelebAHQ", "FFHQ"]
  for find, sub in zip(finds, subs):
    name = name.replace(find, sub)
  return name


def get_args_name(optims=["adam-0.01"], total_linears=[1], layer_detachs=[0], loss_types=["focal"], ls=""):
  for optim in optims:
    for total_linear in total_linears:
      for layer_detach in layer_detachs:
        for loss_type in loss_types:
          yield 


def aggregate_iou(res):
  # r[0] is pixelacc, r[1] is IoU
  ic_iou = torch.stack([r[1] for r in res])
  c_iou = torch.zeros(ic_iou.shape[1])
  #print(ic_iou.shape, c_iou.shape)
  for c in range(ic_iou.shape[1]):
    val = ic_iou[:, c]
    val = val[val > -0.1]
    c_iou[c] = -1 if val.shape[0] == 0 else val.mean()
    #if c_iou[c] > 0:
    #  print(f"{c} : {c_iou[c]} : {val[val > -0.1].shape}")
  mIoU = c_iou[c_iou > -1].mean()
  return mIoU, c_iou


def iou_from_pth(fpath, force_calc=False):
  if "NSE-2" in fpath:
    fpath = fpath.replace("adam-0.01", "adam-0.001")
  res_path = fpath.replace(".pth", ".txt")
  if os.path.exists(res_path) and not force_calc:
    with open(res_path, "r") as f:
      mIoU = float(f.readline().strip())
      c_iou = [float(i) for i in f.readline().strip().split(" ")]
    return mIoU, c_iou
  if not os.path.exists(fpath):
    ind = fpath.find("focal")
    files = glob.glob(fpath[:ind] + "*")
    print(files)
    fpath = files[0]
  if not os.path.exists(fpath):
    print(f"=> skip {fpath}")
    return -1
  
  # pixelacc, IoU, AP, AR, mIoU, mAP, mAR, sups
  res = torch.load(fpath, map_location='cpu')
  mIoU, c_iou = aggregate_iou(res)
  with open(res_path, "w") as f:
    c_ious = [float(i) for i in c_iou]
    s = [str(c) for c in c_ious]
    f.write(str(float(mIoU)) + "\n")
    f.write(" ".join(s))
  return mIoU, c_ious


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/semantics", help="")
  parser.add_argument("--name", default="all-concise-notrunc-teval-class")
  parser.add_argument("--force-calc", default=0, type=int)
  args = parser.parse_args()

  if "face" in args.name:
    Gs = ["stylegan2_ffhq", "stylegan_celebahq", "pggan_celebahq"]
    labels = CELEBA_CATEGORY[1:]
  else:
    Gs = [
      "stylegan2_bedroom", "stylegan2_church",
      "stylegan_bedroom", "stylegan_church", 
      "pggan_bedroom", "pggan_church"]
    

  model_label = read_selected_labels()
  #if "old" in args.dir:
  #  model_label = read_selected_labels("figure/old_selected_labels.csv")

  dic = {}
  for G in Gs:
    dic[G] = {}
    for method in ["LSE", "NSE-1", "NSE-2"]:
      if "face" not in args.name:
        labels = model_label[G]
      dic[G][method] = {k:0 for k in labels}
      optim = "adam-0.001" 
      args_name = f"{optim}_t1_d0_lfocal_lsnotrunc-mixwp_elstrunc-wp"
      fpath = f"{args.dir}/{G}_{method}_{args_name}_evaluation.pth"
      mIoU, c_iou = iou_from_pth(fpath, args.force_calc == 1)
      if mIoU < 0:
        continue
      print(G, len(c_iou), len(labels))
      for i in range(len(c_iou)):
        dic[G][method][labels[i]] = c_iou[i]

  #strs = str_table_multiple(dic)
  for k, v in dic.items():
    strs = str_table_single(dic[k], False, 0)
    with open(f"results/tex/{args.name}_{k}.tex", "w") as f:
      f.write(str_latex_table(strs))

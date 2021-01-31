import torch, sys, os, argparse, glob
sys.path.insert(0, ".")
import numpy as np

from figure.methods_compare import *
from lib.misc import str_latex_table, read_selected_labels
from lib.op import torch2numpy
from predictors.face_segmenter import CELEBA_CATEGORY
from evaluate import read_results


def formal_name(name):
  if type(name) is list:
    return [formal_name(n) for n in name]
  finds = ["stylegan", "pggan", "bedroom", "Church", "celebahq", "ffhq"]
  subs = ["StyleGAN", "PGGAN", "LSUN-Bedroom", "LSUN-Church", "CelebAHQ", "FFHQ"]
  for find, sub in zip(finds, subs):
    name = name.replace(find, sub)
  return name


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/semantics", help="")
  parser.add_argument("--name", default="all-notrunc-teval")
  parser.add_argument("--label-file", default="figure/selected_labels.csv", type=int)
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
      mIoU, c_iou = read_results(fpath)
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

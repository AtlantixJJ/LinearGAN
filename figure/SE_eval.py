"""Summarize the evaluation of all the semantic extractors."""
import torch, sys, os, argparse, glob
sys.path.insert(0, ".")
import numpy as np
from collections import OrderedDict

from lib.misc import *
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
  parser.add_argument("--label-file", default="figure/selected_labels.csv", type=str)
  args = parser.parse_args()

  face_labels = CELEBA_CATEGORY[1:]
  Gs = [
    "stylegan2_ffhq", "stylegan_celebahq", "pggan_celebahq",
    "stylegan2_bedroom", "stylegan2_church",
    "stylegan_bedroom", "stylegan_church", 
    "pggan_bedroom", "pggan_church"]
  subfix = ""# "_other" #

  if args.label_file != "":
    model_label = read_selected_labels(args.label_file)
  else:
    labels = read_ade20k_labels()[1:]
    model_label = {G : labels for G in Gs}

  dic = OrderedDict()
  gdic = OrderedDict()
  for G in Gs:
    dic[G] = OrderedDict()
    group_name = listkey_convert(G,
      ["stylegan2", "stylegan", "pggan"])
    if group_name not in gdic:
      gdic[group_name] = OrderedDict()
    ds = G.split("_")[1]
    if ds not in gdic[group_name]:
      gdic[group_name][ds] = OrderedDict()
    for method in ["LSE", "NSE-1", "NSE-2"]:
      labels = face_labels if "ffhq" in G or "celebahq" in G \
        else model_label[G]
      dic[G][method] = {k:0 for k in labels}
      args_name = f"lnormal_lstrunc-wp_lwsoftplus_lr0.001_elstrunc-wp"
      fpath = f"{args.dir}/{G}_{method}_{args_name}.txt"
      mIoU, c_iou = read_results(fpath)
      if mIoU < 0:
        continue
      for i in range(len(c_iou)):
        dic[G][method][labels[i]] = c_iou[i]
      gdic[group_name][ds][method] = mIoU

  for k, v in dic.items():
    strs = str_table_single(dic[k])
    with open(f"results/tex/SE{subfix}_{k}.tex", "w") as f:
      f.write(str_latex_table(strs))
  key_sorted = list(gdic.keys())
  key_sorted.sort()
  print(key_sorted)
  gdic = {k : gdic[k] for k in key_sorted}
  strs = str_table_multiple(gdic)
  with open(f"results/tex/SE{subfix}.tex", "w") as f:
    f.write(str_latex_table(strs))

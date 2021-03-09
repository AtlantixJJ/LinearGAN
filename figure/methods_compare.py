import torch, sys, os, argparse
sys.path.insert(0, ".")
import numpy as np

from lib.misc import *
from lib.op import torch2numpy
from evaluate import read_results


def get_table_suit(name):
  # default: full
  methods = ["LSE", "NSE-1", "NSE-2"]
  layer_weights = ["softplus"]
  loss_types = ["normal"]

  if "notrunc" in name:
    ls = "lsnotrunc-mixwp"
  else:
    ls = "lstrunc-wp"

  layer_weights = ["lwsoftplus"]
  lrs = ["lr0.001"]

  # Which latent strategy is the SE evaluated on
  if "-nteval" in name:
    els = "elsnotrunc-mixwp"
  elif "-teval" in name:
    els = "elstrunc-wp"  

  if "LSE_arch_compare" in name:
    Gs = ["stylegan2_bedroom", "stylegan_bedroom"]
    methods = ["LSE"]

  if "face" in name:
    Gs = ["pggan_celebahq", "stylegan_celebahq", "stylegan2_ffhq"]
  elif "bedroom" in name:
    Gs = ["pggan_bedroom", "stylegan_bedroom", "stylegan2_bedroom"]
  elif "church" in name:
    Gs = ["pggan_church", "stylegan_church", "stylegan2_church"]
  elif "other" in name:
    Gs = {
      "pggan" : ["pggan_bedroom", "pggan_church"],
      "stylegan" : ["stylegan_bedroom", "stylegan_church"],
      "stylegan2" : ["stylegan2_bedroom", "stylegan2_church"]}
  elif "all" in name:
    Gs = {
      "pggan" : ["pggan_celebahq", "pggan_bedroom", "pggan_church"],
      "stylegan" : ["stylegan_celebahq", "stylegan_bedroom", "stylegan_church"],
      "stylegan2" : ["stylegan2_ffhq", "stylegan2_bedroom", "stylegan2_church"]}
  return Gs, methods, loss_types, ls, layer_weights, lrs, els
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/semantics/", help="")
  parser.add_argument("--name", default="all-trunc-teval")
  parser.add_argument("--force-calc", default=0, type=int)
  args = parser.parse_args()
  params = get_table_suit(args.name)
  print(params)
  dic = {}
  for group_name, Gs in params[0].items():
    dic[group_name] = {}
    for G in Gs:
      ds = G.split("_")[1]
      dic[group_name][ds] = {}
      for args_name in get_args_name(*params[1:]):
        method = args_name.split("_")[0]
        fpath = f"{args.dir}/{G}_{args_name}.txt"
        mIoU, cious = read_results(fpath)
        if mIoU < 0:
          print(f"!> {fpath} is empty")
          continue
        dic[group_name][ds][method] = float(mIoU)
  strs = str_table_multiple(dic)

  with open(f"results/tex/{args.name}.tex", "w") as f:
    f.write(str_latex_table(strs))

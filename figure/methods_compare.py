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


def str_table_single(dic, indicate_best=True, T=0):
  for k2 in list(dic["LSE"].keys()):
    maxi = -1
    for k1 in list(dic.keys()):
      if dic[k1][k2] > maxi:
        maxi = dic[k1][k2]
    if maxi < T:
      for k1 in list(dic.keys()):
        del dic[k1][k2]
  idic = invert_dic(dic)
  methods = list(dic.keys())
  Gs = list(idic.keys())
  strs = [formal_name(Gs)]
  for method in methods:
    s = [method]
    for G in Gs:
      best_ind, best_method, best_val = max_key(idic[G])
      acc = f"{dic[method][G] * 100:.1f}"
      comp = (dic[method][G] - best_val) / (best_val + 1e-6) * 100
      comp = "*" if best_method == method else f"({comp:.1f})"
      item_str = acc
      if indicate_best:
        item_str = f"{acc} {comp}"
      s.append(item_str)
    strs.append(s)
  return strs


def str_table_multiple(dic, T=0): # group, G, method
  groups = list(dic.keys()) # 1st column name
  def latex_header(n):
    return f"\\multicolumn" + "{" + str(n) + "}" + "{c|}"
  strs = [["Generator"] + [f"{latex_header(len(dic[g].keys()))}" + \
            "{" + formal_name(g) + "}" for g in groups]]
  s = ["Dataset"]
  for g in groups:
    Gs = list(dic[groups[0]].keys()) # 2nd column name
    s.extend(formal_name(Gs))
  strs.append(s)

  s_ = []
  for group in dic.keys():
    for ds in dic[group].keys():
      best_ind, best_method, best_val = max_key(dic[group][ds])
      for i, method in enumerate(dic[group][ds].keys()):
        acc = f"{dic[group][ds][method] * 100:.1f}"
        comp = (dic[group][ds][method] - best_val) / best_val * 100
        if best_method == method:
          item_str = "\\textbf{" + acc + "}"
        else:
          item_str = f"{acc} ({comp:.1f})"
        if len(s_) <= i:
          s_.append([method])
        s_[i].append(item_str)
  strs.extend(s_)
  return strs


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
          continue
        dic[group_name][ds][method] = float(mIoU)

  strs = str_table_multiple(dic)

  with open(f"results/tex/{args.name}.tex", "w") as f:
    f.write(str_latex_table(strs))

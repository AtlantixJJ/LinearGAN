import torch, sys, os, argparse
sys.path.insert(0, ".")
import numpy as np

from lib.misc import str_latex_table, formal_generator_name, get_args_name
from lib.op import torch2numpy
from evaluate import read_results


def get_table_suit(name):
  # default: full
  methods = ["LSE", "NSE-1", "NSE-2"]
  layer_weights = ["softplus"]
  loss_types = ["focal"]

  # Which latent strategy is the SE trained on
  if "notrunc" in name:
    ls = "lsnotrunc-mixwp"
  else:
    ls = "lstrunc-wp"

  # Which latent strategy is the SE evaluated on
  if "-nteval" in name:
    els = "elsnotrunc-mixwp"
  elif "-teval" in name:
    els = "elstrunc-wp"  

  if "LSE_arch_compare" in name:
    Gs = ["stylegan2_bedroom", "stylegan_bedroom"]
    methods = ["LSE"]
    layer_weights = ["softplus", "none"]
    loss_types = ["focal", "normal"]

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
  return methods, Gs, layer_weights, loss_types, ls, els


def max_key(dic):
  keys = list(dic.keys())
  ind = np.argmax([dic[k] for k in keys])
  return ind, keys[ind], dic[keys[ind]]


def invert_dic(dic):
  idic = {}
  for k1 in dic.keys():
    for k2 in dic[k1].keys():
      if k2 not in idic:
        idic[k2] = {}
      idic[k2][k1] = dic[k1][k2]
  return idic


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
  strs = [formal_generator_name(Gs)]
  for method in methods:
    s = [method]
    for G in Gs:
      best_ind, best_method, best_val = max_key(idic[G])
      acc = f"{dic[method][G] * 100:.1f}\\%"
      comp = (dic[method][G] - best_val) / (best_val + 1e-6) * 100
      comp = "*" if best_method == method else f"({comp:.1f}\\%)"
      item_str = acc
      if indicate_best:
        item_str = f"{acc} {comp}"
      s.append(item_str)
    strs.append(s)
  return strs


def str_table_multiple(dic, T=0):
  methods = list(dic.keys()) # row names
  groups = list(dic[methods[0]].keys()) # 1st column name
  mulcols = len(dic[methods[0]][groups[0]].keys())
  cs = "c" * (mulcols - 1)
  latex_header = "\\multicolumn{" + str(mulcols) + "}{" + cs + "|}"
  strs = [["Generator"] + [latex_header + "{" + \
    formal_generator_name(g) + "}" for g in groups]]
  s = ["Dataset"]
  for g in groups:
    Gs = list(dic[methods[0]][g].keys()) # 2nd column name
    s.extend(formal_generator_name(Gs))
  strs.append(s)
  for method in methods:
    s = [method]
    for group in groups:
      for G in dic[method][group].keys():
        dic_ = {m : dic[m][group][G] if G in dic[m][group] else 0
                  for m in methods}
        best_ind, best_method, best_val = max_key(dic_)
        acc = f"{dic[method][group][G] * 100:.1f}\\%"
        comp = (dic[method][group][G] - best_val) / best_val * 100
        if best_method == method:
          item_str = "\\textbf{" + acc + "}"
        else:
          item_str = f"{acc} ({comp:.1f}\\%)"
        s.append(item_str)
    strs.append(s)
  return strs


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/semantics/", help="")
  parser.add_argument("--name", default="LSE_arch_compare-trunc-teval")
  parser.add_argument("--force-calc", default=0, type=int)
  args = parser.parse_args()
  params = get_table_suit(args.name)
  dic = {}
  for method in params[0]:
    dic[method] = {}
    if type(params[1]) is dict:
      for group, Gs in params[1].items():
        dic[method][group] = {}
        for G in Gs:
          for args_name in get_args_name(*params[2:]): 
            fpath = f"{args.dir}/{G}_{method}_{args_name}_evaluation.pth"
            mIoU = iou_from_pth(fpath)
            if mIoU < 0:
              continue
            show_name = G.split("_")[1]
            dic[method][group][show_name] = float(mIoU)
    else:
      for G in params[1]:
        # for 2-D dictionary, this is a single-loop
        for args_name in get_args_name(*params[2:]): 
          fpath = f"{args.dir}/{G}_{method}_{args_name}.txt"
          if not os.path.exists(fpath):
            print(f"!> {fpath} not found")
            continue
          mIoU, c_ious = read_results(fpath)
          dic[method][G] = float(mIoU)
          print(method, args_name, G, mIoU)

  if "all" in args.name or "other" in args.name:
    strs = str_table_multiple(dic)
  else:
    strs = str_table_single(dic)
  with open(f"results/tex/{args.name}.tex", "w") as f:
    f.write(str_latex_table(strs))

import torch, sys, os, argparse
sys.path.insert(0, ".")
from utils.misc import listkey_convert, str_latex_table
from utils.op import torch2numpy
import numpy as np


def get_table_suit(name):
  # default: full
  methods = ["LSE", "LSE-W", "LSE-F", "LSE-WF", "NSE-1", "NSE-2"]
  if "concise" in name:
    methods = ["LSE", "NSE-1", "NSE-2"]
  ls = ""
  if "notrunc" in name:
    ls = "_lsnotrunc-mixwp"
  if "-nteval" in name:
    ls += "_elsnotrunc-mixwp"
  elif "-teval" in name:
    ls += "_elstrunc-wp"  
  
  # default
  total_linears = [1]
  layer_detachs = [0]
  loss_types = ["focal"]
  optims = ["adam-0.001"]

  if "LSE_arch_compare" in name:
    Gs = ["stylegan2_ffhq", "stylegan2_church"]
    total_linears = [0, 1]
    layer_detachs = [0, 1]
    loss_types = ["focal", "normal"]
  elif "face" in name:
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
  return methods, Gs, optims, total_linears, layer_detachs, loss_types, ls


def formal_name(name):
  if type(name) is list:
    return [formal_name(n) for n in name]
  finds = ["stylegan", "pggan", "bedroom", "church", "celebahq", "ffhq"]
  subs = ["StyleGAN", "PGGAN", "Bedroom", "Church", "CelebAHQ", "FFHQ"]
  for find, sub in zip(finds, subs):
    name = name.replace(find, sub)
  return name


def get_args_name(optims=["adam-0.001"], total_linears=[1], layer_detachs=[0], loss_types=["focal"], ls=""):
  for optim in optims:
    for total_linear in total_linears:
      for layer_detach in layer_detachs:
        for loss_type in loss_types:
          yield f"{optim}_t{total_linear}_d{layer_detach}_l{loss_type}{ls}"


def aggregate_iou(res):
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
  return mIoU


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
  strs = [formal_name(Gs)]
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
  strs = [["Generator"] + [latex_header + "{" + formal_name(g) + "}" for g in groups]]
  s = ["Dataset"]
  for g in groups:
    Gs = list(dic[methods[0]][g].keys()) # 2nd column name
    s.extend(formal_name(Gs))
  strs.append(s)
  for method in methods:
    s = [method]
    for group in groups:
      for G in dic[method][group].keys():
        #print(method, group, dic[method][group].keys())
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


def iou_from_pth(fpath):
  if "NSE-2" in fpath:
    fpath = fpath.replace("adam-0.01", "adam-0.001")
  if not os.path.exists(fpath):
    print(f"!> {fpath} not found")
    return -1

  res_path = fpath.replace(".pth", ".txt")
  if os.path.exists(res_path):
    with open(res_path, "r") as f:
      res = f.readline().strip().split(" ")
    return float(res[0])
  
  # pixelacc, IoU, AP, AR, mIoU, mAP, mAR, sups
  res = torch.load(fpath, map_location='cpu')
  mIoU = aggregate_iou(res)
  with open(res_path, "w") as f:
    f.write(str(float(mIoU)))
  return mIoU


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/semantics/", help="")
  parser.add_argument("--name", default="other-concise-notrunc-teval")
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
          fpath = f"{args.dir}/{G}_{method}_{args_name}_evaluation.pth"
          mIoU = iou_from_pth(fpath)
          if mIoU < 0:
            continue
          dic[method][G] = float(mIoU)
          print(method, G, mIoU)

  if "all" in args.name or "other" in args.name:
    strs = str_table_multiple(dic)
  else:
    strs = str_table_single(dic)
  with open(f"results/tex/{args.name}.tex", "w") as f:
    f.write(str_latex_table(strs))

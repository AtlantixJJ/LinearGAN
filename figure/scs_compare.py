import torch, sys, os, argparse
sys.path.insert(0, ".")
from utils.misc import listkey_convert, str_latex_table
from utils.op import torch2numpy
import numpy as np
from figure.methods_compare import iou_from_pth, formal_name


def get_table_suit():
  params = []
  for G in ["stylegan2_bedroom"]:#["stylegan2_church", "stylegan2_bedroom", "stylegan2_ffhq"]:
    for n in [1, 8]:
      for r in range(5):
        params.append((G, n, r))
  return params # G, n, r


def str_table_single(dic, indicate_best=True, T=0):
  Gs = list(dic.keys())
  strs = [[""] + formal_name(Gs)]
  for n in dic[Gs[0]].keys():
    s = [f"{n}"]
    for G in Gs:
      mini, mean, maxi = dic[G][n]
      delta = max(mean - mini, maxi - mean)
      item_str = f"{mean*100:.1f}({delta*100:.1f})"
      s.append(item_str)
    strs.append(s)
  return strs

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/scs/", help="")
  parser.add_argument("--force-calc", default=0, type=int)
  args = parser.parse_args()
  params = get_table_suit()
  dic = {}
  for G, n, r in params:
    if G not in dic:
      dic[G] = {}
    if n not in dic[G]:
      dic[G][n] = []
    fpath = f"{args.dir}/{G}_r{r}_n{n}_repeat{r}.pth"
    mIoU = iou_from_pth(fpath)
    if mIoU > 0:
      dic[G][n].append(mIoU)


  vdic = {}
  for G in dic.keys():
    vdic[G] = {}
    for n in dic[G].keys():
      v = np.array(dic[G][n])
      vdic[G][n] = [v.min(), v.mean(), v.max()]
    # baseline
    fpath = f"{args.dir}/{G}_{G}_baseline_repeat0.pth"
    vdic[G]["baseline"] = [0,iou_from_pth(fpath),0]
  strs = str_table_single(vdic)
  with open(f"results/tex/scs.tex", "w") as f:
    f.write(str_latex_table(strs))

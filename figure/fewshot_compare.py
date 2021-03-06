import torch, sys, os, argparse, glob
sys.path.insert(0, ".")
import numpy as np

from lib.misc import listkey_convert, str_latex_table
from lib.op import torch2numpy
from evaluate import read_results

def invert_dic(dic):
  idic = {}
  for k1 in dic.keys():
    for k2 in dic[k1].keys():
      if k2 not in idic:
        idic[k2] = {}
      idic[k2][k1] = dic[k1][k2]
  return idic


def str_table_single_multicol(dic, indicate_best=True):
  row_names = list(dic.keys())
  col_names = list(dic[row_names[0]].keys())
  show_names = ["\\multicolumn{2}{c|}{" + c + "}"
    for c in col_names]
  strs = [show_names]
  for row_name in row_names:
    s = [row_name]
    for col_name in col_names:
      midv, delta, full = dic[row_name][col_name]
      midv *= 100
      delta *= 100
      item_str = f"{midv:.1f} $\\pm$ {delta:.1f}"
      s.append(item_str)

      p = midv / full * 100
      dp = delta / full * 100
      item_str = f"({p:.1f} $\\pm$ {dp:.1f})"
      s.append(item_str)
    strs.append(s)
  return strs


def str_table_single(dic, indicate_best=True):
  row_names = list(dic.keys())
  col_names = list(dic[row_names[0]].keys())
  strs = [col_names]
  for row_name in row_names:
    s = [row_name]
    for col_name in col_names:
      midv, delta, full = dic[row_name][col_name]
      midv *= 100
      delta *= 100
      p = midv / full * 100
      item_str = f"{midv:.1f} ({p:.1f}) $\\pm$ {delta:.1f}"
      s.append(item_str)
    strs.append(s)
  return strs

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/fewshot", help="")
  parser.add_argument("--force-calc", default=0, type=int)
  args = parser.parse_args()
  full_res = {
    "stylegan2_ffhq" : 79.7,
    "stylegan2_bedroom" : 53.9,
    "stylegan2_church" : 37.7}
  res = {
    "stylegan2_ffhq" : {},
    "stylegan2_bedroom" : {},
    "stylegan2_church" : {}}
  files = glob.glob(f"{args.dir}/*.txt")
  for G_name in res.keys():
    dic = {1:[], 4:[], 8:[], 16:[]}
    for fpath in files:
      if G_name not in fpath:
        continue
      mIoU, cious = read_results(fpath)
      if mIoU < 0:
        continue
      n, r = fpath.split("_")[2:4]
      n, r = int(n[1:]), int(r[1:])
      dic[n].append(mIoU)

    for n, v in dic.items():
      minv, midv, maxv = np.min(v), np.mean(v), np.max(v)
      delta = max(abs(minv - midv), abs(midv - maxv))
      res[G_name][str(n)] = (midv, delta, full_res[G_name])
    
  strs = str_table_single(invert_dic(res))
  with open(f"results/tex/fewshot.tex", "w") as f:
    f.write(str_latex_table(strs))
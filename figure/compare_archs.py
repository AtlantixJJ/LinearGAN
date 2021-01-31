import torch, sys, os, argparse
sys.path.insert(0, ".")
import numpy as np

from lib.misc import str_latex_table, formal_generator_name, get_args_name
from lib.op import torch2numpy
from evaluate import read_results


def enumerate_names(prev=[], i=0, groups=[]):
  res = []
  for key in groups[i]:
    if len(groups[i]) > 1:
      cur = prev + [key]
    else:
      cur = prev

    if i < len(groups) - 1:
      t = enumerate_names(cur, i + 1, groups)
      res.extend(t)
    else:
      res.append(cur)
  return res


def enumerate_args(prev=[], i=0, groups=[]):
  res = []
  for key in groups[i]:
    if i < len(groups) - 1:
      t = enumerate_args(prev + [key], i + 1, groups)
      res.extend(t)
    else:
      res.append(prev + [key])
  return res


def LSE_table():
  Gs = ["SB", "S2B"]
  methods = ["LSE"]
  loss_types = ["N"] # "F"
  lrs = ["0.001"] #["0.01", "0.001"]
  lw_types = ["SP", "None"]
  ls = ["Trunc"]#["Tmixed", "Ttrunc"]
  els = ["Etrunc"]#["Emixed", "Etrunc"]
  row_groups = [Gs, methods, loss_types, ls]
  col_groups = [lw_types, lrs, els]
  row_names = enumerate_names(groups=row_groups)
  col_names = enumerate_names(groups=col_groups)

  Gs = ["stylegan_bedroom", "stylegan2_bedroom"]
  methods = ["LSE"]
  loss_types = ["lnormal"] # "lfocal"
  lrs = ["lr0.001"] #["0.01", "0.001"]
  lw_types = ["lwsoftplus", "lwnone"]
  ls = ["lstrunc-wp"] #["lsnotrunc-mixwp", "lstrunc-wp"]
  els = ["elstrunc-wp"] #["elsnotrunc-mixwp", "elstrunc-wp"]
  row_groups = [Gs, methods, loss_types, ls]
  col_groups = [lw_types, lrs, els]
  row_args = enumerate_args(groups=row_groups)
  col_args = enumerate_args(groups=col_groups)
  for row_name, row_arg in zip(row_names, row_args):
    for col_name, col_arg in zip(col_names, col_args):
      row = "-".join(row_name)
      col = "-".join(col_name)
      arg = "_".join(row_arg + col_arg)
      yield row, col, arg #row_arg, col_arg


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


def str_table_single(dic):
  strs = []
  for row_name in dic.keys():
    if len(strs) == 0: # table header
      strs.append([] + list(dic[row_name].keys()))
    s = [row_name]
    for col_name in dic[row_name].keys():
      s.append(f"{dic[row_name][col_name]*100:.2f}")
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


def get_table(args):
  dic = {}
  count = 0
  for row_name, col_name, arg in LSE_table():
    if row_name not in dic:
      dic[row_name] = {}
    fpath = f"{args.dir}/{arg}.txt"
    if not os.path.exists(fpath):
      print(f"=> {fpath} not found")
      dic[row_name][col_name] = -1
    else:
      mIoU, cIoUs = read_results(fpath)
      dic[row_name][col_name] = mIoU
      count += 1
  
  strs = str_table_single(dic)
  with open(f"results/tex/{args.name}.tex", "w") as f:
    f.write(str_latex_table(strs))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/semantics/", help="")
  parser.add_argument("--name", default="LSE_arch_compare")
  parser.add_argument("--force-calc", default=0, type=int)
  args = parser.parse_args()

  get_table(args)

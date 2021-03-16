"""
Select categories and output the latex table.
"""
import torch, sys, os, argparse
sys.path.insert(0, ".")
import numpy as np
from collections import OrderedDict

from lib.misc import *
from lib.op import torch2numpy
from evaluate import read_results


def get_table_suit(G_name, ds):
  ds_name = "Bedroom" if ds == "bedroom" else "Church"
  FGs = [formal_name(G_name) + "-" + ds_name]
  methods = ["LSE", "NSE-1", "NSE-2"]
  loss_types = ["N"]
  lrs = ["0.001"]
  lw_types = ["SP"]
  ls = ["Trunc"]#["Tmixed", "Ttrunc"]
  els = ["Etrunc"]#["Emixed", "Etrunc"]
  row_groups = [FGs, methods, loss_types, ls]
  col_groups = [lw_types, lrs, els]
  row_names = enumerate_names(groups=row_groups)
  col_names = enumerate_names(groups=col_groups)

  Gs = [G_name + "_" + ds]
  loss_types = ["lnormal"]
  lrs = ["lr0.001"]
  lw_types = ["lwsoftplus"]#, "lwnone"]
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


def get_class_table(data_dir, G_name, ds):
  dic = OrderedDict()
  for row_name, col_name, arg in get_table_suit(G_name, ds):
    if row_name not in dic:
      dic[row_name] = OrderedDict()
    fpath = f"{data_dir}/{arg}.txt"
    if not os.path.exists(fpath):
      print(f"=> {fpath} not found")
      dic[row_name][col_name] = -1
    else:
      mIoU, cIoUs = read_results(fpath)
      clabels = []
      cious = []
      for i in range(len(cIoUs)):
        if float(cIoUs[i]) > 0.1:
          cious.append(float(cIoUs[i]))
          clabels.append(labels[i])
      dic[row_name][col_name] = cIoUs
  return dic


def get_common_labels(dic):
  common_labels = set()
  for k1 in dic.keys():
    for k2 in dic[k1].keys():
      cious = dic[k1][k2]
      for i in range(len(cious)):
        if cious[i] > 0.1:
          common_labels.add(labels[i])
  common_labels = list(common_labels)
  common_label_indice = [labels.index(n) for n in common_labels]
  common_label_indice.sort()
  return common_label_indice


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/full_label_SE/", help="")
  args = parser.parse_args()
  labels = read_ade20k_labels()[1:]
  G_labels = OrderedDict()
  gdic = OrderedDict()
  # read classwise data
  for G_name in ["pggan", "stylegan", "stylegan2"]:
    G_labels[G_name] = OrderedDict()
    gdic[G_name] = OrderedDict()
    for ds in ["bedroom", "church"]:
      gdic[G_name][ds] = OrderedDict()
      dic = get_class_table(args.dir, G_name, ds)
      label_indice = get_common_labels(dic)
      selected_labels = np.array(labels)[label_indice]
      G_labels[G_name][ds] = selected_labels
      ndic = OrderedDict()
      for k1 in dic.keys():
        for k2 in dic[k1].keys():
          ndic[f"{k1}_{k2}"] = OrderedDict()
          cious = np.array(dic[k1][k2])[label_indice]
          for n, v in zip(selected_labels, cious):
            ndic[f"{k1}_{k2}"][n] = v
          gdic[G_name][ds][k1] = cious.mean()

      strs = str_table_single(ndic)
      with open(f"results/tex/catselect_class_{G_name}_{ds}.tex", "w") as f:
        f.write(str_latex_table(strs))

  strs = str_table_multiple(gdic)
  with open(f"results/tex/catselect_global.tex", "w") as f:
    f.write(str_latex_table(strs))

  with open("figure/selected_labels.csv", "w") as f:
    for G in ["stylegan2", "stylegan", "pggan"]:
      for ds in ["bedroom", "church"]:
        s = ",".join([G + "_" + ds] + list(G_labels[G][ds]))
        f.write(s + "\n")
"""Compare different architectures of LSE, NSE-1, and NSE-2.
"""
import torch, sys, os, argparse
sys.path.insert(0, ".")
import numpy as np

from lib.misc import *
from lib.op import torch2numpy
from evaluate import read_results


def all_methods():
  FGs = [
    "CelebAHQ", "CelebAHQ", "FFHQ",
    "Bedroom", "Bedroom", "Bedroom",
    "Church", "Church", "Church"]
  methods = ["LSE", "NSE-1", "NSE-2"]
  loss_types = ["F"]
  lrs = ["0.001"]
  lw_types = ["SP"]
  ls = ["Trunc"]#["Tmixed", "Ttrunc"]
  els = ["Etrunc"]#["Emixed", "Etrunc"]
  row_groups = [FGs, methods, loss_types, ls]
  col_groups = [lw_types, lrs, els]
  row_names = enumerate_names(groups=row_groups)
  col_names = enumerate_names(groups=col_groups)

  Gs = [
    "pggan_celebahq", "stylegan_celebahq", "stylegan2_ffhq",
    "pggan_bedroom", "stylegan_bedroom", "stylegan2_bedroom",
    "pggan_church", "stylegan_church", "stylegan2_church"]
  loss_types = ["lfocal"]
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


def LSE_table():
  Gs = ["SB", "S2B"]
  methods = ["NSE-1"]
  loss_types = ["N", "F"]
  lrs = ["0.001"] #["0.01", "0.001"]
  lw_types = ["SP", "None"]
  ls = ["Trunc"]#["Tmixed", "Ttrunc"]
  els = ["Etrunc"]#["Emixed", "Etrunc"]
  row_groups = [Gs, methods, loss_types, ls]
  col_groups = [lw_types, lrs, els]
  row_names = enumerate_names(groups=row_groups)
  col_names = enumerate_names(groups=col_groups)

  Gs = ["stylegan_bedroom", "stylegan2_bedroom"]
  loss_types = ["lnormal", "lfocal"]
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


def get_table(args):
  dic = {}
  for row_name, col_name, arg in all_methods():
    if row_name not in dic:
      dic[row_name] = {}
    fpath = f"{args.dir}/{arg}.txt"
    if not os.path.exists(fpath):
      print(f"=> {fpath} not found")
      dic[row_name][col_name] = -1
    else:
      mIoU, cIoUs = read_results(fpath)
      dic[row_name][col_name] = mIoU
  
  strs = str_table_single(dic)
  with open(f"results/tex/{args.name}.tex", "w") as f:
    f.write(str_latex_table(strs))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/semantics/", help="")
  parser.add_argument("--name", default="all")
  parser.add_argument("--force-calc", default=0, type=int)
  args = parser.parse_args()

  get_table(args)

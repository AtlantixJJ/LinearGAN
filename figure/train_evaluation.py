"""Visualize the training evolution of mIoU."""
import torch, sys, os, argparse, glob
sys.path.insert(0, ".")
from lib.misc import formal_name
from lib.visualizer import plot_dict
import numpy as np


def get_SE_names(name):
  Gs = ["pggan", "stylegan", "stylegan2"]
  Ds = [name] #["face", "bedroom", "church"]
  SEs = ["LSE", "NSE-1", "NSE-2"]
  for G in Gs:
    for D in Ds:
      for SE in SEs:
        if D == "face":
          D = "ffhq" if G == "stylegan2" else "celebahq"
        yield f"{G}_{D}_{SE}"

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="expr/semantics", help="")
  parser.add_argument("--name", default="", help="")
  args = parser.parse_args()

  if args.name == "":
    for name in ["face", "bedroom", "church"]:
      os.system(f"python figure/train_evaluation.py --dir {args.dir} --name {name}")
    exit(0)

  dic = {}
  for SE_name in get_SE_names(args.name):
    G, D, SE = SE_name.split("_")
    G_name, D_name = formal_name(G), formal_name(D)
    expr_dir = glob.glob(f"{args.dir}/{SE_name}*")
    if len(expr_dir) == 0:
      print(f"!> {SE_name} not exist!")
      continue
    expr_dir = expr_dir[0]
    res = torch.load(f"{expr_dir}/train_evaluation.pth")
    if len(res) < 50:
      print(f"!> {SE_name} data point number : {len(res)} < 50")

    mIoU = [r[0] for r in res]
    key = f"{G_name}-{D_name}"
    if key not in dic:
      dic[key] = {}
    dic[key][SE] = mIoU
  
  plot_dict(dic, f"results/train_eval/train_eval_{args.name}.pdf", 1, 3)


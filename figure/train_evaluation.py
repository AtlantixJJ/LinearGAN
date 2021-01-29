import torch, sys, os, argparse, glob
sys.path.insert(0, ".")
from utils.misc import listkey_convert, str_latex_table, read_selected_labels
from utils.op import torch2numpy
import numpy as np
from predictors.face_segmenter import CELEBA_CATEGORY
import matplotlib.pyplot as plt

def get_SE_names():
  Gs = ["pggan", "stylegan", "stylegan2"]
  Ds = ["face", "bedroom", "church"]
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
  args = parser.parse_args()

  for SE_name in get_SE_names():
    expr_dir = glob.glob(f"{args.dir}/{SE_name}*")
    if len(expr_dir) == 0:
      print(f"!> {SE_name} not exist!")
      continue
    expr_dir = expr_dir[0]
    res = torch.load(f"{expr_dir}/train_evaluation.pth")
    if len(res) < 50:
      print(f"!> {SE_name} data point number : {len(res)} < 50")

    pixelacc = [r[0] for r in res]
    mIoU = [r[0] for r in res]
    ax = plt.subplot(1, 2, 1)
    ax.plot(pixelacc)
    ax = plt.subplot(1, 2, 2)
    ax.plot(mIoU)
    plt.savefig(f"results/train_eval/{SE_name}.png")
    plt.close()


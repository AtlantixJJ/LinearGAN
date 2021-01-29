import torch, sys, os, argparse, glob
sys.path.insert(0, ".")
from utils.misc import listkey_convert, str_latex_table
from utils.op import torch2numpy
import numpy as np


def invert_dic(dic):
  idic = {}
  for k1 in dic.keys():
    for k2 in dic[k1].keys():
      if k2 not in idic:
        idic[k2] = {}
      idic[k2][k1] = dic[k1][k2]
  return idic


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


def iou_from_pth(fpath):
  if "NSE-2" in fpath:
    fpath = fpath.replace("adam-0.01", "adam-0.001")
  if not os.path.exists(fpath):
    #fpath = fpath.replace('lsnotrunc-mixwp', 'lsnotrunc-mixwp_elstrunc-wp')
    #if not os.path.exists(fpath):
    print(f"=> skip {fpath}")
    return -1

  res_path = fpath.replace(".pth", ".txt")
  if os.path.exists(res_path) and args.force_calc == 0:
    with open(res_path, "r") as f:
      val = float(f.readline().strip())
    return val
  
  # IoU, AP, AR, mIoU, mAP, mAR, sups
  res = torch.load(fpath, map_location='cpu')
  mIoU = aggregate_iou(res)
  with open(res_path, "w") as f:
    f.write(str(float(mIoU)))
  return mIoU


def str_table_single(dic, indicate_best=True):
  row_names = list(dic.keys())
  col_names = list(dic[row_names[0]].keys())
  strs = [col_names]
  for row_name in row_names:
    s = [row_name]
    for col_name in col_names:
      midv, delta = dic[row_name][col_name]
      item_str = f"{midv*100:.1f}\\% $\\pm$ {delta*100:.1f}\\%"
      s.append(item_str)
    strs.append(s)
  return strs


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="expr/fewshot", help="")
  parser.add_argument("--force-calc", default=0, type=int)
  args = parser.parse_args()
  
  res = {
    "stylegan2_ffhq" : {},
    "stylegan2_bedroom" : {},
    "stylegan2_church" : {}}
  dirs = glob.glob(f"{args.dir}/*")
  for d in dirs:
    files = glob.glob(f"{d}/*evaluation.pth")
    files.sort()
    G_name = listkey_convert(d, list(res.keys()))
    dic = {1:[], 4:[], 8:[], 16:[]}
    for fpath in files:
      print(fpath)
      mIoU = iou_from_pth(fpath)
      if mIoU < 0:
        continue
      name = fpath[fpath.rfind("r"):fpath.rfind("_")]
      r, n = name.split("_")
      r = int(r[1:])
      n = int(n[1:])
      dic[n].append(mIoU)
    print(dic)
    for n, v in dic.items():
      minv, midv, maxv = np.min(v), np.mean(v), np.max(v)
      delta = max(abs(minv - midv), abs(midv - maxv))
      res[G_name][str(n)] = (midv, delta)
  
  strs = str_table_single(invert_dic(res))
  with open(f"results/tex/fewshot.tex", "w") as f:
    f.write(str_latex_table(strs))
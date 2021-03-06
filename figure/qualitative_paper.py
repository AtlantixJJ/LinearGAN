"""Create qualitative figures used in the paper."""
import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch, cv2
import torch.nn.functional as F
from torchvision import utils as vutils

from lib.visualizer import segviz_numpy
from lib.misc import formal_name, listkey_convert
from lib.op import torch2image, torch2numpy, bu
from predictors.face_segmenter import FaceSegmenter, CELEBA_CATEGORY
from predictors.scene_segmenter import SceneSegmenter
from models.helper import build_generator, load_semantic_extractor


def get_classes(l, start=0):
  x = np.array(l)
  y = x.argsort()
  k = 0
  while x[y[k]] < 1e-3:
    k += 1
  y = y[k:][::-1]
  # all classes are the same
  names = label_list[y - 1 + start] 
  return x[y], names.tolist(), y + start

def get_text(gt, ct):
  s = []
  vals, names, cats = get_classes(ct["IoU"])
  iou = gt["mIoU"]
  s.append([f"mIoU {iou:.3f}", high_contrast[0]])
  s.extend([[f"{name} {val:.3f}", high_contrast[cat]]
    for cat, name, val in zip(cats, names, vals)])
  return s[:7]

def process(res):
  res = torch.cat(res)
  res = F.interpolate(res, 256,
    mode="bilinear", align_corners=True)
  return utils.torch2numpy(res * 255).transpose(0, 2, 3, 1)


def get_result_G(G_name, z):
  G = Gs[G_name]
  P = Ps[G_name]
  if hasattr(G, "truncation"):
    wp = G.truncation(G.mapping(z))
    image, feature = G.synthesis(wp, generate_feature=True)
  else:
    image, feature = G(z, generate_feature=True)
  label = P(image, size=256)
  label_viz = segviz_numpy(label.cpu())
  image_set = [torch2image(bu(image, 256))[0], label_viz]
  text_set = [
    formal_name(G_name).split("_")[0],
    "UNet" if is_face else "DeeplabV3"]
  for i, (SE_name, SE) in enumerate(SE_models[G_name].items()):    
    seg = SE(feature, size=label.shape[2])[-1]
    est_label = seg.argmax(1)
    est_label_viz = segviz_numpy(est_label[0].cpu())
    image_set.append(est_label_viz)
    text_set.append(SE_name)
  return image_set, text_set


def SE_from_dir(data_dir, G_names):
  model_dirs = glob.glob(f"{data_dir}/*")
  model_files = [d for d in model_dirs if os.path.isdir(d)]
  model_files = [glob.glob(f"{f}/*.pth") for f in model_files]
  model_files = [[m for m in ms if "eval" not in m] \
    for ms in model_files]
  model_files = [m[0] for m in model_files if len(m) == 1]
  model_files.sort()
  SE_model_paths = {
    G_name : [p for p in model_files if G_name in p]
    for G_name in G_names.split(",")}
  SE_models = {G_name : {} for G_name in G_names.split(",")}
  for G_name in SE_model_paths:
    model_paths = SE_model_paths[G_name]
    for fpath in model_paths:
      SE_name = listkey_convert(fpath, ["LSE", "NSE-1", "NSE-2"])
      SE_models[G_name][SE_name] = load_semantic_extractor(fpath).cuda()
  return SE_models

def put_text(img, text, pos):
  N_text = len(text)
  textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 1)[0]
  pos = (pos[0] - int(textsize[0] // 2),
         pos[1] + int(textsize[1] // 2))
  cv2.putText(img, text, pos,
    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0),
    1, cv2.LINE_AA)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="expr/semantics")
  parser.add_argument("--op", default="", help="face,bedroom,church")
  parser.add_argument("--place", default="paper", help="paper | appendix")
  parser.add_argument("--viz-models", default="LSE,NSE-1,NSE-2")
  parser.add_argument("--repeat", default=1, type=int)
  parser.add_argument("--scale", default=1.0, type=float)
  parser.add_argument("--row-set-num", default=1, type=int)
  parser.add_argument("--gpu-id", default=0, type=int)
  args = parser.parse_args()

  if args.op == "":
    args = f"--dir {args.dir} --place {args.place} --viz-models {args.viz_models} --repeat {args.repeat} --row-set-num {args.row_set_num} --gpu-id {args.gpu_id}"
    for op in ["face", "bedroom", "church"]:
      os.system(f"python figure/qualitative_paper.py {args} --op {op}")
    exit(0)
    
  G_names = "pggan_celebahq,stylegan_celebahq,stylegan2_ffhq"
  if args.op == "bedroom":
    G_names = "pggan_bedroom,stylegan_bedroom,stylegan2_bedroom"
  elif args.op == "church":
    G_names = "pggan_church,stylegan_church,stylegan2_church"
  # setup and constants
  data_dir = args.dir
  Gs = {G_name : build_generator(G_name).net for G_name in G_names.split(",")}
  is_face = "ffhq" in G_names or "celebahq" in G_names
  net = FaceSegmenter if is_face else SceneSegmenter
  Ps = {G_name : net(model_name=G_name) for G_name in G_names.split(",")}
  label_list = CELEBA_CATEGORY if is_face else []
  n_class = len(label_list)
  N_repeat = args.repeat
  SE_models = SE_from_dir(data_dir, G_names)
  torch.manual_seed(1701 if args.place == 'paper' else 3941)
  z = torch.randn(N_repeat, 512)

  # get result from all models
  paper_img = []
  paper_text = []
  count = 0
  with torch.no_grad():
    for G_name, G in Gs.items():
      for i in range(N_repeat):
        image_set, text_set = get_result_G(G_name, z[i:i+1].cuda())
        paper_img.append(image_set)
        paper_text.append(text_set)
        
  segnet_name = "UNet" if is_face else "DeeplabV3"
  imageset_headers = ["image", segnet_name] + list(SE_models.keys())

  row_set_num = args.row_set_num # place 2 set of images in every row
  image_set_num = len(paper_img[0])
  N_col = row_set_num * image_set_num
  N_imgs = image_set_num * N_repeat * len(Gs)
  N_row = N_imgs // N_col
  imsize = 256
  text_height = 33
  pad = 5
  CW = imsize + pad
  CH = imsize + text_height
  canvas_width = imsize * N_col + pad * (N_col - 1)
  canvas_height = (text_height + imsize) * N_row
  canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
  canvas.fill(255)

  for idx, (image_set, text_set) in enumerate(zip(paper_img, paper_text)):
    col = (idx % row_set_num) * image_set_num
    row = idx // row_set_num
    for img, text in zip(image_set, text_set):
      # labeling
      if (row == 0 and "GAN" not in text) or col == 0:
        put_text(canvas, text, (CW * col + imsize // 2, CH * row + text_height // 2))

      stx = text_height + CH * row
      edx = stx + imsize
      sty = CW * col
      edy = sty + imsize
      #print(canvas.shape, row, col, stx, edx, sty, edy)
      canvas[stx:edx, sty:edy] = img
      col += 1
      """ # put miou annotations on both sides
      if idx % 3 == 0:
          idx = idx // 3
          delta = CW * (N_col + 1) if idx % 2 == 1 else 0
          for i, (text, rgb) in enumerate(paper_text[idx]):
              i += 1
              cv2.putText(canvas, text,
                  (5 + delta, text_height + (idx // 2) * CH + 33 * i),
                  cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                  2 if "mIoU" in text else 1, cv2.LINE_AA)
      """

  sizes = (N_col, N_row * 1.05)
  sizes = (sizes[0] * args.scale, sizes[1] * args.scale)
  fig = plt.figure(figsize=sizes) # paper: 11, 7
  plt.imshow(canvas)
  plt.axis("off")
  plt.tight_layout()
  plt.savefig(f"results/qualitative/{args.op}_{args.place}_{args.repeat}.pdf")
  plt.close()

"""Visualize layer-wise plot."""
import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy, math
import numpy as np
import matplotlib.pyplot as plt
import torch, cv2
import torch.nn.functional as F
from torchvision import utils as vutils
from lib.visualizer import segviz_numpy, high_contrast_arr, make_grid_numpy
from lib.op import torch2image, torch2numpy, bu
from predictors.face_segmenter import FaceSegmenter, CELEBA_CATEGORY
from predictors.scene_segmenter import SceneSegmenter
from models.helper import build_generator, load_semantic_extractor

def formal_name(name):
  if type(name) is list:
    return [formal_name(n) for n in name]
  finds = ["stylegan", "pggan", "bedroom", "church", "celebahq", "ffhq", "_"]
  subs = ["StyleGAN", "PGGAN", "", "", "", "", ""]
  for find, sub in zip(finds, subs):
    name = name.replace(find, sub)
  return name

def put_text(img, text, pos):
  N_text = len(text)
  textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 1)[0]
  pos = (
    pos[0] - int(textsize[0] // 2),
    pos[1] + int(textsize[1] // 2))
  cv2.putText(img, text, pos,
          cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0),
          1, cv2.LINE_AA)
          
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
  image_set = []
  text_set = []
  SE = viz_models[G_name]
  segs = SE(feature, size=label.shape[2])
  for i, seg in enumerate(segs):
    est_label = bu(seg, 256).argmax(1)
    est_label_viz = segviz_numpy(est_label[0].cpu())
    image_set.append(est_label_viz)
    text_set.append(f"{i}")
  text_set[-1] = "LSE"
  text_set.append(formal_name(G_name))
  is_face = "ffhq" in G_name or "celebahq" in G_name
  text_set.append("UNet" if is_face else "DeeplabV3")
  image_set.append(torch2image(bu(image, 256))[0])
  image_set.append(label_viz)
  return image_set, text_set



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-dir", default="expr/semantics_layerwise")
  parser.add_argument("--out-dir", default="results/layerplot/layer_loss")
  parser.add_argument("--place", default="paper", help="paper | appendix")
  parser.add_argument("--viz-model", default="LSE")
  parser.add_argument("--repeat", default=2, type=int)
  parser.add_argument("--gpu-id", default=0, type=int)
  parser.add_argument("--show-weight", default=0, type=int)
  args = parser.parse_args()

  G_names = "pggan_celebahq,stylegan_celebahq,stylegan2_ffhq,pggan_church,stylegan_church,stylegan2_church,pggan_bedroom,stylegan_bedroom,stylegan2_bedroom"
  # setup and constants
  data_dir = args.model_dir
  Gs = {G_name : build_generator(G_name).net for G_name in G_names.split(",")}
  is_face = "ffhq" in G_names or "celebahq" in G_names
  unet = FaceSegmenter()

  def P_from_name(G_name):
    if "ffhq" in G_name or "celebahq" in G_name:
      return unet
    return SceneSegmenter(model_name=G_name)

  Ps = {G_name : P_from_name(G_name) for G_name in G_names.split(",")}
  label_list = CELEBA_CATEGORY if is_face else []
  n_class = len(label_list)
  N_repeat = args.repeat
  model_dirs = glob.glob(f"{data_dir}/*")
  model_files = [d for d in model_dirs if os.path.isdir(d)]
  model_files = [glob.glob(f"{f}/*.pth") for f in model_files]
  model_files = [[m for m in ms if "eval" not in m] \
    for ms in model_files]
  model_files = [m[0] for m in model_files if len(m) == 1]
  model_files.sort()
  viz_model_paths = {
    G_name : [p for p in model_files if G_name in p]
    for G_name in G_names.split(",")}
  viz_models = {G_name : {} for G_name in G_names.split(",")}
  for G_name in viz_model_paths:
    model_paths = viz_model_paths[G_name]
    fpath = model_paths[0]
    viz_models[G_name] = load_semantic_extractor(fpath).cuda()

  torch.manual_seed(7279 if args.place == 'paper' else 7979)
  z = torch.randn(N_repeat, 512)

  # get result from all models
  res = {}
  count = 0
  with torch.no_grad():
    for G_name, G in Gs.items():
      paper_img = []
      paper_text = []
      for i in range(N_repeat):
        image_set, text_set = get_result_G(G_name, z[i:i+1].cuda())
        paper_img.append(image_set)
        paper_text.append(text_set)
      res[G_name] = (paper_img, paper_text)

  imsize = 256
  text_height = 33
  line_height = 10
  pad = 5

  for G_name, image_text in res.items():
    image_set_num = len(image_text[0][0])
    N_col = image_set_num
    if N_col > 10:
      N_col = math.ceil(N_col / 2)
    N_imgs = image_set_num * N_repeat
    N_row = math.ceil(N_imgs / N_col)
    CW = imsize + pad
    CH = imsize + text_height
    canvas_width = imsize * N_col + pad * (N_col - 1)
    canvas_height = (text_height + imsize) * N_row
    if args.show_weight:
      canvas_height += line_height + 2 * pad

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
    canvas.fill(255)
    if args.show_weight:
      x = torch.nn.functional.softplus(viz_models[G_name].layer_weight)
      x = x / x.max() # this is max rather than sum

    col = row = 0
    for idx, (image_set, text_set) in enumerate(zip(*image_text)):
      for i, (img, text) in enumerate(zip(image_set, text_set)):
        stx = text_height + CH * row
        if args.show_weight:
          stx += line_height + 2 * pad
        edx = stx + imsize
        sty = CW * col
        edy = sty + imsize

        put_text(canvas, text,
          (CW * col + imsize // 2,
          CH * row + text_height // 2))

        if args.show_weight and row == 0 and i < x.shape[0]:
          # weight labeling
          length = int(x[i] * imsize * 0.9)
          p = int(imsize * 0.05)
          a = stx - line_height - pad
          canvas[a : stx - pad, sty + p: sty + p + length] = \
            (200, 50, 60)

        #print(canvas.shape, row, col, stx, edx, sty, edy)
        canvas[stx:edx, sty:edy] = img
        col += 1
        if col == N_col:
          col = 0
          row += 1

    sizes = (N_col, N_row * 1.2)
    fig = plt.figure(figsize=sizes) # paper: 11, 7
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/{G_name}_{args.place}_{args.repeat}_{args.show_weight}.pdf")
    plt.close()


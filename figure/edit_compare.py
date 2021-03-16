import sys, argparse, glob, os
sys.path.insert(0, ".")
import torch
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np

from lib.visualizer import segviz_torch
from lib.misc import imread, listkey_convert, set_cuda_devices
from lib.op import generate_images, bu
from models.helper import build_generator, load_semantic_extractor
from predictors.face_segmenter import FaceSegmenter
from predictors.scene_segmenter import SceneSegmenter
from manipulation.spie import ImageEditing

G2SE = {
  "stylegan2_ffhq" : "predictors/pretrain/stylegan2_ffhq_LSE.pth"}

def norm(x):
  return (x.clamp(-1, 1) + 1) / 2


def read_data(data_dir, n_class=15):
  #kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
  files = glob.glob(f"{data_dir}/*")
  files.sort()
  N = 7
  z = []
  image_stroke = []
  label_stroke = []
  image_mask = []
  label_mask = []
  for i, f in enumerate(files):
    if i % N == 0:
      z.append(np.load(f)[0])
    elif i % N == 1:
      img = imread(f).transpose(2, 0, 1)
      image_stroke.append((img - 127.5) / 127.5)
    elif i % N == 2:
      label_img = imread(f)
      t = np.zeros(label_img.shape[:2]).astype("uint8")
      for j in range(n_class):
        c = get_label_color(j)
        t[color_mask(label_img, c)] = j
      label_stroke.append(t)
    elif i % N == 3:
      img = imread(f) #img = cv2.erode(imread(f), kernel)
      image_mask.append((img[:, :, 0] > 127).astype("uint8"))
    elif i % N == 4:
      img = imread(f) #img = cv2.erode(imread(f), kernel)
      label_mask.append((img[:, :, 0] > 127).astype("uint8"))
  res = [z, image_stroke, image_mask, label_stroke, label_mask]
  return [torch.from_numpy(np.stack(r)) for r in res]


def eval_single(G, P, SE, eval_file, origin_zs, image_strokes, image_masks, label_strokes, label_masks):
  print(f"=> Loading from {eval_file}")
  z, wp = torch.load(eval_file, map_location='cpu')

  origin_image, tar_image, tar_label, origin_int_label, origin_ext_label = fuse_stroke(G, SE, P, origin_zs, image_strokes, image_masks, label_strokes, label_masks)

  with torch.no_grad():
    edited_image, edited_feature = G.synthesis(
      wp.cuda(), generate_feature=True)
    edited_ext_label = P(edited_image, size=256)[0]
    seg = SE(edited_feature, last_only=True, size=256)
    edited_int_label = seg[0][0].argmax(1)
    del edited_feature
  tar_label_viz, oel_viz, oil_viz, eel_viz, eil_viz = [
    torch.stack([segviz_torch(l) for l in label_set]) for label_set in [tar_label, origin_ext_label, origin_int_label, edited_ext_label, edited_int_label]]
  tar_image = norm(tar_image)
  origin_image = norm(origin_image)
  edited_image = norm(edited_image)
  return origin_image, edited_image, tar_image, tar_label_viz, oel_viz, oil_viz, eel_viz, eil_viz


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--eval-file', type=str, default='results/edit',
    help='The path to the experiment result file.')
  parser.add_argument('--data-dir', type=str, default='data/collect_data_ffhq',
    help='The output directory.')
  parser.add_argument('--out-dir', type=str, default='results/scs',
    help='The output directory.')
  parser.add_argument('--gpu-id', default='0',
    help='Which GPU(s) to use. (default: `0`)')
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)
  res = read_data(args.data_dir)
  origin_zs, image_strokes, image_masks, label_strokes, label_masks = res

  G_name = "stylegan2_ffhq"
  G = build_generator(G_name).net
  if "ffhq" in G_name:
    P = FaceSegmenter()
  else:
    P = SceneSegmenter(model_name=G_name)
  SE = load_from_pth(G2SE[G_name])
  SE.cuda().eval()

  if ".pth" in args.eval_file:
    origin_image, edited_image, tar_label_viz, oel_viz, oil_viz, eel_viz, eil_viz = eval_single(G, P, SE, args.eval_file, *res)
    all_images = torch.cat(bu([origin_image.cpu(),
      oil_viz.cpu(), tar_label_viz.cpu(), eil_viz.cpu(), edited_image.cpu()], 256))
    disp_image = vutils.make_grid(all_images,
      nrow=origin_image.shape[0], padding=10, pad_value=1)
    fpath = args.eval_file.replace(".pth", ".png")
    vutils.save_image(disp_image.unsqueeze(0), fpath)
  elif "fewshot" in args.eval_file:
    ffhq_path = "expr/fewshot/stylegan2_ffhq_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r3_n{num_sample}LSE_15_512,512,512,512,512,512,512,512,512,256,256,128,128,64,64,32,32_0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16.pth"
    tar_label_vizs = []
    eimgs = []
    eil_vizs = []
    for num_sample in [1, 4, 8, 16]:
      eval_file = f"results/edit/fewshot{num_sample}-stylegan2_ffhq_internal_100.pth"
      SE = load_from_pth_LSE(ffhq_path.format(num_sample=num_sample))
      SE.cuda().eval()
      oimg, eimg, _, tar_label_viz, _, oil_viz, _, eil_viz = eval_single(
        G, P, SE, eval_file, *res)
      tar_label_vizs.append(tar_label_viz)
      eimgs.append(eimg)
      eil_vizs.append(eil_viz)
    all_images = [oimg.cpu()]
    for i in range(len(eimgs)):
      all_images.append(tar_label_vizs[i].cpu())
      all_images.append(eil_vizs[i].cpu())
      all_images.append(eimgs[i].cpu())
    all_images = torch.stack(bu(all_images, 256), 1)
    disp_image = vutils.make_grid(all_images.view(-1, 3, 256, 256),
      nrow=all_images.shape[1], padding=10, pad_value=1)
    fpath = "results/edit/fewshot_stylegan2_ffhq_edit.png"
    vutils.save_image(disp_image.unsqueeze(0), fpath)
  elif "full" in args.eval_file:
    eval_files = [
      "results/edit/stylegan2_ffhq_internal_100.pth",
      "results/edit/stylegan2_ffhq_external_100.pth",
      "results/edit/stylegan2_ffhq_image_20.pth",
      "results/edit/fewshot8-stylegan2_ffhq_internal_100.pth"]
    SEfewshot = load_from_pth_LSE("expr/fewshot/stylegan2_ffhq_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r0_n1LSE_15_512,512,512,512,512,512,512,512,512,256,256,128,128,64,64,32,32_0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16.pth")
    SEfewshot.cuda().eval()
    oimg, eimg1, _, tar_label_viz, _, oil_viz, _, eil_viz = eval_single(
      G, P, SE, eval_files[0], *res)
    _, eimg2, _, _, oel_viz, _, eel_viz, _ = eval_single(
      G, P, SE, eval_files[1], *res)
    _, eimg3, tar_image, _, _, _, _, _ = eval_single(
      G, P, SE, eval_files[2], *res)
    _, eimg4, _, _, _, _, _, eil_viz2 = eval_single(
      G, P, SEfewshot, eval_files[3], *res)
    PAD_SIZE = 10
    make_col = lambda x : vutils.make_grid(bu(x.cpu(), 256),
      nrow=1, padding=PAD_SIZE, pad_value=1)[:, :, :-PAD_SIZE].unsqueeze(0)
    comb_col = lambda x : vutils.make_grid(torch.cat(x),
      nrow=len(x), padding=0, pad_value=1).unsqueeze(0)
    # original image
    oimg = make_col(oimg)
    # color stroke editing
    tar_image, eimg3 = make_col(tar_image), make_col(eimg3)
    color_stroke = comb_col([tar_image, eimg3])
    # segmentation & semantic stroke
    oel_viz, tar_label_viz =  make_col(oel_viz), make_col(tar_label_viz)
    seg_stroke = comb_col([oel_viz, tar_label_viz])
    # SPIE(UNet)
    eimg2, eel_viz = make_col(eimg2), make_col(eel_viz)
    spie_unet = comb_col([eimg2, eel_viz])
    # SPIE(fewshot LSE)
    eimg4, eil_viz2 = make_col(eimg4), make_col(eil_viz2)
    spie_fewshot = comb_col([eimg4, eil_viz2])
    # SPIE(full LSE)
    eimg1, eil_viz = make_col(eimg1), make_col(eil_viz)
    spie_full = comb_col([eimg1, eil_viz])
    imgs = [oimg, color_stroke, seg_stroke, spie_unet, spie_full, spie_fewshot]
    res = []
    for i in range(len(imgs)):
      imgs[i] = imgs[i][:, :, PAD_SIZE:-PAD_SIZE] \
        if imgs[i].shape[2] > imgs[0].shape[2] else imgs[i]
      res.append(imgs[i])
      res.append(torch.ones(1, 3, imgs[i].shape[2], PAD_SIZE))
    disp_image = torch.cat(res[:-1], 3)
    """
    all_images = torch.stack(bu([oimg.cpu(), 
      tar_image.cpu(), eimg3.cpu(),
      oel_viz.cpu(), tar_label_viz.cpu(), 
      eimg2.cpu(), eel_viz.cpu(),
      eimg4.cpu(), eil_viz2.cpu(),
      eimg1.cpu(), eil_viz.cpu()], 256), 1)
    disp_image = vutils.make_grid(all_images.view(-1, 3, 256, 256),
      nrow=all_images.shape[1], padding=10, pad_value=1)
    """
    fpath = "results/edit/stylegan2_ffhq_edit_full.png"
    vutils.save_image(disp_image, fpath)
  else:
    eval_files = [
      args.eval_file + "/stylegan2_ffhq_internal_50.pth",
      args.eval_file + "/stylegan2_ffhq_external_50.pth",
      args.eval_file + "/stylegan2_ffhq_image_50.pth"]
      #"/fewshot1-stylegan2_ffhq_internal_50.pth"]
    oimg, eimg1, _, tar_label_viz, _, oil_viz, _, eil_viz = eval_single(
      G, P, SE, eval_files[0], *res)
    _, eimg2, _, _, oel_viz, _, eel_viz, _ = eval_single(
      G, P, SE, eval_files[1], *res)
    _, eimg3, tar_image, _, _, _, _, _ = eval_single(
      G, P, SE, eval_files[2], *res)
    all_images = torch.stack(bu([oimg.cpu(), oel_viz.cpu(),
      tar_label_viz.cpu(), 
      tar_image.cpu(), eimg3.cpu(),
      eimg2.cpu(), eel_viz.cpu(),
      eimg1.cpu(), eil_viz.cpu()], 256), 1)
    disp_image = vutils.make_grid(all_images.view(-1, 3, 256, 256),
      nrow=all_images.shape[1], padding=10, pad_value=1)
    fpath = args.eval_file + "/stylegan2_ffhq_edit.png"
    vutils.save_image(disp_image.unsqueeze(0), fpath)
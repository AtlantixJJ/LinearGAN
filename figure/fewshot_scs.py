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
from evaluate import evaluate_predictions, aggregate_iou, write_results
from manipulation.scs import read_labels

LSE_format = "expr/semantics/{G_name}_LSE_lnormal_lstrunc-wp_lwsoftplus_lr0.001/{G_name}_LSE.pth"
fewshot_LSE_format = "expr/fewshot/{G_name}_LSE_fewshot/r{r}_n{n}.pth"


def eval_single(Gs, Ps, eval_file):
  G_name = listkey_convert(eval_file,
    ["stylegan2_ffhq", "stylegan2_bedroom", "stylegan2_church"])

  G = Gs[G_name]
  P = Ps[G_name]

  target_labels = read_labels(G_name, G, P)
  size = target_labels.shape[2]

  print(f"=> Loading from {eval_file}")
  z, wp = torch.load(eval_file, map_location='cpu')
  print(z.shape, wp.shape, target_labels.shape)
  N, M = z.shape[0] // 10, 10 # 10 repeats
  N_show = 4

  res_file = eval_file.replace(".pth", ".txt")
  is_gen = True #args.generate_image == "1" 
  is_eval = not os.path.exists(res_file)

  if is_gen or is_eval:
    images = []
    sample_labels = []
    for i in tqdm(range(wp.shape[0])):
      if not is_eval and i >= N_show * M:
        break
      with torch.no_grad():
        image = G.synthesis(wp[i:i+1].cuda())
        sample_labels.append(P(image, size=size).cpu())
        if i < N_show * M:
          images.append((bu(image, size).cpu() + 1) / 2)
    images = torch.cat(images)
    sample_labels = torch.cat(sample_labels)
    sample_labels = sample_labels.view(
      -1, M, *sample_labels.shape[1:])
    target_label_viz = bu(torch.stack([
      segviz_torch(x) for x in target_labels[:N_show]]), size)
    if is_gen:
      show_labels = bu(
        target_label_viz[:N_show].cpu(), 256).unsqueeze(1)
      show_images = bu(images, 256).view(-1, M, *images.shape[1:]).cpu()
      print(show_labels.shape, show_images.shape)
      all_images = torch.cat([show_labels, show_images], 1)
      disp_image = vutils.make_grid(all_images.view(
        -1, *all_images.shape[2:]),
        nrow=M+1, padding=10, pad_value=1)
      fpath = eval_file.replace(".pth", ".pdf")
      vutils.save_image(disp_image.unsqueeze(0), fpath)

    if is_eval:
      mIoU, c_ious = aggregate_iou(evaluate_predictions(
        target_labels, sample_labels))
      write_results(res_file, mIoU, c_ious)
  else:
    mIoU, c_iou = read_results(res_file)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--eval-file', type=str, default='results/scs',
    help='The path to the experiment result file.')
  parser.add_argument('--out-dir', type=str, default='results/scs',
    help='The output directory.')
  parser.add_argument('--gpu-id', default='0',
    help='Which GPU(s) to use. (default: `0`)')
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)
  Gs = {
      "stylegan2_ffhq" : build_generator("stylegan2_ffhq").net,
      "stylegan2_bedroom" : build_generator("stylegan2_bedroom").net,
      "stylegan2_church" :  build_generator("stylegan2_church").net
    }
  Ps = {
      "stylegan2_ffhq" : FaceSegmenter(),
      "stylegan2_bedroom" : SceneSegmenter(model_name="stylegan2_bedroom"),
      "stylegan2_church" : SceneSegmenter(model_name="stylegan2_church")
    }

  if ".pth" in args.eval_file:
    eval_single(Gs, Ps, args.eval_file)
  else:
    eval_files = glob.glob(f"{args.eval_file}/*.pth")
    eval_files.sort()
    for eval_file in eval_files:
      eval_single(Gs, Ps, eval_file)
  

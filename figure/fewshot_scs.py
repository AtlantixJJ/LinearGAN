import sys, argparse, glob, os
sys.path.insert(0, ".")
import torch
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
from dataset import NoiseDataModule
from models.semantic_extractor import load_from_pth_LSE, save_to_pth, SELearner
from utils.visualizer import HtmlPageVisualizer, segviz_torch
from utils.misc import imread, listkey_convert, set_cuda_devices
from utils.op import generate_images, bu
from utils.editor import ConditionalSampling
from models.helper import build_generator, build_semantic_extractor
from predictors.face_segmenter import FaceSegmenter
from predictors.scene_segmenter import SceneSegmenter
from pytorch_lightning.metrics.functional import iou, precision_recall


SE_bedroom_path = "notruncwp_expr/methods_compare/stylegan2_bedroom_LSE_adam-0.01_t1_d0_lfocal_lsnotrunc-mixwp/LSE_16_512,512,512,512,512,512,512,512,512,256,256,128,128_0,1,2,3,4,5,6,7,8,9,10,11,12.pth"
SE_church_path = "notruncwp_expr/methods_compare/stylegan2_church_LSE_adam-0.01_t1_d0_lfocal_lsnotrunc-mixwp/LSE_14_512,512,512,512,512,512,512,512,512,256,256,128,128_0,1,2,3,4,5,6,7,8,9,10,11,12.pth"
SE_ffhq_path = "notruncwp_expr/methods_compare/stylegan2_ffhq_LSE_adam-0.01_t1_d0_lfocal_lsnotrunc-mixwp/LSE_15_512,512,512,512,512,512,512,512,512,256,256,128,128,64,64,32,32_0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16.pth"
fewshot_SE_bedroom_path = "expr/fewshot/stylegan2_bedroom_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r0_n{num_sample}LSE_16_512,512,512,512,512,512,512,512,512,256,256,128,128_0,1,2,3,4,5,6,7,8,9,10,11,12.pth"
fewshot_SE_church_path = "expr/fewshot/stylegan2_church_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r0_n{num_sample}LSE_14_512,512,512,512,512,512,512,512,512,256,256,128,128_0,1,2,3,4,5,6,7,8,9,10,11,12.pth"
fewshot_SE_ffhq_path = "expr/fewshot/stylegan2_ffhq_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r0_n{num_sample}LSE_15_512,512,512,512,512,512,512,512,512,256,256,128,128,64,64,32,32_0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16.pth"


def read_labels(G_name, G, P):
  if "ffhq" in G_name:
    #image_ids = [0, 1, 2, 1004, 1006]
    image_ids = np.random.RandomState(1116).choice(list(range(2000)), (100,))
    target_labels = np.stack([imread(f"../datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15/{i}.png")
      for i in image_ids])[:, :, :, 0] # (N, H, W, 3)
    target_labels = torch.from_numpy(target_labels).long()
  else:
    P = SceneSegmenter(model_name=G_name)
    wp = np.load(f"data/trunc_{G_name}/wp.npy")
    wp = torch.from_numpy(wp).float().cuda()
    with torch.no_grad():
      images = generate_images(G, wp)
      target_labels = torch.cat([P(img.unsqueeze(0).cuda())[0] for img in images], 0)
  return target_labels


def calc_metric(target_labels, sample_labels):
  N, M = sample_labels.shape[:2]
  n_class = sample_labels.max() + 1
  pixelacc = torch.zeros((N, M))
  IoU = torch.zeros((N, M, n_class - 1))
  for i in range(N):
    gt = target_labels[i].cuda()
    for j in range(M):
      dt = sample_labels[i, j].cuda()
      IoU[i, j] = iou(dt, gt,
        num_classes=n_class,
        ignore_index=0, # This will cause background to be ignored
        absent_score=-1, # resulting in n_class - 1 vectors
        reduction='none').cpu()
      pixelacc[i, j] = (dt == gt).sum() / float(dt.shape.numel())
  pixelacc = pixelacc.mean()
  IoU = IoU.view(-1, n_class - 1).permute(1, 0).clone()
  c_IoU = torch.zeros((IoU.shape[0],)) # n_class - 1
  for i in range(IoU.shape[0]):
    v = IoU[i][IoU[i] > 0]
    c_IoU[i] = -1 if len(v) == 0 else v.mean()
  mIoU = c_IoU[c_IoU > -1].mean()
  return pixelacc, mIoU, c_IoU

def eval_single(Gs, Ps, eval_file):
  G_name = listkey_convert(eval_file,
    ["stylegan2_ffhq", "stylegan2_bedroom", "stylegan2_church"])

  G = Gs[G_name]
  P = Ps[G_name]
  if "r0_n" in eval_file:
    ind = eval_file.find("r0_n")
    num_sample = eval_file[ind+4:-4]

  target_labels = read_labels(G_name, G, P)
  size = target_labels.shape[2]

  print(f"=> Loading from {eval_file}")
  z, wp = torch.load(eval_file, map_location='cpu')
  print(z.shape, wp.shape)
  #target_labels = target_labels[:2]
  N, M = z.shape[0] // 10, 10 # 10 repeats
  N_show = 4

  res_file = eval_file.replace(".pth", ".txt")
  is_gen = True #args.generate_image == "1" 
  is_eval = not os.path.exists(res_file)

  if is_gen or is_eval:
    images = []
    sample_labels = []
    for i in tqdm(range(wp.shape[0])):
      with torch.no_grad():
        image = G.synthesis(wp[i:i+1].cuda())
        image = bu(image, size).clamp(-1, 1)
        sample_labels.append(P(image, size=size)[0].cpu())
        images.append((image.cpu() + 1) / 2)
    images = torch.cat(images)
    images = images.view(N, M, *images.shape[1:])
    sample_labels = torch.cat(sample_labels)
    sample_labels = sample_labels.view(N, M, *sample_labels.shape[1:])
    target_label_viz = torch.stack([segviz_torch(x) for x in target_labels])
    target_label_viz = bu(target_label_viz, size) # (5, C, H, W)
    if is_gen:
      all_images = torch.cat([
          target_label_viz[:N_show].unsqueeze(1).cpu(),
          images[:N_show].cpu()], 1)
      disp_image = vutils.make_grid(all_images.view(
        -1, *all_images.shape[2:]),
        nrow=M+1, padding=10, pad_value=1)
      fpath = eval_file.replace(".pth", ".png")
      vutils.save_image(disp_image.unsqueeze(0), fpath)

    if is_eval:
      pixelacc, mIoU, c_iou = calc_metric(target_labels, sample_labels)
      c_iou = [float(i) for i in c_iou]
      s = [str(c) for c in c_iou]
      mIoU = float(mIoU)
      pixelacc = float(pixelacc)
      with open(res_file, "w") as f:
        f.write(f"{mIoU} {pixelacc}\n")
        f.write(" ".join(s))
  else:
    with open(res_file, "r") as f:
      mIoU, pixelacc = f.readline().strip().split(" ")
      c_iou = [float(i) for i in f.readline().strip().split(" ")]


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
  
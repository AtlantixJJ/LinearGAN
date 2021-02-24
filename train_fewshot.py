import sys, argparse
sys.path.insert(0, ".")
import torch, os
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
from torch.utils.data.dataloader import DataLoader


def get_features(synthesis, wp, P=None, is_large_mem=False):
  features = []
  labels = []
  with torch.no_grad():
    for i in range(wp.shape[0]):
      image, feature = synthesis(wp[i:i+1], generate_feature=True)
      if P:
        labels.append(P(image, size=resolution).long())
      if is_large_mem:
        feature = [f.cpu() for f in feature]
      features.append(feature)
  features = [torch.cat([feats[i] for feats in features])
    for i in range(len(features[0]))]
  if P:
    labels = torch.cat(labels)
    return features, labels
  return features


class UpdateDataCallback(pl.Callback):
  def __init__(self, wp=None, is_large_mem=False):
    super().__init__()
    self.is_large_mem = is_large_mem
    self.wp = wp
  
  def on_epoch_end(self, trainer, pl_module):
    del pl_module.feature
    features = get_features(G.net.synthesis, self.wp,
      is_large_mem=self.is_large_mem)
    pl_module.feature = features


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--expr', type=str, default='expr/fewshot',
  help='The experiment directory')
  parser.add_argument('--gpu-id', type=str, default='0',
  help='GPUs to use. (default: %(default)s)')
  parser.add_argument('--num-sample', type=int, default=1,
  help='The total number of few shot samples.')
  parser.add_argument('--repeat-ind', type=int, default=0,
  help='The repeat index.')
  parser.add_argument('--G', type=str, default='stylegan2_ffhq',
  help='The model type of generator')
  args = parser.parse_args()

  is_large_mem = "ffhq" in args.G and args.num_sample >= 4
  is_face = "celebahq" in args.G or "ffhq" in args.G
  resolution = 512 if is_face else 256
  DIR = f"{args.expr}/{args.G}_LSE_fewshot"
  fpath = f"{DIR}_LSE_fewshot/r{args.repeat_ind}_n{args.num_sample}.pth"
  if os.path.exists(fpath):
    print(f"=> Skip {fpath}")
    exit(0)

  seed = [19961116, 1997816, 1116, 816, 19980903][args.repeat_ind]
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  
  # These import will affect cuda device setting
  from lib.misc import set_cuda_devices
  set_cuda_devices(args.gpu_id)
  from lib.dataset import NoiseDataModule
  from models.semantic_extractor import *
  from models.helper import build_semantic_extractor, save_semantic_extractor
  from models.stylegan2_generator import StyleGAN2Generator
  from lib.callback import SEVisualizerCallback
  from evaluate import evaluate_SE, write_results, aggregate_iou
  from predictors.face_segmenter import FaceSegmenter
  from predictors.scene_segmenter import SceneSegmenter

  G = StyleGAN2Generator(model_name=args.G, randomize_noise=True) 
  P = FaceSegmenter() if is_face else SceneSegmenter(model_name=args.G)
  P.eval().cuda()
  print(f"=> Segmenter has {P.num_categories} classes")
  # get the dims and layers
  features = G(G.easy_sample(1))['feature']
  dims = [s.shape[1] for s in features]
  layers = list(range(len(dims)))
  logger = pl_logger.TensorBoardLogger(DIR)
  udc = UpdateDataCallback(is_large_mem=is_large_mem)
  callbacks = [udc,
    SEVisualizerCallback(torch.randn(6, 512).cuda(), interval=1000)]
  dm = NoiseDataModule(train_size=128, batch_size=1)

  SE = build_semantic_extractor(lw_type="none", model_name="LSE",
      n_class=P.num_categories, dims=dims, layers=layers)
  SE.cuda().train()
  learner = SEFewShotLearner(model=SE, G=G.net, optim="adam-0.001",
    loss_type="normal",
    resolution=resolution,
    latent_strategy="trunc-wp",
    save_dir=DIR)
  wp = learner.get_wp(torch.randn(args.num_sample, 512).cuda())
  udc.wp = wp
  features, labels = get_features(G.net.synthesis, wp, P, is_large_mem)
  learner.P = P
  trainer = pl.Trainer(
      logger=logger,
      profiler=True,
      accumulate_grad_batches=args.num_sample,
      max_epochs=50,
      progress_bar_refresh_rate=1,
      log_every_n_steps=1,
      callbacks=callbacks,
      gpus=1)
  learner.feature, learner.label = features, labels
  trainer.fit(learner, dm)

  fpath = f"{DIR}/r{args.repeat_ind}_n{args.num_sample}.pth"
  save_semantic_extractor(SE, fpath)
  num = 10000
  mIoU, c_ious = evaluate_SE(SE, G.net, P,
    resolution, num, "trunc-wp")
  name = f"{args.G}_n{args.num_sample}_r{args.repeat_ind}_elstrunc-wp"
  write_results(f"results/fewshot/{name}.txt", mIoU, c_ious)

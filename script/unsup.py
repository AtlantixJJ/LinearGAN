"""Training script for LSE, NSE-1, NSE-2.
"""
import sys, argparse
sys.path.insert(0, ".")
import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger

from models.helper import *
from lib.callback import *
from lib.dataset import NoiseDataModule
from models.semantic_extractor import SELearner
from evaluate import evaluate_SE, write_results, aggregate_iou


def main(args):
  from predictors.face_segmenter import FaceSegmenter
  from predictors.scene_segmenter import SceneSegmenter

  DIR = f"{args.expr}/{args.G}"
  G = build_generator(args.G)
  is_face = "celebahq" in args.G or "ffhq" in args.G
  if is_face:
    P = FaceSegmenter()
  else:
    P = SceneSegmenter(model_name=args.G)
  print(f"=> Segmenter has {P.num_categories} classes")

  if len(args.reload) > 1:
    SE = load_semantic_extractor(args.reload)
  else:
    features = G(G.easy_sample(1))['feature']
    dims = [s.shape[1] for s in features]
    layers = list(range(len(dims)))
    SE = build_semantic_extractor(
      lw_type="none",
      model_name="LSE",
      n_class=P.num_categories,
      dims=dims,
      layers=layers)
  SE.cuda().train()

  dm = NoiseDataModule(train_size=1024, batch_size=1)
  z = torch.randn(6, 512).cuda()
  resolution = 512 if is_face else 256
  callbacks = [
    SEVisualizerCallback(z, interval=5 * 1024),
    TrainingEvaluationCallback()]

  logger = pl_logger.TensorBoardLogger(DIR)
  learner = SELearner(model=SE, G=G.net, P=P,
    lr=args.lr,
    loss_type="normal",
    latent_strategy=args.latent_strategy,
    resolution=resolution,
    save_dir=DIR)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Architecture setting
  parser.add_argument('--latent-strategy', type=str, default='trunc-wp',
    choices=['notrunc-mixwp', 'trunc-wp', 'notrunc-wp'],
    help='notrunc-mixwp: mixed W+ without truncation. trunc-wp: W+ with truncation. notrunc-wp: W+ without truncation.')
  parser.add_argument('--G', type=str, default='stylegan2_ffhq',
    help='The model type of generator')
  # Training setting
  parser.add_argument('--reload', type=str, default='',
    help='The path to saved file of semantic extractor.')
  parser.add_argument('--lr', type=float, default=0.001,
    help='The learning rate.')
  parser.add_argument('--expr', type=str, default='expr/unsup',
    help='The experiment directory.')
  parser.add_argument('--slurm', type=bool, default=False,
    help='If the script is run on slurm.')
  parser.add_argument('--gpu-id', type=str, default='0',
    help='GPUs to use. (default: %(default)s)')
  # evaluation settings
  parser.add_argument('--eval', type=int, default=1,
    help="Whether to evaluate after training.")
  args = parser.parse_args()
  from lib.misc import set_cuda_devices
  set_cuda_devices(args.gpu_id)
  main(args)
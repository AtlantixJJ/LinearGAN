import sys, argparse
sys.path.insert(0, ".")
import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
from torch.utils.data.dataloader import DataLoader

from models.helper import *
from lib.callback import *
from lib.dataset import NoiseDataModule
from models.semantic_extractor import SELearner
from evaluate import evaluate_SE, write_results, aggregate_iou


def main(args):
  from predictors.face_segmenter import FaceSegmenter
  from predictors.scene_segmenter import SceneSegmenter

  DIR = f"{args.expr}/{args.G}_{args.SE}_l{args.loss_type}_ls{args.latent_strategy}_lw{args.layer_weight}_lr{args.lr}"
  G = build_generator(args.G)
  is_face = "celebahq" in args.G or "ffhq" in args.G
  if is_face:
    P = FaceSegmenter()
  else:
    if args.full_label:
      P = SceneSegmenter()
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
      lw_type=args.layer_weight,
      model_name=args.SE,
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

  if hasattr(SE, "layer_weight"):
    print("=> Layer weight")
    weight = SE._calc_layer_weight()
    s = " ".join([f"{w:.2f}" for w in weight])
    print(f"=> Layer weight: {s}")
    callbacks.append(WeightVisualizerCallback())

  logger = pl_logger.TensorBoardLogger(DIR)
  learner = SELearner(model=SE, G=G.net, P=P,
    lr=args.lr,
    loss_type=args.loss_type,
    latent_strategy=args.latent_strategy,
    resolution=resolution,
    save_dir=DIR)
  trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=False,
    accumulate_grad_batches={0:1, 2:4, 18:64},
    max_epochs=50,
    progress_bar_refresh_rate=0 if args.slurm else 1,
    callbacks=callbacks,
    gpus=1,
    distributed_backend='dp')
  trainer.fit(learner, dm)
  save_semantic_extractor(SE, f"{DIR}/{args.G}_{args.SE}.pth")

  res_dir = DIR.replace(args.expr, "results/semantics/")
  num = 10000
  mIoU, c_ious = evaluate_SE(SE, G.net, P,
    resolution, num, args.latent_strategy)
  write_results(f"{res_dir}_els{args.latent_strategy}.txt", mIoU, c_ious)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Architecture setting
  parser.add_argument('--layer-weight', type=str, default='softplus',
    choices=['softplus', 'sigmoid', 'none'],
    help='Different layer weight strategy.')
  parser.add_argument('--latent-strategy', type=str, default='trunc-wp',
    choices=['notrunc-mixwp', 'trunc-wp', 'notrunc-wp'],
    help='notrunc-mixwp: mixed W+ without truncation. trunc-wp: W+ with truncation. notrunc-wp: W+ without truncation.')
  parser.add_argument('--G', type=str, default='stylegan2_ffhq',
    help='The model type of generator')
  parser.add_argument('--SE', type=str, default='LSE',
    help='The model type of semantic extractor')
  parser.add_argument('--loss-type', type=str, default='normal',
    help='focal: use Focal loss. normal: use CE loss.')
  parser.add_argument('--full-label', type=int, default=0,
    help='Default: 0, use selected label. 1: use full label.')
  # Training setting
  parser.add_argument('--reload', type=str, default='',
    help='The path to saved file of semantic extractor.')
  parser.add_argument('--lr', type=float, default=0.001,
    help='The learning rate.')
  parser.add_argument('--expr', type=str, default='expr/semantics',
    help='The experiment directory.')
  parser.add_argument('--slurm', type=bool, default=False,
    help='If the script is run on slurm.')
  parser.add_argument('--gpu-id', type=str, default='0',
    help='GPUs to use. (default: %(default)s)')
  args = parser.parse_args()
  from lib.misc import set_cuda_devices
  set_cuda_devices(args.gpu_id)
  main(args)
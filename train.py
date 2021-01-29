import sys, argparse
sys.path.insert(0, ".")
import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
from torch.utils.data.dataloader import DataLoader

# These import will affect cuda device setting
from utils.misc import print
from lib.dataset import NoiseDataModule
from models.semantic_extractor import SELearner
from models.helper import *
from lib.callback import *
from predictors.face_segmenter import FaceSegmenter
from predictors.scene_segmenter import SceneSegmenter


def main(args):
  DIR = f"{args.expr}/{args.G}_{args.SE}_l{args.loss_type}_ls{args.latent_strategy}"
  G = build_generator(args.G)
  is_face = "celebahq" in args.G or "ffhq" in args.G
  P = FaceSegmenter() if is_face else SceneSegmenter(model_name=args.G)
  print(f"=> Segmenter has {P.num_categories} classes")

  if len(args.reload) > 1:
    SE = load_semantic_extractor(args.reload)
  else:
    features = G(G.easy_sample(1))['feature']
    dims = [s.shape[1] for s in features]
    layers = list(range(len(dims)))
    SE = build_semantic_extractor(
      model_name=args.SE,
      n_class=P.num_categories,
      dims=dims,
      layers=layers)
  SE.cuda().train()


  if hasattr(SE, "layer_weight")
    print("=> Layer weight")
    weight = SE._calc_layer_weight()
    for i in range(weight.shape[0]):
      print(f"=> Layer {j} weight: {weight[i, j]:3f}")


  dm = NoiseDataModule(train_size=1024, batch_size=1)
  z = torch.randn(6, 512).cuda()
  vc = SEVisualizerCallback(z, interval=100)
  callbacks = [vc, TrainingEvaluationCallback()]
  #if "LSE" in args.SE:
  #  callbacks.append(WeightVisualizerCallback())

  logger = pl_logger.TensorBoardLogger(DIR)
  learner = SELearner(model=SE, G=G.net, P=P, optim=args.optim,
    loss_type=args.loss_type,
    resolution=512 if is_face and not is_feature_norm else 256,
    latent_strategy=args.latent_strategy,
    save_dir=DIR)
  trainer = pl.Trainer(
    logger=logger,
    profiler=True, #pl.profiler.AdvancedProfiler(),
    accumulate_grad_batches=64 if args.reload else {0:1, 2:4, 18:64},
    #track_grad_norm=2,
    max_epochs=50,
    progress_bar_refresh_rate=0 if args.slurm else 1,
    callbacks=callbacks,
    gpus=1,
    distributed_backend='dp')
  trainer.fit(learner, dm)
  save_to_pth(SE, DIR)


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
  parser.add_argument('--loss-type', type=str, default='focal',
    help='focal: use Focal loss. normal: use CE loss.')
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
  from utils.misc import set_cuda_devices
  set_cuda_devices(args.gpu_id)
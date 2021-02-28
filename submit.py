import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--func', default="test", type=str)
parser.add_argument('--gpu', default='0/1/2/3/4/5/6/7')
args = parser.parse_args()


def sr_LSE_arch():
  cmds = []
  srcmd = "python train.py --G {G} --SE LSE --lr 0.001 --loss-type {loss_type} --layer-weight {layer_weight} --latent-strategy trunc-wp"
  Gs = ["stylegan2_bedroom", "stylegan_bedroom", "pggan_bedroom", "stylegan2_church", "stylegan_church", "pggan_church"]
  for G in Gs:
    for layer_weight in ["softplus", "none"]:
      for loss_type in ['focal', 'normal']:
        cmds.append(srcmd.format(G=G, layer_weight=layer_weight,
          loss_type=loss_type))
  return cmds


def sr_NSE_arch():
  cmds = []
  srcmd = "python train.py --G {G} --SE NSE-1 --lr 0.001 --loss-type {loss_type} --layer-weight {layer_weight} --latent-strategy trunc-wp"
  Gs = ["stylegan2_bedroom", "stylegan_bedroom", "pggan_bedroom", "stylegan2_church", "stylegan_church", "pggan_church"]
  for G in Gs:
    for layer_weight in ["softplus", "none"]:
      for loss_type in ['focal', 'normal']:
        cmds.append(srcmd.format(G=G, layer_weight=layer_weight,
          loss_type=loss_type))
  return cmds


def sr_all_method():
  cmds = []
  srcmd = "python train.py --G {G} --SE {SE}"
  Gs = "stylegan2_bedroom,stylegan_bedroom,pggan_bedroom,stylegan2_church,stylegan_church,pggan_church,pggan_celebahq,stylegan_celebahq,stylegan2_ffhq".split(",")
  for G in Gs:
    for SE in ["LSE", "NSE-1", "NSE-2"]:
      cmd = srcmd.format(G=G, SE=SE)
      cmds.append(cmd)
  return cmds


def sr_all_method_face():
  cmds = []
  srcmd = "python train.py --G {G} --SE {SE}"
  Gs = "stylegan2_ffhq,stylegan_celebahq,pggan_celebahq".split(",")
  for SE in ["LSE", "NSE-1", "NSE-2"]:
    for G in Gs:
      cmd = srcmd.format(G=G, SE=SE)
      cmds.append(cmd)
  return cmds


def sr_all_method_other():
  cmds = []
  srcmd = "python train.py --G {G} --SE {SE}"
  Gs = "stylegan2_bedroom,stylegan2_church,stylegan_bedroom,stylegan_church,pggan_bedroom,pggan_church".split(",")
  for SE in ["LSE", "NSE-1", "NSE-2"]:
    for G in Gs:
      cmd = srcmd.format(G=G, SE=SE)
      cmds.append(cmd)
  return cmds


def train_fewshot():
  cmds = []
  evalcmd = "python3 train_fewshot.py --G {G} --num-sample {num_sample} --repeat-ind {repeat_ind}"
  for repeat_ind in range(5):
    for G in ["stylegan2_ffhq"]:#["stylegan2_church", "stylegan2_bedroom"]:
      for num_sample in [8, 1, 4, 16]:
        cmds.append(evalcmd.format(G=G,
          num_sample=num_sample, repeat_ind=repeat_ind))
  return cmds


def scs():
  cmds = []
  Gs = ["stylegan2_ffhq", "stylegan2_bedroom", "stylegan2_church"]
  n_inits = [10, 100, 100]
  SE_format = "expr/fewshot/{G}_LSE_fewshot/r{rind}_n{num_sample}.pth"
  evalcmd = "python manipulation/scs.py --SE {SE} --n-init {n_init}"
  for rind in range(5):
    for num_sample in [1, 8, 4, 16]:
      for G, n_init in zip(Gs, n_inits):
        SE = SE_format.format(G=G, rind=rind, num_sample=num_sample)
        cmds.append(evalcmd.format(SE=SE, rind=rind, n_init=n_init))
        if num_sample == 1 and rind == 0:
          SE = f"{G}_baseline"
          cmds.append(evalcmd.format(SE=SE, n_init=n_init))

  return cmds


def qualitative_figures():
  evalcmd = "python figure/qualitative_paper.py --op {op} --place {place} --repeat {repeat} --row-set-num {row_set_num}"
  cmds = []
  for place in ["paper", "appendix"]:
    for op in ["face", "bedroom", "church"]:
      for row_set_num in [1, 2]:
        if place == "paper":
          repeat = row_set_num
        else:
          repeat = row_set_num * 5
        cmds.append(evalcmd.format(
          op=op, place=place,
          repeat=repeat, row_set_num=row_set_num))
  return cmds


def generator_semantics():
  ALL_GANs = "stylegan2_bedroom,stylegan2_church,stylegan_bedroom,stylegan_church,pggan_bedroom,pggan_church,pggan_celebahq,stylegan_celebahq,stylegan2_ffhq,stylegan2_car,stylegan_ffhq"
  cmds = []
  basecmd = "python script/semantics/generator_semantics.py --G {G}"
  for G in ALL_GANs.split(","):
    cmds.append(basecmd.format(G=G))
  return cmds


funcs = {
  "sr_LSE_arch" : sr_LSE_arch,
  "sr_NSE_arch" : sr_NSE_arch,
  "sr_all_method" : sr_all_method,
  "sr_all_method_face" : sr_all_method_face,
  "sr_all_method_other" : sr_all_method_other,
  "train_fewshot" : train_fewshot,
  "scs" : scs,
  "qf" : qualitative_figures,
  "gs" : generator_semantics
  }

print(args.gpu)
if args.gpu in ["bzhou", "chpc"]:
  for i, cmd in enumerate(funcs[args.func]()):
    logfile = f"logs/{args.func}_{i}.txt"
    srun = f"srun -p {args.gpu} -o {logfile} -J {args.func} -n1 --gres=gpu:1 -l {cmd}"
    print(srun)
    os.system(srun)
else:
  gpus = args.gpu.split('/')
  slots = [[] for _ in gpus]
  for i, cmd in enumerate(funcs[args.func]()):
    gpu = gpus[i % len(gpus)]
    slots[i % len(gpus)].append(f"{cmd} --gpu-id {gpu}")
  for s in slots:
    cmd = " && ".join(s) #+ " &"
    print(cmd)
    os.system(cmd)

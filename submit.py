import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--func', default="test", type=str)
parser.add_argument('--gpu', default='0/1/2/3/4/5/6/7')
args = parser.parse_args()


def sr_LSE_arch():
  cmds = []
  srcmd = "python train.py --G {G} --SE LSE --lr 0.001 --loss-type normal --layer-weight {layer_weight} --latent-strategy {latent_strategy}"
  for G in ["stylegan2_bedroom", "stylegan_bedroom", "stylegan_celebahq"]:
    for layer_weight in ["softplus", "none"]:
      for latent_strategy in ['notrunc-mixwp', 'trunc-wp']:
        cmds.append(srcmd.format(G=G, layer_weight=layer_weight,
          latent_strategy=latent_strategy))
  return cmds


def sr_NSE_arch():
  cmds = []
  srcmd = "python train/extract_semantics.py --G {G} --optim adam-0.01 --SE {SE}"
  for G in ["pggan_bedroom", "pggan_church", "stylegan2_church", "stylegan2_bedroom", "stylegan_church", "stylegan_bedroom"]:
    cmds.append(srcmd.format(G=G, SE="NSE-1"))
  return cmds


def sr_all_method():
  cmds = []
  srcmd = "python train.py --G {G} --SE {SE}"
  Gs = "stylegan2_bedroom,stylegan_bedroom,pggan_bedroom,stylegan2_church,stylegan_church,pggan_church,pggan_celebahq,stylegan_celebahq,stylegan2_ffhq".split(",")
  for G in Gs:
    for SE in ["LSE", "NSE-1", "NSE-2"]:
      cmd = srcmd.format(G=G, SE=SE)
      if "NSE-2" in SE:
        cmd = cmd + " --optim adam-0.001"
      cmds.append(cmd)
  return cmds


def sr_all_method_face():
  cmds = []
  srcmd = "python train/extract_semantics.py --G {G} --SE {SE}"
  Gs = "stylegan2_ffhq,stylegan_celebahq,pggan_celebahq".split(",")
  for SE in ["LSE", "NSE-1", "NSE-2"]:
    for G in Gs:
      cmd = srcmd.format(G=G, SE=SE)
      cmds.append(cmd)
  return cmds


def sr_all_method_other():
  cmds = []
  srcmd = "python train/extract_semantics.py --G {G} --SE {SE}"
  Gs = "stylegan2_bedroom,stylegan2_church,stylegan_bedroom,stylegan_church,pggan_bedroom,pggan_church".split(",")
  for SE in ["LSE", "NSE-1", "NSE-2"]:
    for G in Gs:
      cmd = srcmd.format(G=G, SE=SE)
      cmds.append(cmd)
  return cmds


def train_fewshot():
  cmds = []
  evalcmd = "python3 train/fewshot_lse.py --G {G} --num-sample {num_sample}"
  for G in ["stylegan2_ffhq"]: #["stylegan2_church", "stylegan2_bedroom"]:
    for num_sample in [1, 8, 4, 16]:
      cmds.append(evalcmd.format(G=G, num_sample=num_sample))
  return cmds


def scs():
  cmds = []
  SEs = ["expr/fewshot/{G}_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r{repeat_ind}_n{num_sample}LSE_15_512,512,512,512,512,512,512,512,512,256,256,128,128,64,64,32,32_0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16.pth", "expr/fewshot/{G}_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r{repeat_ind}_n{num_sample}LSE_13_512,512,512,512,512,512,512,512,512,256,256,128,128_0,1,2,3,4,5,6,7,8,9,10,11,12.pth", "expr/fewshot/{G}_LSE_fewshot_adam-0.01_lsnotrunc-mixwp/r{repeat_ind}_n{num_sample}LSE_9_512,512,512,512,512,512,512,512,512,256,256,128,128_0,1,2,3,4,5,6,7,8,9,10,11,12.pth"]
  Gs = ["stylegan2_ffhq", "stylegan2_bedroom", "stylegan2_church"]
  n_inits = [10, 100, 100]
  evalcmd = "python script/semantics/conditional_sampling.py --SE {SE} --n-init {n_init} --repeat-ind {repeat_ind}"
  #Gs = Gs[1:]; SEs = SEs[1:]; n_inits = n_inits[1:]
  for repeat_ind in range(5):
    for G, se, n_init in zip(Gs, SEs, n_inits):
      for num_sample in [1, 8]:#4, 8, 16]:
        SE = se.format(
          G=G,
          num_sample=num_sample,
          repeat_ind=repeat_ind)
        cmds.append(evalcmd.format(
          SE=SE,
          n_init=n_init,
          repeat_ind=repeat_ind))
        """
        if num_sample == 1 and repeat_ind == 0:
          cmds.append(evalcmd.format(
            SE=f"{G}_baseline",
            n_init=n_init,
            repeat_ind=repeat_ind))
        """

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
    cmd = " && ".join(s) + " &"
    print(cmd)
    os.system(cmd)

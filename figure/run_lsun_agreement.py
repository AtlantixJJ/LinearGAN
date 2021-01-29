import os
basecmd = "python figure/lsun_agreement.py --model %s --task %s"

for model in ["pggan", "stylegan", "stylegan2"]:
  for task in ["bedroom", "church", "dinningroom", "resturant"]:
    os.system(basecmd % (model, task))
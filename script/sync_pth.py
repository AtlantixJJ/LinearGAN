import sys, os, torch, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='expr',
  help='The path to the experiment result file.')
parser.add_argument('--op', type=str, default='makelist',
  help='makelist, fromjr, fromhk')
args = parser.parse_args()


files = []
dirs = glob.glob(f"{args.data_dir}/*")
dirs.sort()
for d in dirs:
  if not os.path.isdir(d):
    continue
  files.extend(glob.glob(d + "/*.pth"))
files.sort()


if args.op == "fromhk":
  scpcmd = "ssh hk 'cd jianjin/RealGAN; tar cvfz {files}' | tar xvf - -C {data_dir}"
elif args.op == "tohk":
  scpcmd = "tar cvfz - {files} | ssh hk \"cd jianjin/RealGAN/; tar xvfz -\""
elif args.op == "tojr":
  scpcmd = "tar cvfz - {files} | ssh jr \"cd data/RealGAN; tar xvfz -\""

os.system(scpcmd.format(files=" ".join(files), data_dir=args.data_dir))

"""Make pytorch saved files compatible with previous version.
"""
import sys, os, torch, glob
fpath = sys.argv[1]

if ".pth" in fpath:
  os.system(f"cp {fpath} {fpath}.new")
  torch.save(torch.load(fpath), fpath,
    _use_new_zipfile_serialization=False)
else:
  dirs = glob.glob(f"{fpath}/*")
  dirs.sort()
  for d in dirs:
    if not os.path.isdir(d):
      if ".pth" == d[-4:]:
        print(f"=> Processing: {d}")
        os.system(f"cp {d} {d}.new")
        torch.save(torch.load(d), d,
          _use_new_zipfile_serialization=False)
      continue
    fpaths = glob.glob(d + "/*.pth")
    for fpath in fpaths:
      if ".new" in fpath:
        continue
      print(f"=> Processing: {fpath}")
      os.system(f"cp {fpath} {fpath}.new")
      try:
        torch.save(torch.load(fpath), fpath,
          _use_new_zipfile_serialization=False)
      except:
        print("!> Failed!")

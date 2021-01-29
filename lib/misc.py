# python 3.7
"""Misc utility functions without dependence on Deep Learning library."""

import os
import numpy as np
from PIL import Image


def read_ade20k_labels(fpath="figure/ade20k_labels.csv"):
  """Read label file for ADE20K dataset
  
  Args:
    fpath : The store location of the label file.
  Returns:
    A label list. Note that the default label has one more background category comparing to the official label.
  """
  lines = open(fpath, "r").readlines()
  label_list = [l.split(",")[-1].split(";")[0].strip() for l in lines[1:]]
  return label_list


def read_selected_labels(fpath="figure/selected_labels.csv"):
  """Read the labels for models.
  
  Args:
    fpath : The path to label file. Stardard format is one line for each model, with the first column being the model name.

  Returns:
    A dict object specifying the mapping from model names to category names.
  """
  dic = {}
  with open(fpath, "r") as f:
    for line in f.readlines():
      items = line.split(",")
      dic[items[0]] = [i.strip() for i in items[1:]]
  return dic


def imread(fpath):
  """Read image and returns a numpy array in [0, 255] scale."""
  with open(os.path.join(fpath), "rb") as f:
    return np.asarray(Image.open(f), dtype="uint8")


def imwrite(fpath, image, format="RGB"):
  """Write an numpy image to file.

  Args:
    image : an array of shape [H, W, 3] and scale in [0, 255].
  """
  if ".jpg" in fpath or ".jpeg" in fpath:
    ext = "JPEG"
  elif ".png" in fpath:
    ext = "PNG"
  with open(os.path.join(fpath), "wb") as f:
    Image.fromarray(image.astype("uint8")).convert(format).save(f, format=ext)


def set_cuda_devices(device_ids, use_cuda=True):
  """Sets visible CUDA devices.

  Example:

  set_cuda_devices('0,1', True)  # Enable device 0 and 1.
  set_cuda_devices('3', True)  # Enable device 3 only.
  set_cuda_devices('all', True)  # Enable all devices.
  set_cuda_devices('-1', True)  # Disable all devices.
  set_cuda_devices('0', False)  # Disable all devices.

  Args:
    devices_ids: A string, indicating all visible devices. Separated with comma.
      To enable all devices, set this field as `all`.
    use_cuda: Whether to use cuda. If set as False, all devices will be
      disabled. (default: True)
  """
  if not use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return 0
  assert isinstance(device_ids, str)
  if device_ids.lower() == 'all':
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
      del os.environ['CUDA_VISIBLE_DEVICES']
    return 8
  os.environ['CUDA_VISIBLE_DEVICES'] = device_ids.replace(' ', '')
  return len(device_ids.split(","))


def listkey_convert(name, listkey, output=None):
  """Check which key in listkey is a substring of name and return a value.
  
  Args:
    name : The raw string. It may contain one or more keys from listkey.
    listkey : A list of keys.
    output : When output is None, the matched key will be returned directly. 
             When output is a list, the function will return the element of 
             output at index of the matched key.
  Returns:
    A matched key, or the output word corresponding to the index of the 
    matched key, or an empty string if matching fails
  """
  for i, key in enumerate(listkey):
    if key in name:
      if output is not None:
        return output[i]
      return key
  return ""


def print_table(t):
  """Print a table
  
  Args:
    t : A 2D numpy array or a 2D list.
  """
  for i in range(len(t)):
    s = ""
    for j in range(len(t[0])):
      try:
        s += f"{t[i, j]:.3f}\t"
      except:
        s += f"{t[i][j]:.3f}\t"
    print(s)


def str_num(n, F="%.3f"):
  """Formatting numerical values."""
  return (F % n).replace(".000", "")


def get_dic_depth(dic):
  """Get the depth of a dict."""
  v = next(iter(dic.values()))
  count = 1
  while type(v) is dict:
    count += 1
    v = next(iter(v.values()))
  return count


def trim_dic(dic):
  """Remove empty key-value pairs."""
  for k in list(dic.keys()):
    if type(dic[k]) is dict:
      if len(dic[k]) == 0:
        del dic[k]
      else:
        trim_dic(dic[k])


def dic2table(dic, transpose=True):
  """Convert dict of depth 2 to latex table.

  Args:
    dic : In the form of dic[row_key][col_key].
    transpose : When True, the row_key of dic corresponds to the 
                col_key of the output table.
  """
  strs = []
  col_names = list(next(iter(dic.values())).keys())
  ncols = len(col_names) + 1
  strs.append([""] + col_names)
  for row_name, row_vals in dic.items():
    strs.append([row_name] + [row_vals[k] for k in col_names])
  if transpose:
    nstrs = [[] for _ in range(len(strs[0]))]
    for j in range(len(strs[0])):
      for i in range(len(strs)):
        nstrs[j].append(strs[i][j])
    return nstrs
  return strs

  
def str_latex_table(strs):
  """Format a string table to a latex table.
  
  Args:
    strs : A 2D string table. Each item is a cell.
  Returns:
    A single string for the latex table.
  """
  for i in range(len(strs)):
    for j in range(len(strs[i])):
      if "_" in strs[i][j]:
        strs[i][j] = strs[i][j].replace("_", "-")

    ncols = len(strs[0])
    seps = "".join(["c" for i in range(ncols)])
    s = []
    s.append("\\begin{table}")
    s.append("\\centering")
    s.append("\\begin{tabular}{%s}" % seps)
    s.append(" & ".join(strs[0]) + " \\\\\\hline")
    for line in strs[1:]:
      s.append(" & ".join(line) + " \\\\")
    s.append("\\end{tabular}")
    s.append("\\end{table}")

    for i in range(len(strs)):
      for j in range(len(strs[i])):
        if "_" in strs[i][j]:
          strs[i][j] = strs[i][j].replace("\\_", "_")

  return "\n".join(s)


def str_csv_table(strs):
  """Format a string table to a csv table."""
  s = []
  for i in range(len(strs)):
    s.append(",".join(strs[i]))
  return "\n".join(s)
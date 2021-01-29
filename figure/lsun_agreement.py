import sys, glob, argparse, os
sys.path.insert(0, ".")
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="expr", help="")
parser.add_argument("--task", default="bedroom", help="")
parser.add_argument("--model", default="stylegan2", help="")
args = parser.parse_args()

import evaluate
from utils.misc import format_test_result, listkey_convert

THRESHOLD = 0.1
data_dir = args.dir
files = glob.glob(data_dir + f"/{args.model}_{args.task}_*")
files = [f"{f}/evaluation.npy" for f in files if os.path.exists(f"{f}/evaluation.npy") and "celebahq" not in f and "ffhq" not in f]
files.sort()
print(files)
if len(files) <= 0:
    exit(0)
metric = evaluate.DetectionMetric(n_class=150)

label_list = open("figure/ade20k_labels.txt", "r").read().split("\n")
label_list = [l.split(" ")[-1].split(",")[0] for l in label_list]

print(label_list)

def get_name(model_path):
    names = ["LSE-WF", "LSE-W", "LSE-F", "NSE-1", "NSE-2", "UNet-512", "LSE"]
    return listkey_convert(model_path, names)


def get_topk_classes(dic, namelist):
    res = {}
    for metric_name in ["IoU"]:
        x = np.array(dic[metric_name])
        y = x.argsort()
        k = 0
        while x[y[k]] < THRESHOLD:
            k += 1
        y = y[k:]
        y.sort()
        names = namelist[y] 
        res[metric_name] = x[y] #(names, x[y])
    return res, names.tolist()


def get_row(table, ind, name, delimiter=","):
    row = table.split("\n")[ind + 1]
    items = row.split(delimiter)
    return delimiter.join([name] + items)


class Summary(object):
    def __init__(self):
        self.gt = []
        self.ct = []
        self.gu = []
        self.cu = []
        self.reset()

    def reset(self):
        self.global_table = []
        self.class_table = []
        self.global_tabular = []
        self.class_tabular = []
        self.ct = []
        self.cu = []

    def process_result(self, res, name):
        global_csv, class_csv = res[2:]
        self.global_table.append(get_row(global_csv, 0, name))
        self.class_table.append(get_row(class_csv, 0, name))
        global_latex, class_latex = res[:2]
        self.global_tabular.append(get_row(global_latex, 1, name, " & "))
        self.class_tabular.append(get_row(class_latex, 1, name, " & "))

        self.global_csv_head = global_csv.split("\n")[0]
        self.class_csv_head = class_csv.split("\n")[0]
        self.global_latex_head = global_latex.split("\n")[:2]
        self.class_latex_head = class_latex.split("\n")[:2]

    def write_class(self, subfix="object"):
        self.ct = [self.class_csv_head] + self.class_table
        self.cu = self.class_latex_head + self.class_tabular
        with open(f"{subfix}_class_result.csv", "w") as f:
            f.write("\n".join(self.ct))
        with open(f"{subfix}_class_tabular.tex", "w") as f:
            f.write("\n".join(self.cu))

    def write_global(self, subfix="object"):
        self.gt = [self.global_csv_head] + self.global_table
        self.gu = self.global_latex_head + self.global_tabular
        with open(f"{subfix}_global_result.csv", "w") as f:
            f.write("\n".join(self.gt))
        with open(f"{subfix}_global_tabular.tex", "w") as f:
            f.write("\n".join(self.gu))

object_summary = Summary()
material_summary = Summary()

# get all classes first
all_objects = []
for f in files:
    name = get_name(f)
    global_dic, dic = np.load(f, allow_pickle=True)[()]
    val, names = get_topk_classes(dic, np.array(label_list))
    all_objects.append(set(names))

all_objects = list(set.union(*all_objects))
obj_inds = np.array([label_list.index(n) for n in all_objects])
print(all_objects)

# find maximum first
obj_max = []
mat_max = []
for f in files:
    name = get_name(f)
    global_dic, dic = np.load(f, allow_pickle=True)[()]
    metric.result = dic
    metric.subset_aggregate("common", obj_inds)

    ious = [dic["IoU"][i] for i in obj_inds]
    ious = np.array([v if v > 0 else 0 for v in ious])
    obj_max.append(ious.mean())
obj_max = max(obj_max)

for f in files:
    name = get_name(f)
    global_dic, dic = np.load(f, allow_pickle=True)[()]
    metric.result = dic
    metric.subset_aggregate("common", obj_inds)
    ious = [dic["IoU"][i] for i in obj_inds]
    ious = np.array([v if v > 0 else 0 for v in ious])
    add = [ious.mean(), (obj_max - ious.mean()) / obj_max * 100]
    object_dic = {
        "mIoU_common" : ious.mean(),
        "IoU" : add + list(ious)}
    res = format_test_result(object_dic,
        global_metrics=["mIoU_common"],
        class_metrics=["IoU"],
        label_list=["mIoU", "Diff."] + all_objects)
    object_summary.process_result(res, name)

object_summary.write_class(f"{args.model}_{args.task}_object")
object_summary.write_global(f"{args.model}_{args.task}_object")
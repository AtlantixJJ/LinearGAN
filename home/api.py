"""API for interaction between GAN and django views.
"""
import sys
sys.path.insert(0, '.')
import time, json, os, torch, threading
import numpy as np

from home.utils import *
from models.helper import *
from lib.op import sample_image_feature, torch2image, torch2numpy, bu
from lib.misc import imwrite
from lib.visualizer import segviz_numpy, get_label_color
from models.semantic_extractor import LSE, SEFewShotLearner
from manipulation.sie import ImageEditing
from manipulation.strategy import EditStrategy
from train_fewshot import get_features


class AddDataThread(threading.Thread):
  def __init__(self, training_thread, f, l, lm):
    threading.Thread.__init__(self)
    self.training_thread = training_thread
    self.f = f
    self.l = l
    self.lm = lm
  
  def run(self):
    self.training_thread.add_annotation(self.f, self.l, self.lm)


class TrainingThread(threading.Thread):
  def __init__(self, learner, max_iter=1000):
    threading.Thread.__init__(self)
    self.learner = learner
    self.lock = threading.Lock()
    self.max_iter = max_iter
    self.reset()

  def add_annotation(self, feature, label, label_mask):
    self.lock.acquire()
    self.features.append(feature)
    self.labels.append(label)
    self.labels_mask.append(label_mask)
    self.learner.feature = [torch.cat([feats[i] for feats in self.features]) for i in range(len(feature))]
    self.learner.label = torch.cat(self.labels).long()
    self.lock.release()

  def reset(self, features=[], labels=[], labels_mask=[]):
    self.lock.acquire()
    self.count = 0
    self.exit = False
    self.running = False
    self.labels = labels
    self.features = features
    self.labels_mask = labels_mask
    self.learner.features = features
    self.learner.labels = labels
    self.lock.release()

  def _check_has_data(self):
    while True:
      time.sleep(1)
      self.lock.acquire()
      if len(self.features) > 0:
        self.lock.release()
        break
      self.lock.release()

  def send_command(self, cmd):
    if cmd == "start":
      if self.running:
        return False
      self.lock.acquire()
      self.running = True
      self.lock.release()
      self.start()
      return True
    elif cmd == "pause":
      if not self.running:
        return False
      self.lock.acquire()
      self.running = False
      self.lock.release()
      return True
    elif cmd == "stop":
      self.lock.acquire()
      self.exit = True
      self.lock.release()
      return True
    elif cmd == "val":
      self.lock.acquire()
      images = []
      segvizs = []
      for _ in range(6):
        z = torch.randn(1, 512).cuda()
        with torch.no_grad():
          image, segs = self.learner(z)
        seg = bu(segs[-1], 128).argmax(1)
        segvizs.append(segviz_numpy(seg.detach().cpu().numpy()))
        images.append(torch2image(bu(image, 128)).astype("uint8"))
      self.lock.release()
      return np.concatenate(images), np.stack(segvizs)

  def run(self):
    self._check_has_data()
    self.count = 0
    self.exit = False
    optim = self.learner.configure_optimizers()[0][0]
    print("=> Training thread started")
    while not self.exit:
      while not self.running: # wait for running
        time.sleep(1)
      self.lock.acquire()
      optim.zero_grad()
      self.learner.training_step(None, self.count).backward()
      optim.step()
      self.count += 1
      self.lock.release()
      time.sleep(1e-3)
    print("=> Training thread ended")


def create_fewshot_LSE(G, n_class=36):
  """Create a LSE model for fewshot learning purpose."""
  with torch.no_grad():
    _, features = sample_image_feature(G)
  layers = [i for i in range(len(features)) \
    if i % 2 == 1 and features[i].size(3) >= 32]
  dims = [features[i].size(1) for i in layers]
  return LSE(n_class=n_class, dims=dims, layers=layers)


class TrainAPI(object):
  def __init__(self, MA):
    self.MA = MA
    self.Gs = MA.Gs # G
    self.SE = MA.SE_new # LSE
    self.SELeaner = MA.SELearner # LSE Learner
    self.training_thread = {k : TrainingThread(v) for k, v in self.SELeaner.items()}
    self.data_dir = MA.data_dir

  def reset_train_model(self, model_name):
    print("=> [TrainerAPI] reset LSE model")
    sdict = self.SE[model_name].arch_info()
    del self.SE[model_name]
    self.SE[model_name] = LSE(**sdict).cuda()
    print("=> [TrainerAPI] done")

  def generate_new_image(self, model_name):
    print("=> [TrainerAPI] generate new image")
    G = self.Gs[model_name]
    z = torch.randn(1, 512).cuda()
    zs = z.repeat(G.num_layers, 1) # use mixwp
    wp = G.mapping(z).unsqueeze(1).repeat(1, G.num_layers, 1)
    image, feature = G.synthesis(wp, generate_feature=True)
    zs = zs.detach().cpu().view(-1).numpy().tolist()
    image = torch2image(image).astype("uint8")[0]
    print("=> [TrainerAPI] done")
    return image, zs
  
  def get_validation(self, model_name):
    """Return validation images. """
    train_thread = self.training_thread[model_name]
    n_iter = train_thread.count
    print(f"=> [TrainerAPI] validate on iteration {n_iter}")
    G = self.Gs[model_name]
    SE = self.SE[model_name]
    image, segviz = train_thread.send_command("val")
    print("=> [TrainerAPI] done")
    return image, segviz

  def ctrl_training(self, model_name, command):
    """Control the training using commands.
      Args:
        model_name : The G name.
        command : [start] -> start training; [stop] -> stop training;
    """
    return self.training_thread[model_name].send_command(command)

  def add_annotation(self, model_name, zs, ann, ann_mask):
    """Add annotations."""
    # select model-specific data
    G = self.Gs[model_name]
    SE = self.SE[model_name]
    train_thread = self.training_thread[model_name]

    # parse and store data
    zs = np.array(zs, dtype=np.float32).reshape((G.num_layers, -1))
    time_str = get_time_str()
    p = f"{self.data_dir}/{time_str}"
    np.save(f"{p}_origin-zs.npy", zs)
    imwrite(f"{p}_ann.png", ann)
    imwrite(f"{p}_ann-mask.png", ann_mask)

    # preprocess data
    zs = torch.from_numpy(zs).float().cuda()
    size = self.MA.models_config[model_name]["output_size"]
    wp = EditStrategy.z_to_wp(G, zs, in_type="zs", out_type="notrunc-wp")
    label_stroke = preprocess_label(ann, SE.n_class, size).cuda()
    with torch.no_grad():
      image, feature = G.synthesis(wp, generate_feature=True)
      label_mask = preprocess_mask(ann_mask, size).squeeze(1).cuda()
    
    # add data into training thread
    AddDataThread(train_thread, feature, label_stroke, label_mask).start()


class ModelAPI(object):
  def update_config(self):
    with open(self.config_file, 'r') as f:
      self.config = json.load(f)
    self.models_config = self.config['models']
    self.data_dir = self.config['collect_data_dir']

  def init_model(self):
    self.Gs = {}
    self.SE = {}
    self.SE_new = {}
    self.SELearner = {}
    for name, mc in self.models_config.items():
      G = build_generator(mc["model_name"]).net
      self.Gs[name] = G # [TODO]: MultiGPU
      SE = None if len(mc["SE"]) == 0 else \
        load_semantic_extractor(mc["SE"]).cuda()
      self.SE[name] = SE
      SE = create_fewshot_LSE(G)
      SE.cuda().train()
      self.SE_new[name] = SE
      self.SELearner[name] = SEFewShotLearner(SE, G)

  def __init__(self, config_file):
    self.config_file = config_file
    self.update_config()
    self.init_model()


class EditAPI(object):
  def __init__(self, MA):
    self.ma = MA
    self.Gs = MA.Gs
    self.SE = MA.SE
    self.data_dir = MA.data_dir
    
  def has_model(self, model_name):
    return model_name in list(self.Gs.keys())

  def generate_image_given_stroke(self, model_name, zs,
    image_stroke, image_mask, label_stroke, label_mask):
    G, SE = self.Gs[model_name], self.SE[model_name]
    zs = np.array(zs, dtype=np.float32).reshape((G.num_layers, -1))
    time_str = get_time_str()
    p = f"{self.data_dir}/{time_str}"
    np.save(f"{p}_origin-zs.npy", zs)
    imwrite(f"{p}_image-stroke.png", image_stroke)
    imwrite(f"{p}_label-stroke.png", label_stroke)
    imwrite(f"{p}_image-mask.png", image_mask)
    imwrite(f"{p}_label-mask.png", label_mask)

    size = self.ma.models_config[model_name]["output_size"]
    zs = torch.from_numpy(zs).float().cuda().unsqueeze(0) # (1, 18, 512)
    wp = EditStrategy.z_to_wp(G, zs,
      in_type="zs", out_type="notrunc-wp")
    image_stroke = preprocess_image(image_stroke, size).cuda()
    image_mask = preprocess_mask(image_mask, size).cuda()
    label_stroke = preprocess_label(label_stroke, SE.n_class, size)
    label_mask = preprocess_mask(label_mask, size).squeeze(1).cuda()
    fused_int_label = ImageEditing.fuse_stroke(
      G, SE, None, wp,
      image_stroke[0], image_mask[0],
      label_stroke[0], label_mask[0])["fused_int_label"]
    zs, wp = ImageEditing.sseg_edit(
      G, zs, fused_int_label, label_mask, SE,
      op="internal",
      latent_strategy="mixwp",
      optimizer='adam',
      n_iter=50,
      base_lr=0.01)

    image, feature = G.synthesis(wp.cuda(), generate_feature=True)
    label = SE(feature)[-1].argmax(1)
    image = torch2image(image)[0]
    label_viz = segviz_numpy(torch2numpy(label))
    zs = zs.detach().cpu().view(-1).numpy().tolist()
    imwrite(f"{p}_new-image.png", image) # generated
    imwrite(f"{p}_new-label.png", label_viz)
    return image, label_viz, zs

  def generate_new_image(self, model_name):
    G = self.Gs[model_name]
    z = torch.randn(1, 512).cuda()
    zs = z.repeat(G.num_layers, 1) # use mixwp
    wp = G.mapping(z).repeat(G.num_layers, 1).unsqueeze(0)
    image, feature = G.synthesis(wp, generate_feature=True)
    seg = self.SE[model_name](feature)[-1]
    label = seg[0].argmax(0)
    image = torch2image(image).astype("uint8")[0]
    label_viz = segviz_numpy(torch2numpy(label))
    zs = zs.detach().cpu().view(-1).numpy().tolist()
    return image, label_viz, zs

import sys
sys.path.insert(0, ".")

from .base_predictor import BasePredictor
from torchvision.transforms import Normalize
import torch
import torch.nn.functional as F

from lib.misc import read_selected_labels, read_ade20k_labels
from lib.op import bu

ADE_NUMCLASSES = 150
SELECTED_LABELS = read_selected_labels()
ADE20K_LABELS = read_ade20k_labels()


class SceneSegmenter(BasePredictor):
  def __init__(self,
               predictor_name='scene_seg',
               model_name=""):
    if model_name in SELECTED_LABELS:
      self.labels = SELECTED_LABELS[model_name]
      self.label_indice = [ADE20K_LABELS.index(l) - 1 for l in self.labels]
    else:
      print(f"!> {model_name} not seen. Use default ADE labels")
    super().__init__(predictor_name)
    self.input_transform = Normalize([.485, .456, .406], [.229, .224, .225])

  def build(self):
    from encoding.models import get_model
    self.net = get_model("DeepLab_ResNeSt200_ADE", pretrained=True).eval()
    self.net.aux = False
    self.num_categories = self.net.nclass + 1 # add background class
    if hasattr(self, "label_indice"):
      self.num_categories = len(self.labels) + 1
      print(f"=> Using partial label {','.join(self.labels)}")
    
  def load(self):
    pass

  def raw_prediction(self, images, size=256):
    """
    Args:
      images : torch.Tensor in [-1, 1]
      size : The target resolution
    """
    x = torch.stack([self.input_transform((1 + i) / 2)
      for i in images])
    y = self.net(x, stride=2)[0]
    if hasattr(self, "label_indice"):
      y = y[:, self.label_indice]
    if y.size(2) != size:
      y = bu(x, size)
    return torch.cat([torch.zeros_like(y[:, :1]), y], 1)

  def __call__(self, images, size=256):
    """
    Args:
      images : torch.Tensor in [-1, 1]
      size : The target resolution
    """
    x = torch.stack([self.input_transform((1 + i) / 2)
      for i in images])
    y = self.net(x, stride=2)[0]
    if hasattr(self, "label_indice"):
      y = y[:, self.label_indice]
    if y.size(2) != size:
      y = bu(x, size)
    y = torch.cat([torch.zeros_like(y[:, :1]), y], 1)
    return y.argmax(1)
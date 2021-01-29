import torch, math
from torch import nn

def sine_init(m, omega0=30):
  with torch.no_grad():
    if hasattr(m, 'weight'):
      num_input = m.weight.size(-1)
      T = math.sqrt(6 / num_input) / omega0
      m.weight.uniform_(-T, T)


def first_layer_sine_init(m):
  with torch.no_grad():
    if hasattr(m, 'weight'):
      num_input = m.weight.size(-1)
      m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
  def __init__(self, omega0=30):
    super().__init__()
    self.omega0 = omega0

  def forward(self, input):
    return torch.sin(self.omega0 * input)

class EditStrategy(object):
  """Manage the optimization, latent space choice and learning rate adaption.
  """
  def __init__(self,
               G,
               latent_strategy='mixwp',
               optimizer='adam',
               n_iter=100,
               base_lr=0.01):
    self.G = G.net
    self.num_layers = G.net.num_layers
    self.G_name = G.model_name
    self.n_iter = n_iter
    self.latent_strategy = latent_strategy
    self.optimizer_type = {"adam" : torch.optim.Adam}[optimizer]
    self.base_lr = base_lr
  
  def to_std_form(self):
    if self.latent_strategy == "mixwp":
      z = torch.stack(self.zs)
      return z, self.G.mapping(z).unsqueeze(0) # (1, L, 512)
    if self.latent_strategy == "z":
      wp = self.G.mapping(self.z).unsqueeze(1)
      return self.z, wp.repeat(1, self.num_layers, 1)

  def setup(self, z_init):
    """
    """
    if self.latent_strategy == "z":
      self.z = z_init.clone().cuda().detach().requires_grad_(True)
      self.optim = self.optimizer_type([self.z], lr=self.base_lr)

    if self.latent_strategy == "mixwp":
      assert list(z_init.shape) == [1, 512]
      get_lr = self.get_layer_lr_func()
      
      if hasattr(self, "zs"): # clear previous results
        del self.z0
        del self.zs
        del self.optims
      self.z0 = z_init.repeat(self.num_layers, 1).cuda()
      # each item in self.zs is of shape (512,)
      self.zs = [self.z0[i].clone().detach().requires_grad_(True)
        for i in range(self.z0.shape[0])]

      self.optims = [self.optimizer_type([self.zs[i]], lr=get_lr(i))
        for i in range(self.z0.shape[0])]
  
  def step(self, loss):
    loss.backward()
    if self.latent_strategy == "z":
      self.optim.step()
      self.optim.zero_grad()

    if self.latent_strategy == "mixwp":
      for i in range(len(self.zs)):
        self.optims[i].step()
        self.optims[i].zero_grad()

  @staticmethod
  def get_lr_ffhq(layer_idx, base_lr=0.01):
    if layer_idx <= 7:
      return base_lr
    elif layer_idx <= 11:
      return base_lr / 10
    elif layer_idx <= 15:
      return base_lr / 100
    else:
      return base_lr / 1000


  @staticmethod
  def get_lr_bedroom(layer_idx, base_lr=0.01):
    if layer_idx <= 5:
      return base_lr / 10
    elif layer_idx <= 11: # [6-12]
      return base_lr
    else:
      return base_lr / 10


  def get_layer_lr_func(self):
    """
    Returns a function get_lr(layer_idx).
    """
    funcs = {
      "stylegan2_ffhq" : EditStrategy.get_lr_ffhq,
      "stylegan2_bedroom" : EditStrategy.get_lr_bedroom}
    return lambda i : funcs[self.G_name](i, self.base_lr)
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../lib")
import torch
import model
from model.semantic_extractor import load_extractor
from home import utils

class WrapedStyledGenerator(torch.nn.Module):
    def __init__(self, resolution=1024, method="", model_path="", n_class=15, category_groups=None, sep_model_path="", gpu=-1):
        super(WrapedStyledGenerator, self).__init__()
        self.device = 'cuda' if gpu >= 0 else 'cpu'
        self.model_path = model_path
        self.method = method
        self.sep_model_path = sep_model_path
        self.external_model = None
        self.category_groups = None
        self.n_class = n_class

        print("=> Constructing network architecture")
        self.model = model.load_model(self.model_path)
        print("=> Loading parameter from %s" % self.model_path)
        state_dict = torch.load(self.model_path, map_location='cpu')
        missed = self.model.load_state_dict(state_dict, strict=False)
        print(missed)

        t = "identity"
        if "stylegan" in model_path:
            if not hasattr(self.model, "g_mapping"):
                self.mapping_network = self.model.style
                t = "style"
            else:
                self.mapping_network = self.model.g_mapping.simple_forward
                t = "g_mapping"
        else:
            self.mapping_network = lambda x : x
        print(f"=> Resolve mapping function: {t}")

        try:
            self.model = self.model.to(self.device)
        except:
            print("=> Fall back to CPU")
            self.device = 'cpu'
        
        self.model.eval()

        print("=> Check running")
        self.noise_length = self.model.set_noise(None)

        print("=> Optimization method %s" % str(self.method))

        self.latent_param = torch.randn(1, 512,
            requires_grad=True, device=self.device)

        with torch.no_grad():
            image, stage = self.model.get_stage(self.latent_param)
            dims = [s.shape[1] for s in stage]

        self.layers = list(range(len(dims)))
        if "layer" in sep_model_path:
            ind = sep_model_path.rfind("layer") + len("layer")
            s = sep_model_path[ind:].split("_")[0]
            if ".model" in s:
                s = s[:s.rfind(".")]
            self.layers = [int(i) for i in s.split(",")]
            dims = [dims[i] for i in self.layers]

        self.sep_model = load_extractor(sep_model_path, category_groups, dims)
        self.sep_model.to(self.device).eval()

    def generate_noise(self):
        print(self.noise_length)
        sizes = [4 * 2 ** (i // 2) for i in range(self.noise_length)]
        length = sum([size ** 2 for size in sizes])
        latent = torch.randn(1, 512, device=self.device)
        noise_vec = torch.randn((length,), device=self.device)
        return latent, noise_vec

    def generate_given_image_stroke(self, latent, noise, image_stroke, image_mask):
        utils.copy_tensor(self.latent_param, latent)
        self.mix_latent_param = self.latent_param.expand(self.noise_length, -1).detach()
        noises = self.model.parse_noise(noise)

        if "ML" in self.method:
            self.param = self.mix_latent_param
        else:
            self.param = self.latent_param

        image, label, latent, noises, record = edit_image_stroke(
            model=self.model, latent=self.latent_param, noises=noises, 
            image_stroke=image_stroke, image_mask=image_mask,
            method=self.method,
            sep_model=self.sep_model, mapping_network=self.mapping_network)

        # Currently no modification to noise
        # noise = torch.cat([n.view(-1) for n in noise])

        image = utils.torch2numpy(image * 255).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(label)
        latent = utils.torch2numpy(latent)
        noise = utils.torch2numpy(noise)

        return image.astype("uint8"), label, latent, noise, record

    def generate_given_label_stroke(self, latent, noise, label_stroke, label_mask):
        utils.copy_tensor(self.latent_param, latent)
        self.mix_latent_param = self.latent_param.expand(self.noise_length, -1).detach()
        if "ML" in self.method:
            self.param = self.mix_latent_param
        else:
            self.param = self.latent_param
        noises = self.model.parse_noise(noise)

        image, label, latent, noises, record = edit_label_stroke(
            model=self.model, latent=self.param, noises=noises, label_stroke=label_stroke, label_mask=label_mask,
            method=self.method.replace("image", "label"),
            sep_model=self.sep_model, mapping_network=self.mapping_network)
        
        # Currently no modification to noise
        # noise = torch.cat([n.view(-1) for n in noise])

        image = utils.torch2numpy(image * 255).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(label)
        latent = utils.torch2numpy(latent)
        noise = utils.torch2numpy(noise)
        
        return image.astype("uint8"), label, latent, noise, record


    def forward(self, latent, noise): # [0, 1] in torch
        self.model.set_noise(self.model.parse_noise(noise))
        image, stage = self.model.get_stage(latent)
        seg = self.sep_model(stage)[0]

        image = (1 + image.clamp(-1, 1)) * 255 / 2
        image = utils.torch2numpy(image).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(seg.argmax(1))
        return image.astype("uint8"), label

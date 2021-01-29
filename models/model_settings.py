# python 3.7
"""Contains basic configurations for models used in this project.

Please download the public released models from the following repositories
OR train your own models, and then put them into the folder
`pretrain/tensorflow`.

PGGAN: https://github.com/tkarras/progressive_growing_of_gans
StyleGAN: https://github.com/NVlabs/stylegan
StyleGAN2: https://github.com/NVlabs/stylegan2
TODO: In-Domain Inversion

NOTE: Any new model should be registered in `MODEL_POOL` before used.
"""

import os

BASE_DIR = os.path.dirname(os.path.relpath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, 'pretrain')
PTH_MODEL_DIR = 'pytorch'
TF_MODEL_DIR = 'tensorflow'

if not os.path.exists(os.path.join(MODEL_DIR, PTH_MODEL_DIR)):
  os.makedirs(os.path.join(MODEL_DIR, PTH_MODEL_DIR))

MODEL_POOL = {
    # PGGAN Official.
    'pggan_celebahq': {
        'resolution': 1024,
        'tf_weight_name': 'karras2018iclr-celebahq-1024x1024.pkl',
    },
    'pggan_bedroom': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-bedroom-256x256.pkl',
    },
    'pggan_livingroom': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-livingroom-256x256.pkl',
    },
    'pggan_diningroom': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-diningtable-256x256.pkl',
    },
    'pggan_kitchen': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-kitchen-256x256.pkl',
    },
    'pggan_church': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-churchoutdoor-256x256.pkl',
    },
    'pggan_tower': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-tower-256x256.pkl',
    },
    'pggan_bridge': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-bridge-256x256.pkl',
    },
    'pggan_restaurant': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-restaurant-256x256.pkl',
    },
    'pggan_classroom': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-classroom-256x256.pkl',
    },
    'pggan_conferenceroom': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-conferenceroom-256x256.pkl',
    },
    'pggan_person': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-person-256x256.pkl',
    },
    'pggan_cat': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-cat-256x256.pkl',
    },
    'pggan_dog': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-dog-256x256.pkl',
    },
    'pggan_bird': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-bird-256x256.pkl',
    },
    'pggan_horse': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-horse-256x256.pkl',
    },
    'pggan_sheep': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-sheep-256x256.pkl',
    },
    'pggan_cow': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-cow-256x256.pkl',
    },
    'pggan_car': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-car-256x256.pkl',
    },
    'pggan_bicycle': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-bicycle-256x256.pkl',
    },
    'pggan_motorbike': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-motorbike-256x256.pkl',
    },
    'pggan_bus': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-bus-256x256.pkl',
    },
    'pggan_train': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-train-256x256.pkl',
    },
    'pggan_boat': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-boat-256x256.pkl',
    },
    'pggan_airplane': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-airplane-256x256.pkl',
    },
    'pggan_bottle': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-bottle-256x256.pkl',
    },
    'pggan_chair': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-chair-256x256.pkl',
    },
    'pggan_pottedplant': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-pottedplant-256x256.pkl',
    },
    'pggan_tvmonitor': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-tvmonitor-256x256.pkl',
    },
    'pggan_diningtable': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-diningtable-256x256.pkl',
    },
    'pggan_sofa': {
        'resolution': 256,
        'tf_weight_name': 'karras2018iclr-sofa-256x256.pkl',
    },

    # StyleGAN Official.
    'stylegan_ffhq': {
        'resolution': 1024,
        'tf_weight_name': 'karras2019stylegan-ffhq-1024x1024.pkl',
    },
    'stylegan_celebahq': {
        'resolution': 1024,
        'tf_weight_name': 'karras2019stylegan-celebahq-1024x1024.pkl',
    },
    'stylegan_bedroom': {
        'resolution': 256,
        'tf_weight_name': 'karras2019stylegan-bedroom-256x256.pkl',
    },
    'stylegan_cat': {
        'resolution': 256,
        'tf_weight_name': 'karras2019stylegan-cats-256x256.pkl',
    },
    'stylegan_car': {
        'resolution': 512,
        'tf_weight_name': 'karras2019stylegan-cars-512x384.pkl',
    },

    # StyleGAN Self-Training.
    'stylegan_mnist': {
        'resolution': 32,
        'image_channels': 1,
        'tf_weight_name': 'stylegan-mnist-32x32.pkl',
    },
    'stylegan_mnist_cond': {
        'resolution': 32,
        'image_channels': 1,
        'label_size': 10,
        'tf_weight_name': 'stylegan-mnist_cond-32x32-020000.pkl',
    },
    'stylegan_mnist_color': {
        'resolution': 32,
        'tf_weight_name': 'stylegan-mnist_color-32x32-020000.pkl',
    },
    'stylegan_mnist_color_cond': {
        'resolution': 32,
        'label_size': 10,
        'tf_weight_name': 'stylegan-mnist_color_cond-32x32-020000.pkl',
    },
    'stylegan_svhn': {
        'resolution': 32,
        'tf_weight_name': 'stylegan-svhn-32x32-030000.pkl',
    },
    'stylegan_svhn_cond': {
        'resolution': 32,
        'label_size': 10,
        'tf_weight_name': 'stylegan-svhn_cond-32x32-030000.pkl',
    },
    'stylegan_cifar10': {
        'resolution': 32,
        'tf_weight_name': 'stylegan-cifar10-32x32-030000.pkl',
    },
    'stylegan_cifar10_cond': {
        'resolution': 32,
        'label_size': 10,
        'tf_weight_name': 'stylegan-cifar10_cond-32x32-030000.pkl',
    },
    'stylegan_cifar100': {
        'resolution': 32,
        'tf_weight_name': 'stylegan-cifar100-32x32-030000.pkl',
    },
    'stylegan_cifar100_cond': {
        'resolution': 32,
        'label_size': 100,
        'tf_weight_name': 'stylegan-cifar100_cond-32x32-030000.pkl',
    },
    'stylegan_celeba_partial': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-celeba_partial-256x256-050000.pkl',
    },
    'stylegan_ffhq256': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-ffhq-256x256.pkl',
    },
    'stylegan_ffhq512': {
        'resolution': 512,
        'tf_weight_name': 'stylegan-ffhq-512x512.pkl',
    },
    'stylegan_livingroom': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-livingroom-256x256.pkl',
    },
    'stylegan_diningroom': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-diningroom-256x256.pkl',
    },
    'stylegan_kitchen': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-kitchen-256x256.pkl',
    },
    'stylegan_apartment': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-apartment-256x256.pkl',
    },
    'stylegan_church': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-church-256x256.pkl',
    },
    'stylegan_tower': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-tower-256x256.pkl',
    },
    'stylegan_bridge': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-bridge-256x256.pkl',
    },
    'stylegan_restaurant': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-restaurant-256x256.pkl',
    },
    'stylegan_classroom': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-classroom-256x256.pkl',
    },
    'stylegan_conferenceroom': {
        'resolution': 256,
        'tf_weight_name': 'stylegan-conferenceroom-256x256.pkl',
    },

    # StyleGAN from Third-Party.
    'stylegan_animeface': {
        'resolution': 512,
        'tf_weight_name': 'stylegan-animeface-512x512.pkl',
    },
    'stylegan_animeportrait': {
        'resolution': 512,
        'tf_weight_name': 'stylegan-animeportrait-512x512.pkl',
    },
    'stylegan_artface': {
        'resolution': 512,
        'tf_weight_name': 'stylegan-artface-512x512.pkl',
    },

    # StyleGAN2 Official.
    'stylegan2_ffhq': {
        'resolution': 1024,
        'tf_weight_name': 'karras2020stylegan2-ffhq-1024x1024.pkl',
    },
    'stylegan2_church': {
        'resolution': 256,
        'tf_weight_name': 'karras2020stylegan2-church-256x256.pkl',
    },
    'stylegan2_cat': {
        'resolution': 256,
        'tf_weight_name': 'karras2020stylegan2-cat-256x256.pkl',
    },
    'stylegan2_horse': {
        'resolution': 256,
        'tf_weight_name': 'karras2020stylegan2-horse-256x256.pkl',
    },
    'stylegan2_car': {
        'resolution': 512,
        'tf_weight_name': 'karras2020stylegan2-car-512x384.pkl',
    },

    # StyleGAN2 Self-Training.
    'stylegan2_mnist': {
        'resolution': 32,
        'image_channels': 1,
        'tf_weight_name': 'stylegan2-mnist-32x32.pkl',
    },
    'stylegan2_mnist_cond': {
        'resolution': 32,
        'image_channels': 1,
        'label_size': 10,
        'tf_weight_name': 'stylegan2-mnist_cond-32x32.pkl',
    },
    'stylegan2_mnist_color': {
        'resolution': 32,
        'tf_weight_name': 'stylegan2-mnist_color-32x32.pkl',
    },
    'stylegan2_mnist_color_cond': {
        'resolution': 32,
        'label_size': 10,
        'tf_weight_name': 'stylegan2-mnist_color_cond-32x32.pkl',
    },
    'stylegan2_svhn': {
        'resolution': 32,
        'tf_weight_name': 'stylegan2-svhn-32x32.pkl',
    },
    'stylegan2_svhn_cond': {
        'resolution': 32,
        'label_size': 10,
        'tf_weight_name': 'stylegan2-svhn_cond-32x32.pkl',
    },
    'stylegan2_cifar10': {
        'resolution': 32,
        'tf_weight_name': 'stylegan2-cifar10-32x32.pkl',
    },
    'stylegan2_cifar10_cond': {
        'resolution': 32,
        'label_size': 10,
        'tf_weight_name': 'stylegan2-cifar10_cond-32x32.pkl',
    },
    'stylegan2_cifar100': {
        'resolution': 32,
        'tf_weight_name': 'stylegan2-cifar100-32x32.pkl',
    },
    'stylegan2_cifar100_cond': {
        'resolution': 32,
        'label_size': 100,
        'tf_weight_name': 'stylegan2-cifar100_cond-32x32.pkl',
    },
    'stylegan2_imagenet': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-imagenet-256x256-250000.pkl',
    },
    'stylegan2_imagenet_cond': {
        'resolution': 256,
        'label_size': 1000,
        'tf_weight_name': 'stylegan2-imagenet_cond-256x256-250000.pkl',
    },
    'stylegan2_celeba_partial': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-celeba_partial-256x256-050000.pkl',
    },
    'stylegan2_ffhq256': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-ffhq-256x256.pkl',
    },
    'stylegan2_ffhq512': {
        'resolution': 512,
        'tf_weight_name': 'stylegan2-ffhq-512x512.pkl',
    },
    'stylegan2_bedroom': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-bedroom-256x256.pkl',
    },
    'stylegan2_livingroom': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-livingroom-256x256.pkl',
    },
    'stylegan2_diningroom': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-diningroom-256x256.pkl',
    },
    'stylegan2_kitchen': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-kitchen-256x256.pkl',
    },
    'stylegan2_apartment': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-apartment-256x256.pkl',
    },
    'stylegan2_apartment_cond': {
        'resolution': 256,
        'label_size': 4,
        'tf_weight_name': 'stylegan2-apartment_cond-256x256.pkl',
    },
    'stylegan2_tower': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-tower-256x256.pkl',
    },
    'stylegan2_bridge': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-bridge-256x256.pkl',
    },
    'stylegan2_restaurant': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-restaurant-256x256.pkl',
    },
    'stylegan2_classroom': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-classroom-256x256.pkl',
    },
    'stylegan2_conferenceroom': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-conferenceroom-256x256.pkl',
    },
    'stylegan2_streetscapes': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-streetscapes-256x256.pkl',
    },
    'stylegan2_places': {
        'resolution': 256,
        'tf_weight_name': 'stylegan2-places-256x256.pkl',
    },
    'stylegan2_places_cond': {
        'resolution': 256,
        'label_size': 365,
        'tf_weight_name': 'stylegan2-places_cond-256x256.pkl',
    },
    'stylegan2_cityscapes': {
        'resolution': 1024,
        'tf_weight_name': 'stylegan2-cityscapes-1024x1024.pkl',
    },
}

# Settings for StyleGAN.
STYLEGAN_TRUNCATION_PSI = 0.7  # 1.0 means no truncation
STYLEGAN_TRUNCATION_LAYERS = 8  # 0 means no truncation
STYLEGAN_RANDOMIZE_NOISE = False

# Settings for StyleGAN2.
STYLEGAN2_TRUNCATION_PSI = 0.5  # 1.0 means no truncation
STYLEGAN2_TRUNCATION_LAYERS = 18  # 0 means no truncation
STYLEGAN2_RANDOMIZE_NOISE = False

# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 4

MAX_IMAGES_ON_RAM = 800


def get_pth_weight_path(weight_name):
  """Gets weight path from `MODEL_DIR/PTH_MODEL_DIR`."""
  assert isinstance(weight_name, str)
  if weight_name == '':
    return ''
  if weight_name[-4:] != '.pth':
    weight_name += '.pth'
  return os.path.join(MODEL_DIR, PTH_MODEL_DIR, weight_name)


def get_tf_weight_path(weight_name):
  """Gets weight path from `MODEL_DIR/TF_MODEL_DIR`."""
  assert isinstance(weight_name, str)
  if weight_name == '':
    return ''
  return os.path.join(MODEL_DIR, TF_MODEL_DIR, weight_name)


def get_code_path(code_name):
  """Gets code path from `BASE_DIR`."""
  assert isinstance(code_name, str)
  if code_name == '':
    return ''
  return os.path.join(BASE_DIR, code_name)

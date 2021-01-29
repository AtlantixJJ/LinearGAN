# python 3.7
"""Contains basic configurations for predictors used in this project.

Please download the public released models and annotations from the following
repositories OR train your own predictor.

places365: https://github.com/CSAILVision/places365

NOTE: Any new predictor should be registered in `PREDICTOR_POOL` before used.
"""

import os.path

BASE_DIR = os.path.dirname(os.path.relpath(__file__))

ANNO_DIR = os.path.join(BASE_DIR, 'annotations')
MODEL_DIR = os.path.join(BASE_DIR, 'pretrain')

# pylint: disable=line-too-long
PREDICTOR_POOL = {
    # Scene Segmenter
    'scene_seg': {
        'weight_path': os.path.join(MODEL_DIR, 'resnest.pth'),
        'resolution': None,  # Image resize will be done automatically.
        'image_channels': 3,
        'channel_order': 'RGB',
    },
    
    # Scene Predictor.
    'scene': {
        'weight_path': os.path.join(MODEL_DIR, 'wideresnet18_places365.pth.tar'),
        'resolution': None,  # Image resize will be done automatically.
        'image_channels': 3,
        'channel_order': 'RGB',
        'category_anno_path': os.path.join(ANNO_DIR, 'categories_places365.txt'),
        'attribute_anno_path': os.path.join(ANNO_DIR, 'labels_sunattribute.txt'),
        'attribute_additional_weight_path': os.path.join(MODEL_DIR, 'W_sceneattribute_wideresnet18.npy'),
    },

    # Face Segmenter.
    'face_seg': {
        'weight_path': os.path.join(MODEL_DIR, 'faceparse_unet_512.pth'),
        'resolution': 512,
        'image_channels': 3,
        'channel_order': 'RGB',
    },

    # Face Predictor.
    'celebahq_male': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_male.pth')},
    'celebahq_smiling': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_smiling.pth')},
    'celebahq_attractive': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_attractive.pth')},
    'celebahq_wavy_hair': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_wavy_hair.pth')},
    'celebahq_young': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_young.pth')},
    'celebahq_five_oclock_shadow': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_five_oclock_shadow.pth')},
    'celebahq_arched_eyebrows': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_arched_eyebrows.pth')},
    'celebahq_bags_under_eyes': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_bags_under_eyes.pth')},
    'celebahq_bald': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_bald.pth')},
    'celebahq_bangs': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_bangs.pth')},
    'celebahq_big_lips': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_big_lips.pth')},
    'celebahq_big_nose': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_big_nose.pth')},
    'celebahq_black_hair': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_black_hair.pth')},
    'celebahq_blond_hair': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_blond_hair.pth')},
    'celebahq_blurry': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_blurry.pth')},
    'celebahq_brown_hair': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_brown_hair.pth')},
    'celebahq_bushy_eyebrows': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_bushy_eyebrows.pth')},
    'celebahq_chubby': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_chubby.pth')},
    'celebahq_double_chin': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_double_chin.pth')},
    'celebahq_eyeglasses': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_eyeglasses.pth')},
    'celebahq_goatee': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_goatee.pth')},
    'celebahq_gray_hair': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_gray_hair.pth')},
    'celebahq_heavy_makeup': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_heavy_makeup.pth')},
    'celebahq_high_cheekbones': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_high_cheekbones.pth')},
    'celebahq_mouth_slightly_open': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_mouth_slightly_open.pth')},
    'celebahq_mustache': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_mustache.pth')},
    'celebahq_narrow_eyes': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_narrow_eyes.pth')},
    'celebahq_no_beard': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_no_beard.pth')},
    'celebahq_oval_face': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_oval_face.pth')},
    'celebahq_pale_skin': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_pale_skin.pth')},
    'celebahq_pointy_nose': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_pointy_nose.pth')},
    'celebahq_receding_hairline': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_receding_hairline.pth')},
    'celebahq_rosy_cheeks': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_rosy_cheeks.pth')},
    'celebahq_sideburns': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_sideburns.pth')},
    'celebahq_straight_hair': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_straight_hair.pth')},
    'celebahq_wearing_earrings': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_wearing_earrings.pth')},
    'celebahq_wearing_hat': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_wearing_hat.pth')},
    'celebahq_wearing_lipstick': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_wearing_lipstick.pth')},
    'celebahq_wearing_necklace': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_wearing_necklace.pth')},
    'celebahq_wearing_necktie': {'weight_path': os.path.join(MODEL_DIR, 'celebahq_wearing_necktie.pth')},

    # Feature Extractor.
    'alexnet': {'architecture_type': 'AlexNet'},
    'vgg11': {'architecture_type': 'VGG'},
    'vgg13': {'architecture_type': 'VGG'},
    'vgg16': {'architecture_type': 'VGG'},
    'vgg19': {'architecture_type': 'VGG'},
    'vgg11_bn': {'architecture_type': 'VGG'},
    'vgg13_bn': {'architecture_type': 'VGG'},
    'vgg16_bn': {'architecture_type': 'VGG'},
    'vgg19_bn': {'architecture_type': 'VGG'},
    'googlenet': {'architecture_type': 'Inception'},
    'inception_v3': {'architecture_type': 'Inception'},
    'resnet18': {'architecture_type': 'ResNet'},
    'resnet34': {'architecture_type': 'ResNet'},
    'resnet50': {'architecture_type': 'ResNet'},
    'resnet101': {'architecture_type': 'ResNet'},
    'resnet152': {'architecture_type': 'ResNet'},
    'resnext50': {'architecture_type': 'ResNet'},
    'resnext101': {'architecture_type': 'ResNet'},
    'wideresnet50': {'architecture_type': 'ResNet'},
    'wideresnet101': {'architecture_type': 'ResNet'},
    'densenet121': {'architecture_type': 'DenseNet'},
    'densenet169': {'architecture_type': 'DenseNet'},
    'densenet201': {'architecture_type': 'DenseNet'},
    'densenet161': {'architecture_type': 'DenseNet'},
}
# pylint: enable=line-too-long

# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 4

MAX_IMAGES_ON_RAM = 800

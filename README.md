# Linear Semantics of Generative Adversarial Networks

This repo aims to introduce a discovery: linear transformations suffice to extract semantics from the generator's feature maps. In other words, the semantic mask corresponding to the generated image can be extracted from the generator's feature maps using `linear transformations`, which we abbreviate as the Linear Semantic Extractor (LSE). We verify this claim by comparing LSE to two nonlinear extraction methods, called Nonlinear Semantic Extractors (NSEs). It is found that the performance drop of the LSE relative to NSEs is negligible (less than 5%) in most cases.

The discovering of GAN's linear semantic encoding allows the `few-shot` learning of LSE. Using 8 annotations, the few-shot LSE can match 70% performance of its fully supervised version. The few-shot LSE can enable semantic controllable image synthesis on GANs in few-shot settings. To be specific, we propose few-shot Semantic-Precise Image Editing and few-shot Semantic Conditional Synthesis. The few-shot SPIE means to edit the semantic layout of a generated image with a few annotations. The few-shot SCS refers to sample images matching a semantic mask target with the help of a few annotations.

## Overview

The pipeline of the Linear Semantic Extractor is shown below.

![pipeline](doc/pipeline.png)

## Installation

1. We highly recommend using `conda` to install `pytorch`. `conda install torch torchvision`.

2. Install other prerequisites. `pip3 install -r requirements.txt`

3. Install [torch-encoding](https://github.com/zhanghang1989/PyTorch-Encoding). `pip3 install git+https://github.com/zhanghang1989/PyTorch-Encoding`.

Folder structure:

```
`- data # Download this folder according to step 4
| `- MaskCelebAHQ # for SCS on StyleGAN2-FFHQ
| `- trunc_stylegan2_bedroom # for SCS on StyleGAN2-Bedroom
| `- trunc_stylegan2_church # for SCS on StyleGAN2-Church
| `- collect_ffhq # for SIE
`- figure # for creating the figures in paper
`- home # part of the web application
`- expr
`- models
| `- pretrain
|   `- pytorch # Download generators here
`- predictors
  `- pretrain
    `- faceparse_unet_512.pth # Download
```


## Reproduce results in the paper

Our results are obtained on a wide range of GAN models and datasets with reproduciable setups.

### Comparison of LSE, NSE-1, and NSE-2

1. Train LSE, NSE-1, and NSE-2 for all the GAN models.

```bash
mkdir -p expr/semantics # Default experiment path
mkdir -p results/semantics # To store results
python submit.py --func sr_all_method
```

After running the command above, you are expected to get dozens of models in `expr/semantics/` folder, and their evaluation results in `results/semantics` as `txt` files.

The category list for each GAN is stored at `figure/selected_labels.csv`.

2. Collect results.

```bash
mkdir -p results/tex # Default table output path
python figure/SE_eval.py
```

After running the script, you are expected to get a latex table named `SE.tex` in `results/tex` folder for the evaluation of all the results. Besides, there are files named as `SE_{G_name}.tex` for each GAN model showing category-wise evaluation results.

3. For your reference, the evaluation results should be close to ours shown below:

![The evaluation of LSE, NSE-1, and NSE-2.](doc/se_quant.png)

4. You can visualize the segmentation by running `python figure/qualitative_paper.py --op <face, bedroom, or church>`. Our results are:

| Face dataset | Bedroom dataset | Church dataset|
|:--|:--|:--|
|![face](doc/face_paper_1.png)|![bedroom](doc/bedroom_paper_1.png)|![church](doc/church_paper_1.png)|

5. You can see if the model converges by running `python figure/train_evaluation.py`. This script will generate the evolution plot of training mIoU, similar to the one shown in the appendix of our paper.

## Few-shot learning of LSE

1. Train all the few-shot LSEs for StyleGAN2 models using the following command:

```bash
mkdir -p expr/fewshot
python submit.py --func train_fewshot
```

This will repeat each training for 5 times with different training dataset to account for the large data variance in few-shot learning.
You should expect three folders named like `{G_name}_LSE_fewshot`, each containing LSE models with different numebrs of training samples and index of training repeat.

2. Collect the results.

```bash
mkdir -p results/fewshot # default results location
python figure/fewshot_compare.py
```

The result files will have the same format as the evaluation of fully trained models.

3. We obtained the results that the few-shot LSEs achieve pretty good performance comparing to their fully trained counterpart, as shown below.

<center><img src="doc/fewshot_quant.png" width=500px alt="The evaluation of few-shot LSEs."/></center>

## Few-shot Semantic Conditional Sampling

1. After trained the few-shot LSEs, run:

```bash
mkdir -p results/scs
python submit.py --func scs
```

This will evaluate SCS on 100 semantic targets, where each target is used to sample 10 images.
Sampled latent vectors are stored to `results/scs` as `pth` files.

2. Visualize the sample results by running `python figure/fewshot_scs.py`. This will visualize the results stored in `results/scs` and save them into the same directory. Our results of SCS(8-shot LSE) for bedroom images are:

![SCS(8-shot) on StyleGAN2-Bedroom.](doc/scs_bedroom.png)

## Common issues

1. If `pytorch-lightning` is giving too much warning like `You have set 17 number of classes which is different from predicted (15) and target (15) number of classes` below, you can disable them using `python -W ignore your_script.py`.

## Acknowledgement

This work borrows a lot from (GenForce)[https://github.com/genforce].
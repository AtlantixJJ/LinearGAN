# Linear Semantics of Generative Adversarial Networks

This repo aims to introduce a discovery: linear transformations suffice to extract semantics from the generator's feature maps. In other words, the semantic mask corresponding to the generated image can be extracted from the generator's feature maps using `linear transformations`, which we abbreviate as the Linear Semantic Extractor (LSE).

We verify this claim by comparing LSE to two nonlinear extraction methods, which are abbreviated as Nonlinear Semantic Extractors. Experiments are conducted on Progressive GAN, StyleGAN, and StyleGAN2 trained on the face dataset, LSUN bedroom, and LSUN church dataset. It is found that the performance drop of the LSE relative to NSEs is negligible (less than 5%) in most cases.

Building upon the linear notion, we propose to train the linear transformation in `few-shot` settings. 8-shot LSE can match around 70% performance of the fully supervised version. Using the few-shot LSE, semantic-controllable image synthesis can be enabled on GANs. We present few-shot Semantic-Precise Image Editing and few-shot Semantic Conditional Synthesis.

## Overview

The pipeline of the Linear Semantic Extractor is shown below.

![pipeline](doc/pipeline.png)



##
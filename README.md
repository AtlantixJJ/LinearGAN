# Linear Semantics of Generative Adversarial Networks

This repo aims to introduce a discovery: linear transformations suffice to extract semantics from the generator's feature maps. In other words, the semantic mask corresponding to the generated image can be extracted from the generator's feature maps using `linear transformations`, which we abbreviate as the Linear Semantic Extractor (LSE). We verify this claim by comparing LSE to two nonlinear extraction methods, called Nonlinear Semantic Extractors (NSEs). It is found that the performance drop of the LSE relative to NSEs is negligible (less than 5%) in most cases.

The discovering of GAN's linear semantic encoding allows the `few-shot` learning of LSE. Using 8 annotations, the few-shot LSE can match 70% performance of its fully supervised version. The few-shot LSE can enable semantic controllable image synthesis on GANs in few-shot settings. To be specific, we propose few-shot Semantic-Precise Image Editing and few-shot Semantic Conditional Synthesis. The few-shot SPIE means to edit the semantic layout of a generated image with a few annotations. The few-shot SCS refers to sample images matching a semantic mask target with the help of a few annotations.

## Overview

The pipeline of the Linear Semantic Extractor is shown below.

![pipeline](doc/pipeline.png)



##

This work is built upon the following repos:

GenForce
MaskGAN
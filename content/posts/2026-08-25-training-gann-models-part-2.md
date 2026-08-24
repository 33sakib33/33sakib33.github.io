---
title: "Training GANN models"
date: 2026-08-25T12:00:00
draft: false
categories: ["Technical"]
tags: ["gann", "dcgan", "deep-learning", "generative-models"]
description: "Empirical examples of training a DCGAN to draw English digits, comparing batch sizes."
ShowToc: false
---

Deep Neural Networks are usually robust to hyper parameter choices. The training will be reasonably good with various reasonable combinations of hyper parameters.

However, Generative Adversarial Neural Networks(GANN) do not follow this pattern. GANNs are used to generate synthetic data, such as real looking fake images. I won't go deep into what a GANN is right now. I'll provide some practical facts.
Training such networks require specific hyper parameters and architectural choices.

For example:
1. DCGAN needs BatchNorm in both generator and discriminators(not in the first and last layers),
2. leaky ReLU activation function
3. Adam Optimizer
4. Strided Convolutions

There is another problem in training these networks. Its called Mode Collapse. It is where the generator keeps generating a subset of data because its easy to do.

To demonstrate how hyperparameters effect training convergence I taught an AI model how to draw digits.
These are videos of a computer learning how to draw English digits. Initially it just generates random noise and gradually digits take shape. These numbers are drawn by a computer and not a human.

Here are training snapshots from two runs:

![Training progress with batch size 64](/images/training_progress_64.png)

![Training progress with batch size 1024](/images/training_progress_1024.png)

The authors of DCGAN recommend batch size of 128.
I tried to train a DCGAN with both 128 and 1024.
The one that has better end-results are trained with batch size 128.

Link to the GANN training notebook [here](https://github.com/33sakib33/Teaching-Machines-To-Learn/blob/main/deep_learning/Gan_model.ipynb).


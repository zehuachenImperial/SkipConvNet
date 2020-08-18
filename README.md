# <b>SkipConvNet</b> [[Interspeech 2020]](https://arxiv.org/abs/2007.09131)

<img src='./Images/Enhancement_algorithms.png' width=1024>

**SkipConvNet: Skip Convolutional Neural Network for Speech
Dereverberation using Optimally Smoothed Spectral Mapping** <br>
Vinay Kothapally, Wei Xia, Shahram Ghorbani, John H.L Hansen, Wei Xue, Jing Huang<br>

**This repository contains official implementation of SkipConvNet for Reverb Challenge Corpus** <br>

## Getting started

### Required Installations
- Get [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
  installed on your machine.
- Install PyTorch ([pytorch.org](http://pytorch.org)).
- Install [Apex](https://github.com/NVIDIA/apex/) from its official repo. This
  will require CUDA 10.1 to work with the latest pytorch version (which is
`pytorch=1.3.1` as being tested against). It is used for fast mix-precision
inference and should work out of the box.
- Install [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) from its official repo.
- Install pkbar, soundfile, kaldiio, librosa 

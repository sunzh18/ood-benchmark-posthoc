# OOD代码整理---posthoc方法

此仓库整理了部分经典posthoc方法的实现，可以通过脚本一键进行模型训练以及posthocood方法测试

## Usage

### 1. Dataset Preparation for Large-scale Experiment

#### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in `./data/ilsvrc2012/train` and  `/data/ilsvrc2012/val`, respectively.

#### Out-of-distribution dataset

We have 4 OOD datasets from [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf),  [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf),  [Places](http://places2.csail.mit.edu/PAMI_places.pdf),  and [Textures](https://arxiv.org/pdf/1311.3618.pdf),  and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset, which can be download via the following links:

```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their [original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./data/ood_data/`.

### 2. Dataset Preparation for CIFAR Experiment

#### In-distribution dataset

The downloading process will start immediately upon running.

#### Out-of-distribution dataset

We provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `/data/ood_data/SVHN`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `./data/ood_data/dtd/images`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `./data/ood_data/places365/test_subset`. We randomly sample 10,000 images from the original test dataset.
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `./data/ood_data/LSUN-C`.
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `./data/ood_data/LSUN_R`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `./data/ood_data/iSUN`.

**For example, run the following commands in the root directory to download LSUN-C**:

```
cd /data/ood_data
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```

### 3. Pre-trained Model Preparation

For CIFAR, the model we used is in the `checkpoints/network/baseline`folder.

For ImageNet, the model we used in the paper is the pre-trained ResNet-50 and MobileNetv2 provided by Pytorch. The download process will start upon running.

## Preliminaries

It is tested under Python 3.9 environment, and requries some packages to be installed:

* [PyTorch](https://pytorch.org/)
* [numpy](http://www.numpy.org/)

## Training

训练方法为最基础的交叉熵损失训练，使用训练脚本 `./scripts/train_baseline.sh`可以完成训练，比如：
```shell
sh scripts/train_baseline.sh CIFAR-10/CIFAR-100 densenet
```
对于ImageNet1K数据集，只需要使用pytorch官方预训练的模型参数初始化模型即可。

## Precompute

对于一些方法如DICE、BATS、HIMPLoS，需要提前提取一些特征信息，在测试时使用这些信息进行测试，因此需要先运行 `./scripts/run_precompute.sh`脚本. 比如:

**CIFAR-10/CIFAR-100 with DenseNet:**

```shell
sh scripts/run_precompute.sh CIFAR-10/CIFAR-100 densenet
```

## Demo

正式测试脚本：包含的posthoc方法包括：MSP、ODIN、Energy、FeatureNorm、dice、dice_react、react、bats、LINE、Mahalanobis以及HIMPLoS

### 1. Demo code for CIFAR Experiment

Run `./scripts/run_eval.sh`. For example:

**CIFAR-10/CIFAR-100 with DenseNet:**

```shell
sh scripts/run_eval.sh CIFAR-10/CIFAR-100 densenet MSP
```

### 2. Demo code for Large-scale Experiment

Run `./scripts/run_eval.sh`. For example:

**ImageNet with ResNet-50:**

```shell
sh scripts/run_eval.sh imagenet resnet50 MSP
```

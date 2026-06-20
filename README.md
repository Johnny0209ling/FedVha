# FedVha

A lightweight federated learning research codebase for comparing FedAvg-style aggregation, HyperFedAvg/FedVha variants, and several baselines under non-IID client splits.

## Supported Experiments

- Datasets: CIFAR-10, CIFAR-100, MNIST, SVHN, Tiny-ImageNet
- Models: VGG16, LeNet, ResNet18
- Algorithms: FedAvg, HyperFedAvg/FedVha, FedProx, FedLC, FedAWA
- Hypernetwork ablations through `--hn_ablation`

## Setup

Install the core dependencies:

```bash
pip install -r requirements.txt
```

## Example Commands

CIFAR-10 with VGG16:

```bash
python -u main.py --dataset cifar10 --model VGG16 --algorithm fedavg --beta 0.1 --data_root ../../data --log_dir ./log_/cifar10_vgg16_beta=0.1
python -u main.py --dataset cifar10 --model VGG16 --algorithm hyperfedavg --beta 0.1 --data_root ../../data --log_dir ./log_/cifar10_vgg16_beta=0.1
```

MNIST with LeNet:

```bash
python -u main.py --dataset mnist --model LeNet --algorithm hyperfedavg --beta 0.1 --data_root ../../data --log_dir ./log_/mnist_lenet_beta=0.1
```

SVHN with VGG16:

```bash
python -u main.py --dataset svhn --model VGG16 --algorithm hyperfedavg --beta 0.1 --data_root ../../data --log_dir ./log_/svhn_vgg16_beta=0.1
```

Tiny-ImageNet with ResNet18:

```bash
python -u main.py --dataset tinyimagenet --model ResNet18 --algorithm hyperfedavg --beta 0.1 --data_root ../../data --server_val_per_class 50 --log_dir ./log_/tinyimagenet_resnet18_beta=0.1
```

## Notes

Datasets, checkpoints, logs, and generated figures are intentionally excluded from this repository. Put datasets outside the repo and pass their location with `--data_root`.

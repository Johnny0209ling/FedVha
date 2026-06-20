# FedVha

A lightweight federated learning research codebase for comparing FedAvg-style aggregation, HyperFedAvg/FedVha variants, and several baselines under non-IID client splits.

## Abstract

Federated learning enables multiple clients to collaboratively train a shared model without exposing raw data. However, real-world client data are often Non-IID, making conventional sample-size-based aggregation insufficient for estimating each local update's contribution to the global model. To address this problem, we propose FedVha, a validation-guided hypernetwork aggregation method for training a single global model under Non-IID federated learning. FedVha introduces a lightweight server-side hypernetwork that maps clients' static data characteristics and dynamic training feedback to aggregation weights. The server uses these weights to aggregate client updates and optimizes the hypernetwork by backpropagating the validation loss of the updated global model on a server-side validation set, aligning weight learning more directly with global generalization. FedVha further combines temperature-scaled softmax and distribution smoothing regularization to reduce excessive weight concentration. Experiments on multiple non-IID image classification tasks demonstrate the effectiveness of FedVha, especially in highly heterogeneous settings. In the highly heterogeneous CIFAR-10 + VGG16 setting with beta=0.1, FedVha outperforms the strongest baseline by 5.71 percentage points. Ablation studies and weight dynamics analysis support that validation feedback, client state modeling, and distribution smoothing improve performance and stabilize aggregation weights.

## Supported Experiments

- Datasets: CIFAR-10, MNIST, SVHN
- Models: VGG16, LeNet
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

## Notes

Datasets, checkpoints, logs, and generated figures are intentionally excluded from this repository. Put datasets outside the repo and pass their location with `--data_root`.

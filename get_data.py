import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)
TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINYIMAGENET_STD = (0.2302, 0.2265, 0.2262)
NUM_CLASSES = 10


@dataclass
class FederatedData:
    train_loaders: list
    server_val_loader: DataLoader
    server_val_eval_loader: DataLoader
    test_loader: DataLoader
    client_features: torch.Tensor
    client_class_counts: torch.Tensor
    num_classes: int
    client_sizes: list
    client_indices: dict
    server_val_indices: list


FederatedCIFAR10Data = FederatedData


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TinyImageNetValDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.transform = transform
        self.samples = []
        ann_path = os.path.join(root, "val", "val_annotations.txt")
        image_dir = os.path.join(root, "val", "images")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"Tiny-ImageNet validation annotations not found: {ann_path}"
            )
        with open(ann_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, class_id = parts[0], parts[1]
                if class_id in class_to_idx:
                    self.samples.append(
                        (os.path.join(image_dir, image_name), class_to_idx[class_id])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def split_server_validation(labels, samples_per_class, rng, num_classes=NUM_CLASSES):
    server_indices = []
    client_pool = []
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        if len(class_indices) <= samples_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(class_indices)} samples; "
                f"cannot reserve {samples_per_class}"
            )
        server_indices.extend(class_indices[:samples_per_class].tolist())
        client_pool.extend(class_indices[samples_per_class:].tolist())
    rng.shuffle(client_pool)
    return np.asarray(client_pool), server_indices


def build_dirichlet_partition(
    labels,
    pool_indices,
    num_clients,
    beta,
    min_client_samples,
    rng,
    num_classes=NUM_CLASSES,
    max_attempts=1000,
):
    average_size = len(pool_indices) / num_clients

    for _ in range(max_attempts):
        client_batches = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_indices = pool_indices[labels[pool_indices] == class_id].copy()
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.full(num_clients, beta))
            under_average = np.asarray(
                [len(indices) < average_size for indices in client_batches],
                dtype=np.float64,
            )
            proportions *= under_average
            if proportions.sum() == 0:
                proportions = np.full(num_clients, 1.0 / num_clients)
            else:
                proportions /= proportions.sum()

            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            class_splits = np.split(class_indices, split_points)
            for client_id, split in enumerate(class_splits):
                client_batches[client_id].extend(split.tolist())

        if min(map(len, client_batches)) >= min_client_samples:
            return {
                client_id: rng.permutation(indices).tolist()
                for client_id, indices in enumerate(client_batches)
            }

    raise RuntimeError(
        "Unable to create the requested Dirichlet partition after "
        f"{max_attempts} attempts. Reduce --min_client_samples or increase --beta."
    )


def compute_client_features(client_indices, labels, num_classes=NUM_CLASSES):
    total_samples = sum(len(indices) for indices in client_indices.values())
    features = []
    for client_id in range(len(client_indices)):
        indices = client_indices[client_id]
        client_labels = labels[indices]
        class_distribution = np.bincount(
            client_labels,
            minlength=num_classes,
        ).astype(np.float32)
        class_distribution /= len(indices)
        data_fraction = len(indices) / total_samples
        features.append(np.concatenate([[data_fraction], class_distribution]))

    feature_tensor = torch.tensor(np.asarray(features), dtype=torch.float32)
    mean = feature_tensor.mean(dim=0, keepdim=True)
    std = feature_tensor.std(dim=0, keepdim=True, unbiased=False)
    return (feature_tensor - mean) / (std + 1e-8)


def compute_client_class_counts(client_indices, labels, num_classes=NUM_CLASSES):
    counts = []
    for client_id in range(len(client_indices)):
        client_labels = labels[client_indices[client_id]]
        counts.append(
            np.bincount(
                client_labels,
                minlength=num_classes,
            ).astype(np.float32)
        )
    return torch.tensor(np.asarray(counts), dtype=torch.float32)


def make_loader(dataset, batch_size, shuffle, seed, args):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator,
        persistent_workers=False,
    )


def build_federated_cifar10(args):
    rng = np.random.default_rng(args.seed)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(
        args.data_root,
        train=True,
        download=args.download,
        transform=train_transform,
    )
    train_eval_dataset = datasets.CIFAR10(
        args.data_root,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        args.data_root,
        train=False,
        download=args.download,
        transform=eval_transform,
    )

    labels = np.asarray(train_dataset.targets)
    client_pool, server_val_indices = split_server_validation(
        labels,
        args.server_val_per_class,
        rng,
    )
    client_indices = build_dirichlet_partition(
        labels,
        client_pool,
        args.K,
        args.beta,
        args.min_client_samples,
        rng,
    )

    train_loaders = []
    for client_id in range(args.K):
        subset = Subset(train_dataset, client_indices[client_id])
        train_loaders.append(
            make_loader(
                subset,
                args.B,
                shuffle=True,
                seed=args.seed + client_id,
                args=args,
            )
        )

    server_subset = Subset(train_eval_dataset, server_val_indices)
    server_val_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=True,
        seed=args.seed + 10000,
        args=args,
    )
    server_val_eval_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=False,
        seed=args.seed + 20000,
        args=args,
    )
    test_loader = make_loader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        seed=args.seed + 30000,
        args=args,
    )

    return FederatedData(
        train_loaders=train_loaders,
        server_val_loader=server_val_loader,
        server_val_eval_loader=server_val_eval_loader,
        test_loader=test_loader,
        client_features=compute_client_features(client_indices, labels),
        client_class_counts=compute_client_class_counts(client_indices, labels),
        num_classes=NUM_CLASSES,
        client_sizes=[len(client_indices[i]) for i in range(args.K)],
        client_indices=client_indices,
        server_val_indices=server_val_indices,
    )


def build_federated_cifar100(args):
    rng = np.random.default_rng(args.seed)
    num_classes = 100

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    train_dataset = datasets.CIFAR100(
        args.data_root,
        train=True,
        download=args.download,
        transform=train_transform,
    )
    train_eval_dataset = datasets.CIFAR100(
        args.data_root,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR100(
        args.data_root,
        train=False,
        download=args.download,
        transform=eval_transform,
    )

    labels = np.asarray(train_dataset.targets)
    client_pool, server_val_indices = split_server_validation(
        labels,
        args.server_val_per_class,
        rng,
        num_classes=num_classes,
    )
    client_indices = build_dirichlet_partition(
        labels,
        client_pool,
        args.K,
        args.beta,
        args.min_client_samples,
        rng,
        num_classes=num_classes,
    )

    train_loaders = []
    for client_id in range(args.K):
        subset = Subset(train_dataset, client_indices[client_id])
        train_loaders.append(
            make_loader(
                subset,
                args.B,
                shuffle=True,
                seed=args.seed + client_id,
                args=args,
            )
        )

    server_subset = Subset(train_eval_dataset, server_val_indices)
    server_val_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=True,
        seed=args.seed + 10000,
        args=args,
    )
    server_val_eval_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=False,
        seed=args.seed + 20000,
        args=args,
    )
    test_loader = make_loader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        seed=args.seed + 30000,
        args=args,
    )

    return FederatedData(
        train_loaders=train_loaders,
        server_val_loader=server_val_loader,
        server_val_eval_loader=server_val_eval_loader,
        test_loader=test_loader,
        client_features=compute_client_features(
            client_indices,
            labels,
            num_classes=num_classes,
        ),
        client_class_counts=compute_client_class_counts(
            client_indices,
            labels,
            num_classes=num_classes,
        ),
        num_classes=num_classes,
        client_sizes=[len(client_indices[i]) for i in range(args.K)],
        client_indices=client_indices,
        server_val_indices=server_val_indices,
    )


def build_federated_svhn(args):
    rng = np.random.default_rng(args.seed)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )

    train_dataset = datasets.SVHN(
        args.data_root,
        split="train",
        download=args.download,
        transform=train_transform,
    )
    train_eval_dataset = datasets.SVHN(
        args.data_root,
        split="train",
        download=False,
        transform=eval_transform,
    )
    test_dataset = datasets.SVHN(
        args.data_root,
        split="test",
        download=args.download,
        transform=eval_transform,
    )

    labels = np.asarray(train_dataset.labels)
    client_pool, server_val_indices = split_server_validation(
        labels,
        args.server_val_per_class,
        rng,
    )
    client_indices = build_dirichlet_partition(
        labels,
        client_pool,
        args.K,
        args.beta,
        args.min_client_samples,
        rng,
    )

    train_loaders = []
    for client_id in range(args.K):
        subset = Subset(train_dataset, client_indices[client_id])
        train_loaders.append(
            make_loader(
                subset,
                args.B,
                shuffle=True,
                seed=args.seed + client_id,
                args=args,
            )
        )

    server_subset = Subset(train_eval_dataset, server_val_indices)
    server_val_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=True,
        seed=args.seed + 10000,
        args=args,
    )
    server_val_eval_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=False,
        seed=args.seed + 20000,
        args=args,
    )
    test_loader = make_loader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        seed=args.seed + 30000,
        args=args,
    )

    return FederatedData(
        train_loaders=train_loaders,
        server_val_loader=server_val_loader,
        server_val_eval_loader=server_val_eval_loader,
        test_loader=test_loader,
        client_features=compute_client_features(client_indices, labels),
        client_class_counts=compute_client_class_counts(client_indices, labels),
        num_classes=NUM_CLASSES,
        client_sizes=[len(client_indices[i]) for i in range(args.K)],
        client_indices=client_indices,
        server_val_indices=server_val_indices,
    )


def resolve_tinyimagenet_root(data_root):
    if os.path.isdir(os.path.join(data_root, "train")):
        return data_root
    nested_root = os.path.join(data_root, "tiny-imagenet-200")
    if os.path.isdir(os.path.join(nested_root, "train")):
        return nested_root
    raise FileNotFoundError(
        "Tiny-ImageNet root not found. Pass --data_root as the "
        "tiny-imagenet-200 directory or its parent directory."
    )


def build_federated_mnist(args):
    rng = np.random.default_rng(args.seed)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )

    train_dataset = datasets.MNIST(
        args.data_root,
        train=True,
        download=args.download,
        transform=transform,
    )
    train_eval_dataset = datasets.MNIST(
        args.data_root,
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        args.data_root,
        train=False,
        download=args.download,
        transform=transform,
    )

    labels = np.asarray(train_dataset.targets)
    client_pool, server_val_indices = split_server_validation(
        labels,
        args.server_val_per_class,
        rng,
    )
    client_indices = build_dirichlet_partition(
        labels,
        client_pool,
        args.K,
        args.beta,
        args.min_client_samples,
        rng,
    )

    train_loaders = []
    for client_id in range(args.K):
        subset = Subset(train_dataset, client_indices[client_id])
        train_loaders.append(
            make_loader(
                subset,
                args.B,
                shuffle=True,
                seed=args.seed + client_id,
                args=args,
            )
        )

    server_subset = Subset(train_eval_dataset, server_val_indices)
    server_val_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=True,
        seed=args.seed + 10000,
        args=args,
    )
    server_val_eval_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=False,
        seed=args.seed + 20000,
        args=args,
    )
    test_loader = make_loader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        seed=args.seed + 30000,
        args=args,
    )

    return FederatedData(
        train_loaders=train_loaders,
        server_val_loader=server_val_loader,
        server_val_eval_loader=server_val_eval_loader,
        test_loader=test_loader,
        client_features=compute_client_features(client_indices, labels),
        client_class_counts=compute_client_class_counts(client_indices, labels),
        num_classes=NUM_CLASSES,
        client_sizes=[len(client_indices[i]) for i in range(args.K)],
        client_indices=client_indices,
        server_val_indices=server_val_indices,
    )


def build_federated_tinyimagenet(args):
    rng = np.random.default_rng(args.seed)
    root = resolve_tinyimagenet_root(args.data_root)
    train_dir = os.path.join(root, "train")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_eval_dataset = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_dataset = TinyImageNetValDataset(
        root,
        train_dataset.class_to_idx,
        transform=eval_transform,
    )

    num_classes = len(train_dataset.classes)
    labels = np.asarray([target for _, target in train_dataset.samples])
    client_pool, server_val_indices = split_server_validation(
        labels,
        args.server_val_per_class,
        rng,
        num_classes=num_classes,
    )
    client_indices = build_dirichlet_partition(
        labels,
        client_pool,
        args.K,
        args.beta,
        args.min_client_samples,
        rng,
        num_classes=num_classes,
    )

    train_loaders = []
    for client_id in range(args.K):
        subset = Subset(train_dataset, client_indices[client_id])
        train_loaders.append(
            make_loader(
                subset,
                args.B,
                shuffle=True,
                seed=args.seed + client_id,
                args=args,
            )
        )

    server_subset = Subset(train_eval_dataset, server_val_indices)
    server_val_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=True,
        seed=args.seed + 10000,
        args=args,
    )
    server_val_eval_loader = make_loader(
        server_subset,
        args.val_batch_size,
        shuffle=False,
        seed=args.seed + 20000,
        args=args,
    )
    test_loader = make_loader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        seed=args.seed + 30000,
        args=args,
    )

    return FederatedData(
        train_loaders=train_loaders,
        server_val_loader=server_val_loader,
        server_val_eval_loader=server_val_eval_loader,
        test_loader=test_loader,
        client_features=compute_client_features(
            client_indices,
            labels,
            num_classes=num_classes,
        ),
        client_class_counts=compute_client_class_counts(
            client_indices,
            labels,
            num_classes=num_classes,
        ),
        num_classes=num_classes,
        client_sizes=[len(client_indices[i]) for i in range(args.K)],
        client_indices=client_indices,
        server_val_indices=server_val_indices,
    )

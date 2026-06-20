import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class CIFAR10Truncated(Dataset):
    """Backward-compatible indexed CIFAR-10 dataset.

    The current training pipeline uses one shared torchvision dataset with
    ``Subset`` objects, so this class is retained only for external scripts
    that imported the original ``CIFAR10_truncated`` name.
    """

    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        dataset = CIFAR10(
            root,
            train=train,
            download=download,
        )
        data = dataset.data
        targets = np.asarray(dataset.targets)
        if dataidxs is not None:
            data = data[dataidxs]
            targets = targets[dataidxs]

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.data)


CIFAR10_truncated = CIFAR10Truncated

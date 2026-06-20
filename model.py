import torch.nn as nn


VGG_CONFIGS = {
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, name="VGG16", num_classes=10):
        super().__init__()
        self.features = self._make_layers(VGG_CONFIGS[name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    @staticmethod
    def _make_layers(config):
        layers = []
        in_channels = 3
        for value in config:
            if value == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            layers.extend(
                [
                    nn.Conv2d(in_channels, value, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(value),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = value
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(12 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(name, dataset="cifar10"):
    if dataset == "mnist":
        if name != "LeNet":
            raise ValueError("MNIST currently supports LeNet")
        return LeNet(num_classes=10)
    if dataset in {"cifar10", "svhn"}:
        if name != "VGG16":
            raise ValueError("CIFAR-10 and SVHN currently support VGG16")
        return VGG(name="VGG16", num_classes=10)
    raise ValueError(f"Unsupported dataset: {dataset}")

import torch.nn as nn


class AggregationHyperNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)

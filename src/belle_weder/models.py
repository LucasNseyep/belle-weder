import torch.nn as nn


class TopBlock(nn.Module):
    """
    Custom classification head from CloudDenseNet (Li et al., Sensors 2023).

    Replaces DenseNet121's single linear classifier with a 2-layer MLP:
      BN → Dropout → Linear(in, hidden) → ReLU → BN → Dropout → Linear(hidden, n_classes)

    Weights are initialised with LeCun uniform distribution, as specified in the paper.
    """

    def __init__(self, in_features: int, n_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim, n_classes),
        )
        self._lecun_init()

    def _lecun_init(self):
        for m in self.block:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x)

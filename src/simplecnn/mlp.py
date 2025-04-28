from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(16 * 224 * 224, 2),
        )

    def forward(self, x):
        flatt = x.view(x.size(0), -1)
        res = self.mlp(flatt)
        return res

# Python imports.
import pdb

# PyTorch imports.
import torch
import torch.nn as nn


class DenseOneClassModel(nn.Module):
    """ Dense SVDD Model for one-class classification. """

    def __init__(self, input_dim, h1=64, h2=32, rep_dim=2, device=torch.device("cuda")):
        super(DenseOneClassModel, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.rep_dim = rep_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, h1, bias=False),
            nn.LeakyReLU(),
            nn.Linear(h1, h2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(h2, rep_dim, bias=False)
        )

        self.to(device)

    def forward(self, x):
        return self.model(x)

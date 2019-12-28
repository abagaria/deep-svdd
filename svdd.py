import pdb
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data.toy_dataset import ToyDataset
from model import DenseOneClassModel
from utils import *


class SVDD(object):
    """A class for the Deep SVDD method.

        Attributes:
            nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
            R: Hypersphere radius R.
            c: Hypersphere center c.
            net: The neural network \phi.
        """

    def __init__(self, in_dim, nu=0.01, device=torch.device("cuda"), tensor_log=False):
        """Initialize DeepSVDD with one of the two objectives and hyperparameter nu."""
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."

        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.device = device
        self.model = DenseOneClassModel(input_dim=in_dim, device=device)

        # Logging
        self.tensor_log = tensor_log
        if tensor_log:
            self.writer = SummaryWriter()

    def pretrain(self):
        pass

    def fit(self, loader, n_epochs=150):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-6)

        n_iterations = 0

        if self.c is None:
            self.c = self.init_center_c(self.model, loader)
            print("Determined c = {}".format(self.c))

        self.model.train()

        for epoch in range(n_epochs):
            for data in tqdm(loader):

                # Move the data to the GPU
                data = data.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(data)

                distance = torch.sum((outputs - self.c) ** 2, dim=1)
                scores = distance - (self.R ** 2)
                loss = (self.R ** 2) + ((1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores)))

                loss.backward()
                optimizer.step()

                if epoch > 5:
                    self.R = torch.tensor(get_radius(distance, self.nu), device=self.device)

                # Logging
                n_iterations = n_iterations + 1

                if self.tensor_log:
                    self.writer.add_scalar("Loss", loss.item(), n_iterations)
                    self.writer.add_scalar("Radius", self.R, n_iterations)
                    self.writer.add_scalar("Average-Scores", scores.mean(), n_iterations)

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
        distance = torch.sum((output - self.c) ** 2, dim=1)
        score = distance - (self.R ** 2)
        inliers = score < 0
        return inliers

    def test(self, loader):
        self.model.eval()
        predictions = []

        for data in tqdm(loader, desc="Testing Epoch"):
            data = data.to(self.device)
            label = self.predict(data)
            predictions.append((data, label))

        return predictions

    # -----------------------------------------------
    # Helper functions
    # -----------------------------------------------

    def init_center_c(self, model, data, eps=0.1):
        n_samples = 0
        c = torch.zeros(model.rep_dim, device=self.device)

        model.eval()
        with torch.no_grad():
            for mini_batch in data:
                mini_batch = mini_batch.to(self.device)
                outputs = model(mini_batch)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = +eps

        return c


def get_radius(dist, nu):
    """ Optimally solve for radius R via the (1-nu)-quantile of distances. """
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


if __name__ == "__main__":
    n_dims = 2
    train_set = ToyDataset(n_dims, "train")
    train_set = DataLoader(train_set, shuffle=True, batch_size=16)
    svdd = SVDD(n_dims, device="cpu")

    test_set = ToyDataset(n_dims, "test")
    test_set = DataLoader(test_set, shuffle=True, batch_size=8)

    svdd.fit(train_set)

    train_predictions = svdd.test(train_set)
    test_predictions = svdd.test(test_set)

    print("[Training Data] Inlier percentage: ", get_inlier_percentage(train_predictions))
    print("[Test Data] Inlier percentage: ", get_inlier_percentage(test_predictions))

    plot_predictions(train_predictions, test_predictions)

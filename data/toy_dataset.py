import torch
import numpy as np
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    """ 2D points dataset. """
    
    def __init__(self, mode):
        assert mode in ("train", "test"), mode
        super(ToyDataset, self).__init__()

        self.mode = mode
        self.data = self.generate_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def generate_data(self):
        if self.mode == "train":
            training_data = np.random.normal((0., 0.), 1., (1000, 2))
            return torch.from_numpy(training_data).float()
        else:
            testing_data = np.random.normal((4., 0.), 1., (1000, 2))
            return torch.from_numpy(testing_data).float()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_set = ToyDataset(mode="train")
    test_set = ToyDataset(mode="test")
    plt.scatter(train_set.data[:, 0], train_set.data[:, 1], label="Training Data")
    plt.scatter(test_set.data[:, 0], test_set.data[:, 1], label="Test Data")
    plt.legend()
    plt.show()

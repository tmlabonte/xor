"""Generator for a dataset with XOR labels and spurious correlation."""

from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset

@dataclass
class DatasetConfig:
    """Configuration for Dataset class."""
    name: str # Dataset name
    d: int = 50 # Data dimension
    n: int = 1000 # Number of data
    m: int = 32 # Batch size
    spurious: bool = False # Whether to generate data with spurious correlation
    lmbda: float | None = None # Correlation strength; lower is stronger
    seed: int = 42 # Random seed for dataset generation

# pylint: disable=abstract-method
class Dataset(IterableDataset):
    """Generator for a dataset with XOR labels and spurious correlation."""
    def __init__(self, config: DatasetConfig) -> None:
        """Initializes a dataset."""
        super().__init__()
        print(config)
        self.name = config.name
        self.config = config

    def generate_data(
        self, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates a single batch of data and targets."""
        # Generate i.i.d. data on {±1}^d with target XOR(x_1, x_2).
        randints = torch.randint(
            0, 2, (self.config.m, self.config.d), generator=rng)
        data = (randints * 2 - 1).float()
        targets = -(data[:, 0] * data[:, 1])

        # Design spurious correlation such that x_3 = y with probability 1 - λ.
        if self.config.spurious:
            data[:, 2] = targets
            probs = torch.full((self.config.m,), self.config.lmbda)
            flip_mask = torch.bernoulli(probs, generator=rng).bool()
            data[flip_mask, 2] *= -1

        return data, targets

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates batches of data until total_n batches are reached."""
        # Drop last batch if n is not a multiple of m.
        real_n = self.config.n - (self.config.n % self.config.m)
        rng = torch.Generator().manual_seed(self.config.seed)

        count = 0
        while count < real_n:
            # Generate data on-demand to save on memory
            data, targets = self.generate_data(rng)
            yield data, targets
            count += self.config.m

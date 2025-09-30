"""Train and test logic for XOR experiments."""

# pylint: disable=invalid-name

from dataclasses import dataclass
import math
import os

import coolname
from safetensors.torch import save_file
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xor.dataset import Dataset
from xor.metrics import Metrics
from xor.model import Model
from xor.util import dump_dataclass_to_yaml

@dataclass
class ExperimentConfig:
    """Configuration for Experiment class."""
    name: str # Experiment name
    loss: str # Loss function
    hybrid: bool = False # Whether to use hybrid step
    eta: float = 0.01 # Learning rate
    exp_dir: str = "output/tmp" # Experiment-level output directory

class Experiment:
    """Class for training and testing of a model on given datasets."""
    def __init__(
        self,
        train_dataset: Dataset,
        test_datasets: list[Dataset],
        model: Model,
        config: ExperimentConfig,
    ) -> None:
        """Initializes an experiment."""
        self.config = config
        self._generate_name_and_dir()
        print(config)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)

        # Instantiate train and test dataloaders.
        num_workers = 1 # IMPORTANT
        pin_memory = device == "cuda"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.test_loaders = [
            DataLoader(
                test_dataset,
                batch_size=None,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            for test_dataset in test_datasets
        ]

        # Instantiate loss function and optimizer.
        self.loss_fn = {
            "lp": self._lp_loss,
            "l0": self._l0_loss,
        }[config.loss]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.eta)

        # Instantiate metrics object with number of train steps.
        # This allocates memory for all the metrics tensors.
        train_steps = math.ceil(train_dataset.config.n / train_dataset.config.m)
        self.metrics = Metrics.create(
            vector_names=self.model.vector_fns.keys(),
            p=self.model.config.p,
            train_steps=train_steps + 1, # Add one to log initial norms
        )

        self._dump_configs()

    def _generate_name_and_dir(self) -> None:
        """Generates a unique name and creates output dir."""
        name = self.config.name.replace("_", "-")
        slug = coolname.generate_slug(2)
        self.config.name = f"{name}_{slug}"
        self.config.exp_dir = os.path.join(
            os.path.dirname(self.config.exp_dir), self.config.name)
        os.makedirs(self.config.exp_dir, exist_ok=True)

    def _dump_configs(self) -> None:
        """Dumps experiment, dataset, and model configs to output dir."""
        config_dir = os.path.join(self.config.exp_dir, "config")
        os.makedirs(config_dir, exist_ok=True)

        dump_dataclass_to_yaml(
            self.config,
            os.path.join(config_dir, "experiment.yaml"),
        )

        dump_dataclass_to_yaml(
            self.train_loader.dataset.config,
            os.path.join(config_dir, "train_dataset.yaml"),
        )
        for j, test_loader in enumerate(self.test_loaders):
            dump_dataclass_to_yaml(
                test_loader.dataset.config,
                os.path.join(config_dir, f"test_dataset_{j}.yaml"),
            )
        dump_dataclass_to_yaml(
            self.model.config,
            os.path.join(config_dir, "model.yaml"),
        )

    @staticmethod
    def _lp_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns logistic loss of logits against y."""
        return (-2 * torch.log(1 / (1 + torch.exp(-y * logits)))).mean()

    @staticmethod
    def _l0_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns first-order logistic loss of logits against y."""
        return (-2 * math.log(0.5) - y * logits).mean()

    def step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A single step of loss computation and optimization."""
        self.optimizer.zero_grad()
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        loss.backward()
        self.optimizer.step()

        return loss, logits

    def hybrid_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps with Lp loss for w_sp and L0 loss for everything else."""
        self.optimizer.zero_grad()
        logits = self.model(X)

        loss = self._l0_loss(logits, y)
        loss.backward(retain_graph=True)

        # Compute Lp grads using autograd.
        sp_loss = self._lp_loss(logits, y)
        sp_grad = torch.autograd.grad(
            sp_loss,
            self.model.hidden.weight,
        )[0]

        # Copy Lp grads to w_sp.
        with torch.no_grad():
            self.model.hidden.weight.grad[:, 2] = sp_grad[:, 2]

        self.optimizer.step()

        return loss, logits

    def train(self) -> tuple[Model, Metrics]:
        """Trains on the train dataloader."""
        print("Training...")
        self.metrics.update_norms(0, self.model.norms())

        self.model.train()
        name = self.train_loader.dataset.name
        for step, (X, y) in enumerate(tqdm(self.train_loader, desc=name)):
            X, y = X.to(self.model.device), y.to(self.model.device)

            if self.config.hybrid:
                _, logits = self.hybrid_step(X, y)
            else:
                _, logits = self.step(X, y)

            self.metrics.update_norms(step + 1, self.model.norms())
            self.metrics.update_grads(step + 1, self.model.grads())
            self.metrics.update_margins(
                step + 1,
                self.model.margins(X, y, logits=logits),
            )

        save_file(
            self.model.state_dict(),
            os.path.join(self.config.exp_dir, "model.safetensors"),
        )
        dump_dataclass_to_yaml(
            self.metrics,
            os.path.join(self.config.exp_dir, "metrics.yaml")
        )

        return self.model, self.metrics

    @torch.no_grad()
    def _test(self, loader: DataLoader) -> None:
        """Evaluates on a single dataloader."""
        name = loader.dataset.name
        correct = 0
        total = 0
        for X, y in tqdm(loader, desc=name):
            X, y = X.to(self.model.device), y.to(self.model.device)
            preds = (self.model(X) > 0).long() * 2 - 1 # Send to {-1, 1}
            correct += (preds == y).sum().item()
            total += y.shape[0]
        self.metrics.accuracies[name] = correct / total

    @torch.no_grad()
    def test(self) -> Metrics:
        """Evaluates on all test dataloaders."""
        print("Testing...")
        self.model.eval()
        for loader in self.test_loaders:
            self._test(loader)

        dump_dataclass_to_yaml(
            self.metrics,
            os.path.join(self.config.exp_dir, "metrics.yaml")
        )

        return self.metrics

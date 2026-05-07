"""Train and test logic for XOR experiments."""

# pylint: disable=invalid-name # Allow X, y naming.

from dataclasses import dataclass
import math
import os

import coolname
from loguru import logger
import safetensors.torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from xor.constants import CONFIG_DIR, OUT_DIR, TMP_DIR
from xor.dataset import Dataset
from xor.metrics import Metrics
from xor.model import Model
from xor.util import dump_dataclass_to_yaml


@dataclass
class ExperimentConfig:
    """Configuration for Experiment class."""

    name: str  # Experiment name
    loss: str  # Loss function
    hybrid: bool = False  # Whether to use hybrid step
    eta: float = 0.01  # Learning rate
    exp_dir: str = os.path.join(OUT_DIR, TMP_DIR)  # Experiment-level output dir
    margin: str = "mean"  # Whether to track the mean or min margin.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Training device
    save_metrics: bool = False # Whether to save metrics; memory intensive


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
        logger.info(config)

        self.model = model.to(self.config.device)

        # Instantiate train and test dataloaders.
        num_workers = 1  # IMPORTANT
        pin_memory = self.config.device == "cuda"
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

        # Instantiate margin tracking function.
        self.margin_fn = {
            "mean": torch.mean,
            "min": torch.min,
        }[config.margin]

        # Instantiate metrics object with number of train steps.
        # This allocates memory for all the metrics tensors.
        self.metrics = Metrics.create(
            keys=self.model.vector_fns.keys(),
            p=self.model.config.p,
            d=self.model.config.d,
            train_steps=len(train_dataset) + 1,  # Add one to log initial norms
        )

        self._dump_configs()

    def _generate_name_and_dir(self) -> None:
        """Generates a unique name and creates output dir."""
        name = self.config.name.replace("_", "-")
        slug = coolname.generate_slug(2)
        self.config.name = f"{name}_{slug}"
        self.config.exp_dir = os.path.join(
            os.path.dirname(self.config.exp_dir), self.config.name
        )
        os.makedirs(self.config.exp_dir, exist_ok=True)

    def _dump_configs(self) -> None:
        """Dumps experiment, dataset, and model configs to output dir."""
        config_dir = os.path.join(self.config.exp_dir, CONFIG_DIR)
        os.makedirs(config_dir, exist_ok=True)

        configs = {
            "experiment": self.config,
            "train_dataset": self.train_loader.dataset.config,
            "model": self.model.config,
        }
        for j, test_loader in enumerate(self.test_loaders):
            configs[f"test_dataset_{j}"] = test_loader.dataset.config

        for name, config in configs.items():
            dump_dataclass_to_yaml(
                config,
                os.path.join(config_dir, f"{name}.yaml"),
            )
        logger.info(f"Configs saved to {config_dir}")

    @staticmethod
    def _lp_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns logistic loss of logits against y."""
        return (-2 * F.logsigmoid(y * logits)).mean()  # pylint: disable=not-callable

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
        logger.info("Training...")
        self.metrics.record(0, norms=self.model.norms(), weights=self.model.weights())

        self.model.train()
        name = self.train_loader.dataset.name
        for step, (X, y) in enumerate(tqdm(self.train_loader, desc=name)):
            X, y = X.to(self.config.device), y.to(self.config.device)

            if self.config.hybrid:
                loss, logits = self.hybrid_step(X, y)
            else:
                loss, logits = self.step(X, y)

            # Compute theoretical predictions for model weights.
            """
            self.model.step_preds(
                self.config.loss,
                self.config.eta,
                self.train_loader.dataset.config.lmbda,
                X,
                y,
                logits=logits,
            )
            """

            # Record relevant step metrics in the self.metrics object.
            self.metrics.record(
                step + 1,
                losses={"train loss": loss.item()},
                norms=self.model.norms(),
                weights=self.model.weights(),
                grads=self.model.grads(),
                margins=self.model.margins(
                    X, y, logits=logits, collate_fn=self.margin_fn
                ),
                # preds=self.model.preds_norms(),
            )

        # Save model weights to disk.
        weights_path = os.path.join(self.config.exp_dir, "model.safetensors")
        safetensors.torch.save_file(self.model.state_dict(), weights_path)
        logger.info(f"Model weights saved to {weights_path}")

        # Saves train metrics to disk.
        if self.config.save_metrics:
            metrics_path = os.path.join(self.config.exp_dir, "metrics.yaml")
            dump_dataclass_to_yaml(self.metrics, metrics_path)
            logger.info(f"Train metrics saved to {metrics_path}")

        return self.model, self.metrics

    @torch.no_grad()
    def _test(self, loader: DataLoader) -> None:
        """Evaluates on a single dataloader."""
        name = loader.dataset.name
        correct = 0
        total = 0
        for X, y in tqdm(loader, desc=name):
            X, y = X.to(self.config.device), y.to(self.config.device)
            preds = (self.model(X) > 0).long() * 2 - 1  # Send to {-1, 1}
            correct += (preds == y).sum().item()
            total += y.shape[0]
        self.metrics.accuracies[name] = correct / total

    @torch.no_grad()
    def test(self) -> Metrics:
        """Evaluates on all test dataloaders."""
        logger.info("Testing...")
        self.model.eval()
        for loader in self.test_loaders:
            self._test(loader)

        # Save test metrics to disk.
        if self.config.save_metrics:
            metrics_path = os.path.join(self.config.exp_dir, "metrics.yaml")
            dump_dataclass_to_yaml(self.metrics, metrics_path)
            logger.info(f"Test metrics saved to {metrics_path}")

        return self.metrics

from dataclasses import dataclass
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xor.dataset import Dataset
from xor.metrics import Metrics
from xor.model import Model

@dataclass
class ExperimentConfig:
    name: str # Experiment name
    loss: str # Loss function
    hybrid: bool = False # Whether to use hybrid step
    eta: float = 0.01 # Learning rate

class Experiment:
    def __init__(
        self,
        train_dataset: Dataset,
        test_datasets: list[Dataset],
        model: Model,
        config: ExperimentConfig,
    ) -> None:
        print(config)
        self.name = config.name
        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

        num_workers = 1
        pin_memory = self.device == "cuda"
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

        self.loss_fns = {
            "lp": self._lp_loss, # l_\rho loss from Glasgow (2024)
            "l0": self._l0_loss, # First order Taylor approx of logistic loss
        }
        self.loss_fn = self.loss_fns[config.loss]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.eta)

        train_steps = math.ceil(train_dataset.config.n / train_dataset.config.m)
        self.metrics = Metrics.create(
            vector_names=self.model.vector_fns.keys(),
            p=self.model.config.p,
            train_steps=train_steps + 1, # Add one to log initial norms
        )

    @staticmethod
    def _lp_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (-2 * torch.log(1 / (1 + torch.exp(-y * logits)))).mean()

    @staticmethod
    def _l0_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (-2 * math.log(0.5) - y * logits).mean()

    def step(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(X), y)
        loss.backward()
        self.optimizer.step()

    def hybrid_step(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        preds = self.model(X)

        loss = self.loss_fns["l0"](preds, y)
        # loss = self.loss_fns["lp"](preds, y)
        loss.backward(retain_graph=True)

        sp_loss = self.loss_fns["lp"](preds, y)
        # sp_loss = self.loss_fns["l0"](preds, y)
        sp_grad = torch.autograd.grad(
            sp_loss,
            self.model.hidden.weight,
        )[0]

        with torch.no_grad():
            self.model.hidden.weight.grad[:, 2] = sp_grad[:, 2]
            # self.model.hidden.weight.grad[:, 2:] = sp_grad[:, 2:]

        self.optimizer.step()

    def train(self) -> tuple[Model, Metrics]:
        print("Training...")
        self.metrics.update_norms(0, self.model.norms())

        self.model.train()
        name = self.train_loader.dataset.name
        for step, (X, y) in enumerate(tqdm(self.train_loader, desc=name)):
            X, y = X.to(self.device), y.to(self.device)

            if self.config.hybrid:
                self.hybrid_step(X, y)
            else:
                self.step(X, y)

            self.metrics.update_norms(step + 1, self.model.norms())
            self.metrics.update_grads(step + 1, self.model.grads())
        return self.model, self.metrics

    @torch.no_grad()
    def _test(self, loader: DataLoader) -> None:
        name = loader.dataset.name
        correct = 0
        total = 0
        for X, y in tqdm(loader, desc=name):
            X, y = X.to(self.device), y.to(self.device)
            preds = (self.model(X) > 0).long() * 2 - 1
            correct += (preds == y).sum().item()
            total += y.shape[0]
        self.metrics.accuracies[name] = correct / total

    @torch.no_grad()
    def test(self) -> Metrics:
        print("Testing...")
        self.model.eval()
        for loader in self.test_loaders:
            self._test(loader)

        return self.metrics

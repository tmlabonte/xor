"""Single-hidden-layer neural network for XOR experiments."""

# pylint: disable=invalid-name

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

@dataclass
class ModelConfig:
    """Configuration for the Model class."""
    d: int = 50 # Data dimension
    p: int = 10 # Hidden dimension
    spurious: bool = False # Whether w_sp should be tracked
    theta: float = 1 # Initialization scale

class Model(nn.Module):
    """Class for a single-hidden-layer neural network with metric tracking."""
    def __init__(self, config: ModelConfig):
        """Initializes a model."""
        super().__init__()

        print(config)
        self.config = config

        # Instantiates architecture and initializes as uniform on the sphere.
        self.hidden = nn.Linear(config.d, config.p, bias=False)
        self.output = nn.Linear(config.p, 1, bias=False)
        self.activation = nn.ReLU()
        self.initialize_parameters()

        # Initialize mu_1 and mu_2 as torch constants.
        # Register constants as buffers to automatically move to self.device.
        mu_1 = torch.tensor([1.0, -1.0])
        mu_2 = torch.tensor([1.0, 1.0])
        self.register_buffer("mu_1", mu_1)
        self.register_buffer("mu_2", mu_2)

        # Set vector names and methods for tracking norms.
        self.vector_fns = {
            "w_sig": self.w_sigs,
            "w_opp": self.w_opps,
            "w_sp": self.w_sps,
            "w_perp": self.w_perps,
        }
        if not self.config.spurious:
            self.vector_fns.pop("w_sp")

    @staticmethod
    @torch.no_grad()
    def linear_like(
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        grad: torch.Tensor | None = None,
        bias_grad: torch.Tensor | None = None,
    ) -> nn.Linear:
        """Instantiates a new Linear layer from given parameters."""
        linear = nn.Linear(
            weight.shape[1], weight.shape[0], bias=bias is not None)
        linear.weight.copy_(weight)

        if bias is not None:
            linear.bias.copy_(bias)
        if grad is not None:
            linear.weight.grad = torch.zeros_like(linear.weight)
            linear.weight.grad.copy_(grad)
        if bias_grad is not None:
            linear.bias.grad = torch.zeros_like(linear.bias)
            linear.bias.grad.copy_(bias_grad)

        return linear

    def neurons(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list of model neuron weights."""
        return list(zip(self.output.weight[0], self.hidden.weight))

    @torch.no_grad()
    def initialize_parameters(self) -> None:
        """Initializes parameters as uniform on the sphere."""
        # Initialize a_j = ε_j θ where ε_j ~ Unif({±1}).
        sign = torch.sign(torch.randn((self.config.p)))
        self.output.weight[0].copy_(sign * self.config.theta)

        # Initialize w_j ~ Unif(S^{d-1}(θ)).
        w = torch.randn_like(self.hidden.weight)
        scaled_w = self.config.theta * (w / w.norm(dim=1, keepdim=True))
        self.hidden.weight.copy_(scaled_w)

    @torch.no_grad()
    def w_sigs(self) -> nn.Linear:
        """Returns projection of each neuron onto the signal component."""
        # Get mu_sig for each neuron depending on sign(a).
        mu_sigs = torch.where(
            self.output.weight[0, :, None] >= 0, self.mu_1, self.mu_2)

        # Project each w onto mu_sig. Note that 0.5 = ||mu_sig||^{-2}.
        dot = mu_sigs * self.hidden.weight[:, :2]
        dot = dot.sum(dim=1, keepdim=True)
        weight = 0.5 * dot * mu_sigs

        # Project each grad onto mu_sig.
        grad = None
        if self.hidden.weight.grad is not None:
            dot = mu_sigs * self.hidden.weight.grad[:, :2]
            dot = dot.sum(dim=1, keepdim=True)
            grad = 0.5 * dot * mu_sigs

        # Return a new Linear layer.
        return Model.linear_like(weight=weight, grad=grad)

    @torch.no_grad()
    def w_opps(self) -> nn.Linear:
        """Returns projection of each neuron onto the opposite component."""
        # Get mu_opp for each neuron depending on sign(a).
        mu_opps = torch.where(
            self.output.weight[0, :, None] >= 0, self.mu_2, self.mu_1)

        # Project each w onto mu_opp. Note that 0.5 = ||mu_opp||^{-2}.
        dot = mu_opps * self.hidden.weight[:, :2]
        dot = dot.sum(dim=1, keepdim=True)
        weight = 0.5 * dot * mu_opps

        # Project each grad onto mu_opp.
        grad = None
        if self.hidden.weight.grad is not None:
            dot = mu_opps * self.hidden.weight.grad[:, :2]
            dot = dot.sum(dim=1, keepdim=True)
            grad = 0.5 * dot * mu_opps

        # Return a new Linear layer.
        return Model.linear_like(weight=weight, grad=grad)

    @torch.no_grad()
    def w_sps(self) -> nn.Linear:
        """Returns projection of each neuron onto the spurious component."""
        if self.config.spurious:
            weight = self.hidden.weight[:, 2:3]
            grad = None
            if self.hidden.weight.grad is not None:
                grad = self.hidden.weight.grad[:, 2:3]
            return Model.linear_like(weight=weight, grad=grad)

        raise ValueError(
            f"Please set config.spurious to track w_sp."
            f" Current value: {self.config.spurious}."
        )

    @torch.no_grad()
    def w_perps(self) -> nn.Linear:
        """Returns projection of each neuron onto the perp component."""
        j = 3 if self.config.spurious else 2
        weight = self.hidden.weight[:, j:]
        grad = None
        if self.hidden.weight.grad is not None:
            grad = self.hidden.weight.grad[:, j:]
        return Model.linear_like(weight=weight, grad=grad)

    @torch.no_grad()
    def norms(self) -> dict[str, np.ndarray]:
        """Returns the norm of every weight across vector_fns."""
        return {
            vector_name: vector_fn().weight.norm(dim=1).cpu().numpy()
            for vector_name, vector_fn in self.vector_fns.items()
        }

    @torch.no_grad()
    def grads(self) -> dict[str, np.ndarray]:
        """Returns the grad of every weight across vector_fns."""
        return {
            vector_name: vector_fn().weight.grad.mean(dim=1).cpu().numpy()
            for vector_name, vector_fn in self.vector_fns.items()
        }

    @torch.no_grad()
    def margins(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        logits: torch.Tensor | None = None,
        collate_fn: Callable[[torch.Tensor], float] = torch.max,
    ) -> dict[str, float]:
        """Returns the margins of the neural network on a batch.

        Collate_fn can be something like torch.mean or torch.max depending on
        whether the average or worst-case margin is desired.
        """

        # Partitions batch into data with/without the spurious correlation.
        sp = X[:, 2] == y
        no_sp = X[:, 2] != y

        if logits is None:
            logits = self(X)

        return {
            "sp": collate_fn(y[sp] * logits[sp]).item(),
            "no_sp": collate_fn(y[no_sp] * logits[no_sp]).item(),
        }

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Computes a forward pass through the model."""
        return self.output(self.activation(self.hidden(X))).squeeze()

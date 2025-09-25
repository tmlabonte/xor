from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn

@dataclass
class ModelConfig:
    d: int = 50 # Data dimension
    p: int = 10 # Hidden dimension
    spurious: bool = False # Whether w_sp should be tracked
    theta: float = 1 # Initialization scale

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        print(config)
        self.config = config
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
    def linear_like(weight, bias=None, grad=None, bias_grad=None) -> nn.Linear:
        # Instantiate a new Linear layer from given parameters.
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
        return list(zip(self.output.weight[0], self.hidden.weight))

    @torch.no_grad()
    def initialize_parameters(self) -> None:
        # Initialize a_j = ε_j θ where ε_j ~ Unif({±1}).
        sign = torch.sign(torch.randn((self.config.p)))
        self.output.weight[0].copy_(sign * self.config.theta)

        # Initialize w_j ~ Unif(S^{d-1}(θ)).
        w = torch.randn_like(self.hidden.weight)
        scaled_w = self.config.theta * (w / w.norm(dim=1, keepdim=True))
        self.hidden.weight.copy_(scaled_w)

    @torch.no_grad()
    def w_sigs(self) -> nn.Linear:
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
        if self.config.spurious:
            weight = self.hidden.weight[:, 2:3]
            grad = None
            if self.hidden.weight.grad is not None:
                grad = self.hidden.weight.grad[:, 2:3]
            return Model.linear_like(weight=weight, grad=grad)
        else:
            raise ValueError(
                f"Please set config.spurious to track w_sp."
                f" Current value: {self.config.spurious}."
            )

    @torch.no_grad()
    def w_perps(self) -> nn.Linear:
        j = 3 if self.config.spurious else 2
        weight = self.hidden.weight[:, j:]
        grad = None
        if self.hidden.weight.grad is not None:
            grad = self.hidden.weight.grad[:, j:]
        return Model.linear_like(weight=weight, grad=grad)

    @torch.no_grad()
    def norms(self) -> dict[str, np.ndarray]:
        return {
            vector_name: vector_fn().weight.norm(dim=1).cpu().numpy()
            for vector_name, vector_fn in self.vector_fns.items()
        }

    @torch.no_grad()
    def grads(self) -> dict[str, np.ndarray]:
        return {
            vector_name: vector_fn().weight.grad.mean(dim=1).cpu().numpy()
            for vector_name, vector_fn in self.vector_fns.items()
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(x))).squeeze()

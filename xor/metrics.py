"""Metrics tracking for XOR experiments."""

from typing import Self

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

@dataclass
class Metrics:
    """Metrics to track over the course of training."""
    accuracies: dict[str, float] = field(default_factory=dict)
    norms: dict[str, np.ndarray] = field(default_factory=dict)
    grads: dict[str, np.ndarray] = field(default_factory=dict)
    margins: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def create(
        cls, vector_names: Sequence[str], p: int, train_steps: int) -> Self:
        """Pre-allocates memory for metrics arrays."""
        return cls(
            norms={
                vector_name: np.zeros((p, train_steps))
                for vector_name in vector_names
            },
            grads={
                vector_name: np.zeros((p, train_steps))
                for vector_name in vector_names
            },
            margins={
                "sp": np.zeros(train_steps),
                "no_sp": np.zeros(train_steps),
            },
        )

    def update_norms(self, step: int, norms: dict[str, np.ndarray]) -> None:
        """Updates norms of each vector at the given step."""
        for vector_name, norm_array in norms.items():
            self.norms[vector_name][:, step] = norm_array

    def update_grads(self, step: int, grads: dict[str, np.ndarray]) -> None:
        """Updates grads of each vector at the given step."""
        for vector_name, grad_array in grads.items():
            self.grads[vector_name][:, step] = grad_array

    def update_margins(self, step: int, margins: dict[str, float]) -> None:
        """Updates margins at the given step."""
        self.margins["sp"][step] = margins["sp"]
        self.margins["no_sp"][step] = margins["no_sp"]

@dataclass
class NamedMetricFn:
    """Helper class for plotting.

    Identity by default, metric_fn can be something like np.abs(x[0] - x[1]).
    """

    name: str
    metric_fn: Callable[[Sequence[np.ndarray]], np.ndarray] = lambda x: x[0]

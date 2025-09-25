from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

@dataclass
class Metrics:
    accuracies: dict[str, float] = field(default_factory=dict)
    norms: dict[str, np.ndarray] = field(default_factory=dict)
    grads: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        vector_names: Sequence[str],
        p: int,
        train_steps: int,
    ) -> Metrics:
        # Pre-allocate memory for norm and grad arrays.
        return cls(
            norms={
                vector_name: np.zeros((p, train_steps))
                for vector_name in vector_names
            },
            grads={
                vector_name: np.zeros((p, train_steps))
                for vector_name in vector_names
            }
        )

    def update_norms(self, step: int, norms: dict[str, np.ndarray]) -> None:
        for vector_name, norm_array in norms.items():
            self.norms[vector_name][:, step] = norm_array

    def update_grads(self, step: int, grads: dict[str, np.ndarray]) -> None:
        for vector_name, grad_array in grads.items():
            self.grads[vector_name][:, step] = grad_array

@dataclass
class NamedMetricFn:
    name: str
    metric_fn: Callable[[Sequence[np.ndarray]], np.ndarray] = lambda x: x[0]

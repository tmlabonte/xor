"""Metrics tracking for XOR experiments."""

from typing import Self

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Metrics:
    """Metrics to track over the course of training."""

    losses: dict[str, np.ndarray] = field(default_factory=dict)
    accuracies: dict[str, float] = field(default_factory=dict)
    norms: dict[str, np.ndarray] = field(default_factory=dict)
    grads: dict[str, np.ndarray] = field(default_factory=dict)
    margins: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def create(cls, keys: Sequence[str], p: int, d: int, train_steps: int) -> Self:
        """Pre-allocates memory for metrics arrays."""

        def make_array() -> dict[str, np.ndarray]:
            return {k: np.zeros((p, train_steps)) for k in keys}

        return cls(
            losses={"train loss": np.zeros(train_steps)},
            norms=make_array(),
            grads=make_array(),
            margins={
                "spurious": np.zeros(train_steps),
                "not spurious": np.zeros(train_steps),
            },
        )

    def _record(
        self, storage: dict, step: int, values: dict[str, np.ndarray | float]
    ) -> None:
        """Records values into storage dict at the given step."""
        for key, val in values.items():
            target = storage[key]
            if isinstance(target, np.ndarray) and target.ndim == 2:
                target[:, step] = val
            else:
                target[step] = val

    def record(self, step: int, **metrics: dict[str, np.ndarray | float]) -> None:
        """Records any combination of metrics at a given step.

        Example usage:
            metrics.record(
                step,
                losses={"train loss": 0.25},
                norms={"w": np.ones(3)},
                margins={"spurious": 0.1}
            )
        """

        storage_map = {
            "losses": self.losses,
            "accuracies": self.accuracies,
            "norms": self.norms,
            "grads": self.grads,
            "margins": self.margins,
        }

        for metric_name, values in metrics.items():
            storage = storage_map.get(metric_name)
            if storage is not None:
                self._record(storage, step, values)

@dataclass
class NamedMetricFn:
    """Helper class for plotting.

    Identity by default, metric_fn can be something like np.abs(x[0] - x[1]).
    """

    name: str  # What metric is being tracked, e.g., "norm"
    key: str  # What the metrics are indexed over, e.g., "weight" (for legend)
    metric_fn: Callable[[Sequence[np.ndarray]], np.ndarray] = lambda x: x[0]

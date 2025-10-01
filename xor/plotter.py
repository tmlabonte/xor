"""Various helpful plotting functions for XOR experiments."""

from collections.abc import Callable, Sequence
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from xor.constants import IMG_DIR
from xor.experiment import Experiment
from xor.metrics import Metrics, NamedMetricFn


class Plotter:
    """Class for various helpful plotting functions."""

    @staticmethod
    def _title(experiment: Experiment) -> str:
        """Returns the title for a plot given an experiment."""
        dataset_config = experiment.train_loader.dataset.config
        lmbda = dataset_config.lmbda
        lmbda_str = f", $\\lambda={lmbda}$" if lmbda is not None else ""
        return (
            f"{dataset_config.name}"
            f", $d={dataset_config.d}$"
            f", $m={dataset_config.m}$"
            f"{lmbda_str}"
            f", $\\eta={round(experiment.config.eta, 3)}$"
            f", $p={experiment.model.config.p}$"
            f", $\\theta={round(experiment.model.config.theta, 3)}$"
        )

    @staticmethod
    def _make_lineplot(
        experiments: Sequence[Experiment],
        data: pd.DataFrame,
        named_metric_fn: NamedMetricFn,
        errorbar: str | None = None,
    ) -> None:
        """Creates the plot and saves to disk."""

        # Automatically plot at full steps and 1/4 steps.
        sns.set_theme()
        for ratio in [0.25, 1.0]:
            thresh = int(ratio * data["step"].max())

            # Plot every single neuron if they exist and errorbar is not specified.
            # Otherwise aggregate over neurons to plot mean and errorbar.
            sns.lineplot(
                # data=data[data["step"] <= thresh],
                data=data.loc[data["step"] <= thresh].reset_index(drop=True),
                x="step",
                y=named_metric_fn.name,
                hue=named_metric_fn.key,
                errorbar=errorbar,
                units="neuron" if "neuron" in data.columns else None,
                estimator=(
                    None if errorbar is None and "neuron" in data.columns else "mean"
                ),
            )

            # Save to all experiment directories.
            for experiment in experiments:
                plt.title("")  # Clears title
                plt.title(Plotter._title(experiment))
                img_dir = os.path.join(
                    experiment.config.exp_dir, IMG_DIR, f"step={thresh}"
                )
                os.makedirs(img_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(img_dir, f"{named_metric_fn.name}.png"),
                    bbox_inches="tight",
                    dpi=600,
                )

            plt.clf()

    @staticmethod
    def _access_metrics(
        experiments: Sequence[Experiment],
        accessor: Callable[[Metrics], dict[str, np.ndarray]],
        exclude_keys: Sequence[str] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Accesses metrics for each experiment while excluding given keys."""
        exclude_keys = [] if exclude_keys is None else exclude_keys
        metrics = [
            {
                k: v
                for k, v in accessor(experiment.metrics).items()
                if k not in exclude_keys
            }
            for experiment in experiments
        ]
        return metrics

    @staticmethod
    def _organize_data(
        experiments: Sequence[Experiment],
        accessor: Callable[[Metrics], dict[str, np.ndarray]],
        named_metric_fn: NamedMetricFn,
        exclude_keys: Sequence[str] | None = None,
        skip_zeroth_step: bool = False,
    ) -> pd.DataFrame:
        """Organizes metrics data into DataFrame."""
        metrics = Plotter._access_metrics(experiments, accessor, exclude_keys)

        # Zip all experiment metrics (they should have the same keys).
        metrics_zip = zip(
            metrics[0].keys(),
            zip(*(metric.values() for metric in metrics)),
        )

        # Organize data into DataFrame so as to take mean/std over neurons.
        data = []
        start = 1 if skip_zeroth_step else 0
        ndim = next(iter(metrics[0].values())).ndim
        if ndim == 1:  # n_steps
            for key, metric in metrics_zip:

                def metric_fn(metric=metric):
                    """Wraps named_metric_fn.metric_fn for consistency."""
                    return named_metric_fn.metric_fn(metric)[start:]

                (n_steps,) = metric[0].shape
                data.append(
                    pd.DataFrame(
                        {
                            "step": np.arange(start, n_steps),
                            named_metric_fn.name: metric_fn(),
                            named_metric_fn.key: key,
                        }
                    )
                )
        elif ndim == 2:  # n_neurons x n_steps
            for key, metric in metrics_zip:

                def metric_fn(neuron, metric=metric):
                    """Wraps named_metric_fn.metric_fn for the current neuron."""
                    return named_metric_fn.metric_fn(metric)[neuron, start:]

                n_neurons, n_steps = metric[0].shape
                for neuron in range(n_neurons):
                    data.append(
                        pd.DataFrame(
                            {
                                "step": np.arange(start, n_steps),
                                named_metric_fn.name: metric_fn(neuron),
                                named_metric_fn.key: key,
                                "neuron": neuron,
                            }
                        )
                    )
        else:
            raise ValueError(
                f"Unexpected number of dimensions of metric. Current value: {ndim}"
            )
        data = pd.concat(data)

        return data

    @staticmethod
    def _plot(
        experiments: Sequence[Experiment],
        accessor: Callable[[Metrics], dict[str, np.ndarray]],
        named_metric_fn: NamedMetricFn,
        errorbar: str | None = None,
        exclude_keys: Sequence[str] | None = None,
        skip_zeroth_step: bool = False,
    ) -> None:
        """Main function for handling different plot requests."""
        # Organize metrics data into DataFrame.
        data = Plotter._organize_data(
            experiments=experiments,
            accessor=accessor,
            named_metric_fn=named_metric_fn,
            exclude_keys=exclude_keys,
            skip_zeroth_step=skip_zeroth_step,
        )

        # Create the plot and save to disk.
        Plotter._make_lineplot(
            experiments=experiments,
            data=data,
            named_metric_fn=named_metric_fn,
            errorbar=errorbar,
        )

    @staticmethod
    def plot_losses(
        experiment: Experiment,
        exclude_keys: Sequence[str] | None = None,
    ) -> None:
        """Plots losses against train steps."""
        Plotter._plot(
            experiments=[experiment],
            accessor=lambda metrics: metrics.losses,
            named_metric_fn=NamedMetricFn("loss", "type"),
            exclude_keys=exclude_keys,
            skip_zeroth_step=True,
        )

    @staticmethod
    def plot_norms(
        experiment: Experiment,
        errorbar: str | None = None,
        exclude_keys: Sequence[str] | None = None,
    ) -> None:
        """Plots vector norms against train steps."""
        Plotter._plot(
            experiments=[experiment],
            accessor=lambda metrics: metrics.norms,
            named_metric_fn=NamedMetricFn("norm", "weight"),
            errorbar=errorbar,
            exclude_keys=exclude_keys,
            skip_zeroth_step=False,
        )

    @staticmethod
    def plot_grads(
        experiment: Experiment,
        errorbar: str | None = None,
        exclude_keys: Sequence[str] | None = None,
    ) -> None:
        """Plots vector grads against train steps."""
        Plotter._plot(
            experiments=[experiment],
            accessor=lambda metrics: metrics.grads,
            named_metric_fn=NamedMetricFn("grad", "weight"),
            errorbar=errorbar,
            exclude_keys=exclude_keys,
            skip_zeroth_step=True,
        )

    @staticmethod
    def plot_margins(
        experiment: Experiment,
        exclude_keys: Sequence[str] | None = None,
    ) -> None:
        """Plots margins against train steps."""
        Plotter._plot(
            experiments=[experiment],
            accessor=lambda metrics: metrics.margins,
            named_metric_fn=NamedMetricFn("margin", "subset"),
            errorbar=None,
            exclude_keys=exclude_keys,
            skip_zeroth_step=True,
        )

    @staticmethod
    def plot_grad_errors(
        experiments: Sequence[Experiment],
        errorbar: str | None = None,
        exclude_keys: Sequence[str] | None = None,
    ) -> None:
        """Plots grad errors against train steps."""
        # Compute absolute difference between L0 and Lp grads.
        error_fn = NamedMetricFn(
            "grad error", "weight", metric_fn=lambda x: np.abs(x[0] - x[1])
        )

        Plotter._plot(
            experiments=experiments,
            accessor=lambda metrics: metrics.grads,
            named_metric_fn=error_fn,
            errorbar=errorbar,
            exclude_keys=exclude_keys,
            skip_zeroth_step=True,
        )

    @staticmethod
    def plot_all(
        experiment: Experiment,
        errorbar: str | None = None,
        exclude_keys: Sequence[str] | None = None,
    ) -> None:
        """Plots all single-experiment metrics."""
        Plotter.plot_losses(experiment, exclude_keys=exclude_keys)
        Plotter.plot_norms(experiment, errorbar=errorbar, exclude_keys=exclude_keys)
        Plotter.plot_grads(experiment, errorbar=errorbar, exclude_keys=exclude_keys)
        Plotter.plot_margins(experiment, exclude_keys=exclude_keys)

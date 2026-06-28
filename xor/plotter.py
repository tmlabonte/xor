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

plt.figure(figsize=(8, 5))

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
        legend: bool = True,
    ) -> None:
        """Creates the plot and saves to disk."""

        # Automatically plot at full steps and 1/4 steps.
        sns.set_theme(
            style="white",
            context="paper",
            font_scale=2,
        )

        # Hardcoded phase ratios; change if desired.
        ratios = [(0, 0.025), (0.025, 0.167), (0.167, 1.0), (0, 1.0)]
        for ratio in ratios:
            lower = int(ratio[0] * data["step"].max())
            upper = int(ratio[1] * data["step"].max())

            palette = None
            if "w_perp" in data[named_metric_fn.key].unique():
                palette = {
                    "w_sig": "tab:blue",
                    "w_opp": "tab:orange",
                    "w_sp": "tab:green",
                    "w_perp": "tab:red",
                }

            data[named_metric_fn.key] = data[named_metric_fn.key].replace({
                "spurious": "majority group",
                "not spurious": "minority group",
            })

            # Plot every single neuron if they exist and errorbar is not specified.
            # Otherwise aggregate over neurons to plot mean and errorbar.
            ax = sns.lineplot(
                data=data.loc[data["step"].between(lower, upper)].reset_index(drop=True),
                x="step",
                y=named_metric_fn.name,
                hue=named_metric_fn.key,
                palette=palette,
                errorbar=errorbar,
                legend=legend,
                units="neuron" if "neuron" in data.columns else None,
                estimator=(
                    None if errorbar is None and "neuron" in data.columns else "mean"
                ),
                linewidth=3,
            )

            label_map = {
                "w_sig": r"$\mathbf{w}_{\mathrm{sig}}$",
                "w_opp": r"$\mathbf{w}_{\mathrm{opp}}$",
                "w_sp": r"$\mathbf{w}_{\mathrm{sp}}$",
                "w_perp": r"$\mathbf{w}_{\perp}$",
            }

            handles, labels = ax.get_legend_handles_labels()

            ax.legend(
                handles,
                [label_map.get(l, l) for l in labels],
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=4,
                frameon=False,
            )
            ax.grid(True, alpha=0.5)
            sns.despine()

            # Save to all experiment directories.
            for experiment in experiments:
                # plt.title("")  # Clears title
                plt.title(Plotter._title(experiment))
                img_dir = os.path.join(
                    experiment.config.exp_dir, IMG_DIR, f"step={upper}"
                )
                os.makedirs(img_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(img_dir, f"{named_metric_fn.name}_{lower}_{upper}.png"),
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
        legend: bool = True,
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
            legend=legend,
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
            named_metric_fn=NamedMetricFn("norm", "vector"),
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
            named_metric_fn=NamedMetricFn("grad", "vector"),
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
    def plot_all(
        experiment: Experiment,
        errorbar: str | None = None,
        exclude_keys: Sequence[str] | None = None,
    ) -> None:
        """Plots all single-experiment metrics."""
        exclude_keys = ["a"]
        Plotter.plot_losses(experiment, exclude_keys=exclude_keys)
        Plotter.plot_norms(experiment, errorbar=errorbar, exclude_keys=exclude_keys)
        Plotter.plot_grads(experiment, errorbar=errorbar, exclude_keys=exclude_keys)
        Plotter.plot_margins(experiment, exclude_keys=exclude_keys)

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from xor.experiment import Experiment
from xor.metrics import Metrics, NamedMetricFn

class Plotter:
    @staticmethod
    def _title(experiment: Experiment) -> str:
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
        experiment: Experiment,
        data: pd.DataFrame,
        metric_name: str,
        errorbar: str | None = "sd",
    ) -> None:
        sns.set_theme()
        sns.lineplot(
            data=data,
            x="step",
            y=metric_name,
            hue="weight",
            errorbar=errorbar,
        )

        plt.title(Plotter._title(experiment))

        dataset_name = experiment.train_loader.dataset.config.name
        import uuid
        u = str(uuid.uuid4())[:4]
        plt.savefig(
            f"{dataset_name}_{metric_name}_{u}.png",
            bbox_inches="tight",
            dpi=600,
        )
        plt.clf()

    @staticmethod
    def _organize_data(
        experiments: Sequence[Experiment],
        accessor: Callable[[Metrics], dict[str, np.ndarray]],
        named_metric_fn: NamedMetricFn,
        exclude_keys: Sequence[str] = None,
        skip_zeroth_step: bool = False,
    ) -> pd.DataFrame:
        # Access metrics for each experiment while excluding given keys.
        exclude_keys = [] if exclude_keys is None else exclude_keys
        metrics = [
            {
                k: n for k, n in accessor(experiment.metrics).items()
                if k not in exclude_keys
            }
            for experiment in experiments
        ]

        # Get the number of train steps (optionally counting the zeroth).
        x_start = 1 if skip_zeroth_step else 0
        x = np.arange(x_start, next(iter(metrics[0].values())).shape[1])

        # Zip all experiment metrics (they should have the same keys).
        metrics_zip = zip(
            metrics[0].keys(),
            zip(*(metric.values() for metric in metrics)),
        )

        # Organize data into DataFrame so as to take mean/std over neurons.
        data = []
        for key, metric in metrics_zip:
            # Wrap named_metric_fn.metric_fn for the current neuron.
            def metric_fn(neuron):
                computed_metric = named_metric_fn.metric_fn(metric)
                return computed_metric[neuron, x_start:]

            n_neurons, n_steps = metric[0].shape
            n_steps = n_steps - 1 if skip_zeroth_step else n_steps
            steps = np.arange(n_steps)
            for neuron in range(n_neurons):
                data.append(pd.DataFrame({
                    "step": np.arange(n_steps),
                    named_metric_fn.name: metric_fn(neuron),
                    "weight": key,
                    "neuron": neuron,
                }))
        data = pd.concat(data)

        return data

    @staticmethod
    def _plot(
        experiments: Sequence[Experiment],
        accessor: Callable[[Metrics], dict[str, np.ndarray]],
        named_metric_fn: NamedMetricFn,
        errorbar: str | None = "sd",
        exclude_keys: Sequence[str] = None,
        skip_zeroth_step: bool = False,
    ) -> None:
        data = Plotter._organize_data(
            experiments=experiments,
            accessor=accessor,
            named_metric_fn=named_metric_fn,
            exclude_keys=exclude_keys,
            skip_zeroth_step=skip_zeroth_step,
        )

        Plotter._make_lineplot(
            experiment=experiments[0],
            data=data,
            metric_name=named_metric_fn.name,
            errorbar=errorbar,
        )

    @staticmethod
    def plot_norms(
        experiment: Experiment,
        errorbar: str | None = "sd",
        exclude_keys: Sequence[str] = None,
        skip_zeroth_step: bool = False,
    ) -> None:
        return Plotter._plot(
            experiments=[experiment],
            accessor=lambda metrics: metrics.norms,
            named_metric_fn=NamedMetricFn("norm"),
            errorbar=errorbar,
            exclude_keys=exclude_keys,
            skip_zeroth_step=False,
        )

    @staticmethod
    def plot_grads(
        experiment: Experiment,
        errorbar: str | None = "sd",
        exclude_keys: Sequence[str] = None,
    ) -> None:
        return Plotter._plot(
            experiments=[experiment],
            accessor=lambda metrics: metrics.grads,
            named_metric_fn=NamedMetricFn("grad"),
            errorbar=errorbar,
            exclude_keys=exclude_keys,
            skip_zeroth_step=True,
        )

    @staticmethod
    def plot_grad_errors(
        experiments: Sequence[Experiment],
        errorbar: str | None = "sd",
        exclude_keys: Sequence[str] = None,
        skip_zeroth_step: bool = False,
    ) -> None:
        error_fn = NamedMetricFn("grad error", lambda x: np.abs(x[0] - x[1]))
        return Plotter._plot(
            experiments=experiments,
            accessor=lambda metrics: metrics.grads,
            named_metric_fn=error_fn,
            errorbar=errorbar,
            exclude_keys=exclude_keys,
            skip_zeroth_step=True,
        )

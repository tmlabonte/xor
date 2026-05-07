"""Simple experiment to exemplify and verify XOR logic."""

from argparse import Namespace
from collections.abc import Sequence
import os

from loguru import logger

from xor.args import parse_args
from xor.constants import IMG_DIR
from xor.dataset import Dataset, DatasetConfig, make_dataset_configs
from xor.experiment import Experiment, ExperimentConfig
from xor.model import Model, ModelConfig
from xor.plotter import Plotter


def run_experiment(
    train_config: DatasetConfig,
    test_configs: Sequence[DatasetConfig],
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
) -> Experiment:
    """Runs an experiment given dataset, model, and experiment configs. Reusable."""
    # Instantiate dataset, model, and experiments.
    train_dataset = Dataset(train_config)
    test_datasets = [Dataset(test_config) for test_config in test_configs]
    model = Model(model_config)
    experiment = Experiment(train_dataset, test_datasets, model, experiment_config)

    # Train and evaluate the model.
    experiment.train()
    metrics = experiment.test()
    logger.info(f"Accuracies: {metrics.accuracies}")

    # Plot metrics of interest and save to disk.
    Plotter.plot_all(experiment)
    plot_dir = os.path.join(experiment.config.exp_dir, IMG_DIR)
    logger.info(f"Plots saved to {plot_dir}")

    return experiment


def main(args: Namespace) -> None:
    """Train and test on XOR dataset with spurious correlation."""
    # Instantiate dataset, model, and experiment configs.
    dataset_configs = make_dataset_configs(args)
    model_config = ModelConfig(d=args.d, p=args.p, theta=args.theta, spurious=True)
    key = "-hybrid" if args.hybrid else ""
    experiment_config = ExperimentConfig(
        name=f"xor-spurious-{args.loss}{key}",
        loss=args.loss,
        eta=args.eta,
        hybrid=args.hybrid,
    )

    # Runs the experiment.
    run_experiment(
        dataset_configs["train_spurious"],
        # [dataset_configs["test_iid"], dataset_configs["test_spurious"]],
        [dataset_configs["test_spurious"]],
        model_config,
        experiment_config,
    )


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)

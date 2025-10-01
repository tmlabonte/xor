"""Experiment for comparison of Lp and L0 losses."""

from argparse import Namespace

from xor.args import parse_args
from xor.dataset import make_dataset_configs
from xor.experiment import Experiment, ExperimentConfig
from xor.main import run_experiment
from xor.model import ModelConfig
from xor.plotter import Plotter


def main(args: Namespace) -> None:
    """Train and test on XOR dataset with spurious correlation."""
    # Instantiate dataset, model, and experiment configs.
    dataset_configs = make_dataset_configs(args)
    model_config = ModelConfig(d=args.d, p=args.p, theta=args.theta, spurious=True)

    def run_loss_experiment(loss: str) -> Experiment:
        """Runs an experiment using a given loss."""
        experiment_config = ExperimentConfig(
            name=f"xor-spurious-{loss}", loss=loss, eta=args.eta
        )

        experiment = run_experiment(
            dataset_configs["train_spurious"],
            [dataset_configs["test_iid"], dataset_configs["test_spurious"]],
            model_config,
            experiment_config,
        )

        return experiment

    lp_experiment = run_loss_experiment("lp")
    l0_experiment = run_loss_experiment("l0")
    Plotter.plot_grad_errors([lp_experiment, l0_experiment])


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)

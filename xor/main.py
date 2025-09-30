"""Simple experiment to exemplify and verify XOR logic."""

from xor.args import parse_args
from xor.dataset import Dataset, DatasetConfig
from xor.experiment import Experiment, ExperimentConfig
from xor.model import Model, ModelConfig
from xor.plotter import Plotter


def main(args):
    """Train and test on XOR dataset with spurious correlation."""

    # Instantiate dataset, model, and experiment configs.
    train_config = DatasetConfig(
        name="train",
        d=args.d,
        n=args.m * args.steps,
        m=args.m,
        seed=args.train_seed,
        lmbda=args.lmbda,
        spurious=True,
    )
    test_config = DatasetConfig(
        name="test",
        d=args.d,
        n=args.n_test,
        m=args.m,
        seed=args.test_seed,
        lmbda=args.lmbda,
        spurious=True,
    )
    model_config = ModelConfig(d=args.d, p=args.p, theta=args.theta, spurious=True)
    experiment_config = ExperimentConfig(
        name="xor-spurious", loss=args.loss, eta=args.eta
    )

    # Instantiate dataset, model, and experiments.
    train_dataset = Dataset(train_config)
    test_dataset = Dataset(test_config)
    model = Model(model_config)
    experiment = Experiment(train_dataset, [test_dataset], model, experiment_config)

    # Train and evaluate the model.
    experiment.train()
    metrics = experiment.test()

    # Print accuracies and plot metrics of interest.
    print(metrics.accuracies)
    Plotter.plot_all(experiment)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)

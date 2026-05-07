"""Experiment for debiasing via last-layer retraining."""

from argparse import Namespace
import os
import pickle
import sys

from configargparse import Parser
from loguru import logger
from torch import nn
import torch.nn.init as init

from xor.args import parse_args
from xor.constants import IMG_DIR
from xor.dataset import Dataset, make_dataset_configs
from xor.experiment import Experiment, ExperimentConfig
from xor.model import Model, ModelConfig, ThreeLayerMLP


def main(args: Namespace) -> None:
    """Train and test on XOR dataset with spurious correlation."""
    args.test_seed = 99
    seeds = [i for i in range(10)]
    #lmbdas = [round(0.01 + i * 0.01, 2) for i in range(5)]
    # lmbdas = [round(0.02 + i * 0.005, 3) for i in range(3)] # This one is OK
    lmbdas = [0.05]
    heldout_lmbdas = [0.01, 0.03, 0.05, 0.08, 0.2, 0.5]

    all_metrics = {}
    for lmbda in lmbdas:
        all_metrics[lmbda] = {}
        for heldout_lmbda in heldout_lmbdas:
            all_metrics[lmbda][heldout_lmbda] = [[0, 0] for _ in range(len(seeds))]

    for seed in seeds:
        args.train_seed = seed
        args.heldout_seed = seed + 1
        for lmbda in lmbdas:
            args.lmbda = lmbda
            # Instantiate dataset, model, and experiment configs.
            dataset_configs = make_dataset_configs(args)
            model_config = ModelConfig(d=args.d, p=args.p, theta=args.theta, spurious=True)

            experiment_config = ExperimentConfig(
                name=f"xor-erm-{seed}-{lmbda}", eta=args.eta, loss="lp"
            )

            # Instantiate dataset, model, and experiment.
            train_dataset = Dataset(dataset_configs["train_spurious"])
            test_datasets = [Dataset(dataset_configs["test_iid"])]
            model = ThreeLayerMLP(model_config)
            # model = Model(model_config)
            experiment = Experiment(train_dataset, test_datasets, model, experiment_config)

            # Train and evaluate the model.
            model, _ = experiment.train()
            metrics = experiment.test()
            erm_acc = metrics.accuracies["test-iid"]
            logger.info(f"ERM Accuracies: {metrics.accuracies}")

            for heldout_lmbda in heldout_lmbdas:
                args.lmbda = heldout_lmbda
                args.seed = args.heldout_seed
                experiment_config = ExperimentConfig(
                    name=f"xor-llr-{seed}-{lmbda}-{heldout_lmbda}", eta=args.eta, loss="lp"
                )

                # Freeze hidden and reinitialize output for LLR.
                #model.freeze_hidden()
                #model.initialize_parameters(output_only=True)
                for p in model.fc1.parameters():
                    p.requires_grad = False
                for p in model.fc2.parameters():
                    p.requires_grad = False
                init.xavier_uniform_(model.fc_out.weight)
                nn.init.zeros_(model.fc_out.bias)

                dataset_configs = make_dataset_configs(args)
                train_dataset = Dataset(dataset_configs["train_spurious"])
                experiment = Experiment(train_dataset, test_datasets, model, experiment_config)

                # Train and evaluate the model.
                experiment.train()
                metrics = experiment.test()
                llr_acc = metrics.accuracies["test-iid"]
                logger.info(f"LLR Accuracies: {metrics.accuracies}")
                all_metrics[lmbda][heldout_lmbda][seed][0] = erm_acc
                all_metrics[lmbda][heldout_lmbda][seed][1] = llr_acc

    with open("all_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)

if __name__ == "__main__":
    parsed_args = parse_args()

    parsed_args.p = 50
    parsed_args.m = 32
    parsed_args.eta = 0.05
    parsed_args.theta = 0.001

    main(parsed_args)

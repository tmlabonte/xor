"""Argument parsing logic for XOR experiments."""

from argparse import Namespace
import math
import sys

from configargparse import Parser
from loguru import logger

from xor.util import strtobool


def _add_input_args(parser: Parser) -> None:
    """Loads configuration parameters into parser."""
    parser.add(
        "--C",
        default=2,
        type=int,
        help="Constant C for initialization as in Glasgow (2024).",
    )
    parser.add("--d", default=100, type=int, help="The dimension of the data.")
    parser.add(
        "--hybrid",
        default=False,
        type=strtobool,
        help=(
            "Whether to use the hybrid gradient, i.e., the Lp gradient for w_sp and"
            " the L0 gradient for other parameters."
        ),
    )
    parser.add(
        "--lmbda",
        "--lambda",
        default=0.2,
        type=float,
        help=(
            "The lambda parameter corresponding to the strength of the spurious"
            " correlation. Should be between 0 and 0.5 with a lower number implying a"
            " stronger correlation. Only applies if spurious=True in dataset config."
        ),
    )
    parser.add(
        "--loss",
        default="lp",
        choices=["l0", "lp"],
        help=(
            "The training loss to use. 'l0' is the first-order logistic loss (Glasgow"
            " (2024)). 'lp' is the standard logistic loss."
        ),
    )
    parser.add(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="The verbosity level for logging.",
    )
    parser.add(
        "--n_test", default=100000, type=int, help="The number of data for testing."
    )
    parser.add("--steps", default=100, type=int, help="The number of train steps.")
    parser.add(
        "--train_seed", default=0, type=int, help="The RNG seed for the train dataset."
    )
    parser.add(
        "--test_seed", default=1, type=int, help="The RNG seed for the test dataset(s)."
    )
    return parser

def _add_initialization_parameters(args: Namespace) -> None:
    """Sets initialization hyperparameters."""
    args.p = 10 # args.p = 10 # args.p = math.ceil(math.log(args.d)) # log(d)
    args.eta = 0.05 # args.eta = 0.05 # args.eta = 1 / (math.log(args.d)) # log^{-1}(d)
    args.theta = 0.0001 # args.theta = 0.01 # args.theta = 1 / (math.log(args.d) ** args.C) # log^{-C}(d)
    args.m = 5000 # args.m = math.ceil((args.d / (args.theta ** 2)))

def parse_args() -> Namespace:
    """Parses input arguments and configures logger."""
    parser = Parser()
    _add_input_args(parser)
    args = parser.parse_args()
    _add_initialization_parameters(args)

    # Configure logger level and format.
    logger.remove()
    logger.add(
        sys.stdout,
        level=args.loglevel,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name: <15}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: >4}</cyan> - "
        "<level>{message}</level>",
    )

    return args

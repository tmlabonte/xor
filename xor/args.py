"""Argument parsing logic for XOR experiments."""

from argparse import Namespace
import math
import sys

from configargparse import Parser
from loguru import logger


def _add_input_args(parser: Parser) -> None:
    """Loads configuration parameters into parser."""
    parser.add("--d", default=100, type=int, help="The dimension of the data.")
    parser.add("--eta", default=0.05, type=float, help="The learning rate.")
    parser.add(
        "--lmbda",
        "--lambda",
        default=0.1,
        type=float,
        help=(
            "The lambda parameter corresponding to the strength of the spurious"
            " correlation. Should be between 0 and 0.5 with a lower number implying a"
            " stronger correlation. Only applies if spurious=True."
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
    parser.add("--m", default=5000, type=int, help="The batch size.")
    parser.add(
        "--no_spurious",
        dest="spurious",
        action="store_false",
        help="Turn off the spurious feature.",
    )
    parser.add(
        "--n_test", default=100000, type=int, help="The number of data for testing."
    )
    parser.add("--p", default=10, type=int, help="The number of neurons.")
    parser.add(
        "--spurious",
        dest="spurious",
        action="store_true",
        help="Turn on the spurious feature.",
    )
    parser.add("--steps", default=500, type=int, help="The number of train steps.")
    parser.add("--theta", default=0.01, type=float, help="The initialization scale.")
    parser.add(
        "--train_seed", default=0, type=int, help="The RNG seed for the train dataset."
    )
    parser.add(
        "--test_seed", default=1, type=int, help="The RNG seed for the test dataset(s)."
    )
    parser.set_defaults(spurious=True)
    return parser

def parse_args() -> Namespace:
    """Parses input arguments and configures logger."""
    parser = Parser()
    _add_input_args(parser)
    args = parser.parse_args()

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

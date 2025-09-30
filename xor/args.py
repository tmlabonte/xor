"""Argument parsing logic for XOR experiments."""

from argparse import Namespace
import math

from configargparse import Parser


def strtobool(value: str) -> bool:
    """Converts a string to a boolean.

    True values: "y", "yes", "t", "true", "on", "1".
    False values: "n", "no", "f", "false", "off", "0".
    Case-insensitive. Raises ValueError if unrecognized.
    """

    true_values = {"y", "yes", "t", "true", "on", "1"}
    false_values = {"n", "no", "f", "false", "off", "0"}

    val = value.strip().lower()
    if val in true_values:
        return True
    if val in false_values:
        return False
    raise ValueError(f"Please input a boolean. Current value: '{value}'")


def _add_input_args(parser: Parser) -> Parser:
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
            "Whether to use the hybrid gradient, i.e., the Lp gradient"
            " for w_sp and the L0 gradient for other parameters."
        ),
    )
    parser.add(
        "--lmbda",
        default=0.2,
        type=float,
        help=(
            "The lambda parameter corresponding to the strength of the"
            " spurious correlation. Should be between 0 and 0.5 with a"
            " lower number implying a stronger correlation. Only"
            " applies if spurious=True in the dataset config."
        ),
    )
    parser.add(
        "--loss",
        default="lp",
        choices=["l0", "lp"],
        help=(
            "The training loss to use."
            " 'l0' is the first-order logistic loss (Glasgow 2024)."
            " 'lp' is the standard logistic loss."
        ),
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
    """Sets initialization hyperparameters specified by Glasgow (2024)."""
    psi = 1 / (math.log(args.d) ** args.C)
    args.m = math.ceil(args.d / psi)
    args.p = math.ceil(1 / psi)
    args.eta = psi
    args.theta = psi / math.sqrt(args.p)


def parse_args() -> Namespace:
    """Parses input arguments."""
    parser = Parser()
    parser = _add_input_args(parser)
    args = parser.parse_args()
    _add_initialization_parameters(args)
    return args

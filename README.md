# XOR with Spurious Correlation

## Installation
```
conda update -n base -c defaults conda
conda create -n xor python==3.12
conda activate xor
python -m pip install -e .
```

## Running Experiments
To run the main driver file, use `python xor/main.py`.
Experiments are randomly assigned unique names and saved to `out/` by default.
Use `python xor/main.py --help` to see command line options.

## Citations
Please consider using the following citation:
```
@article{
  author={Tyler LaBonte and Vidya Muthukumar},
  title={SGD Provably Prioritizes a Shortcut Feature in the XOR Model},
  year={2026},
}
```

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
Experiments are randomly assigned unique names and saved to `out` by default.
To make your own experiment, create a file in `exps`.
Commonly reused functions for new experiments include `xor.dataset.make_dataset_configs`
and `xor.main.run_experiment`.

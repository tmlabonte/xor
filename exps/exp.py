import math


from xor.dataset import Dataset, DatasetConfig
from xor.experiment import Experiment, ExperimentConfig
from xor.model import Model, ModelConfig
from xor.plotter import Plotter

# Set hyperparameters chosen by user.
d = 1000
steps = 2000
lmbda = 0.2
n_test = 100000
train_seed = 0
test_seed = 1

# Set hyperparameters specified by Glasgow (2024).
C = 2 # Just a guess
psi = 1 / (math.log(d) ** C)
m = math.ceil(d / psi)
p = math.ceil(1 / psi)
eta = psi
theta = psi / math.sqrt(p)

# Initialize configurations based on chosen hyperparameters.
train_iid_dataset_config = DatasetConfig(
    name="train_iid",
    d=d,
    n=m * steps,
    m=m,
    seed=train_seed,
)
train_spurious_dataset_config = DatasetConfig(
    name="train_spurious",
    d=d,
    n=m*steps,
    m=m,
    seed=train_seed,
    spurious=True,
    lmbda=lmbda,
)
test_iid_dataset_config = DatasetConfig(
    name="test_iid",
    d=d,
    n=n_test,
    m=m,
    seed=test_seed,
)
test_spurious_dataset_config = DatasetConfig(
    name="test_spurious",
    d=d,
    n=n_test,
    m=m,
    seed=test_seed,
    spurious=True,
    lmbda=lmbda,
)

# Train with spurious correlation under l0 loss.
train_spurious_dataset = Dataset(train_spurious_dataset_config)
test_iid_dataset = Dataset(test_iid_dataset_config)
test_spurious_dataset = Dataset(test_spurious_dataset_config)

model_config = ModelConfig(d=d, p=p, spurious=True, theta=theta)
model = Model(model_config)

l0_experiment_config = ExperimentConfig(name="spurious", loss="l0", eta=eta)
l0_experiment = Experiment(
    train_spurious_dataset,
    [test_iid_dataset, test_spurious_dataset],
    model,
    l0_experiment_config,
)

l0_experiment.train()
l0_metrics = l0_experiment.test()

print(l0_metrics.accuracies)
Plotter.plot_norms(l0_experiment)

# Train with spurious correlation under lp loss.
train_spurious_dataset = Dataset(train_spurious_dataset_config)
test_iid_dataset = Dataset(test_iid_dataset_config)
test_spurious_dataset = Dataset(test_spurious_dataset_config)

model_config = ModelConfig(d=d, p=p, spurious=True, theta=theta)
model = Model(model_config)

lp_experiment_config = ExperimentConfig(name="spurious", loss="lp", eta=eta)
lp_experiment = Experiment(
    train_spurious_dataset,
    [test_iid_dataset, test_spurious_dataset],
    model,
    lp_experiment_config,
)

lp_experiment.train()
lp_metrics = lp_experiment.test()

print(lp_metrics.accuracies)
Plotter.plot_norms(lp_experiment)

# Train with spurious correlation under hybrid gradient.
train_spurious_dataset = Dataset(train_spurious_dataset_config)
test_iid_dataset = Dataset(test_iid_dataset_config)
test_spurious_dataset = Dataset(test_spurious_dataset_config)

model_config = ModelConfig(d=d, p=p, spurious=True, theta=theta)
model = Model(model_config)

hybrid_experiment_config = ExperimentConfig(name="spurious", loss="l0", eta=eta, hybrid=True)
hybrid_experiment = Experiment(
    train_spurious_dataset,
    [test_iid_dataset, test_spurious_dataset],
    model,
    hybrid_experiment_config,
)

hybrid_experiment.train()
hybrid_metrics = hybrid_experiment.test()

print(hybrid_metrics.accuracies)
Plotter.plot_norms(hybrid_experiment)

# Plots grad errors between lp and l0 experiments.
Plotter.plot_grad_errors([lp_experiment, l0_experiment])

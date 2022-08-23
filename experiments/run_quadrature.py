"""Quadrature-based experiments.
"""

import matplotlib

matplotlib.use('Agg')

from itertools import product
import numpy as np
from joblib.parallel import Parallel, delayed
import torch

from nlica.runner import run_mse_quadrature
from nlica.defaults import RESULTS_FOLDER


N_JOBS = 40


# loop over

generations = np.array(["gaussian_mean", "gaussian_var", "gaussian_corr"])
estimations = ['singlemle', 'singlence']
data_vals = {
    "gaussian_mean": np.linspace(start=-5., stop=5., num=21),
    "gaussian_var": np.logspace(start=-1., stop=1., num=21, base=10),
    "gaussian_corr": np.linspace(start=-0.9, stop=0.9, num=31)
}
noise_types = {
    'singlemle': ['data'],
    'singlence': ['data', 'optimal']
}
noise_vals = {
    "gaussian_mean": {
        "singlemle": {
            "data": [0.1],
            "optimal": [0.1]
        },
        "singlence": {
            "data": np.linspace(start=-10, stop=10, num=82),
            "optimal": [0.1]
        }
    },
    "gaussian_var": {
        "singlemle": {
            "data": [0.1],
            "optimal": [0.1]
        },
        "singlence": {
            "data": np.logspace(start=-3, stop=2, num=82, base=10),
            "optimal": [0.1]
        }
    },
    "gaussian_corr": {
        "singlemle": {
            "data": [0.1],
            "optimal": [0.1]
        },
        "singlence": {
            "data": np.linspace(start=-0.99, stop=0.99, num=81),
            "optimal": [0.1]
        }
    }
}
prop_noises = {
    "gaussian_mean": np.linspace(0.01, 0.99, 41),
    "gaussian_var": np.linspace(0.01, 0.99, 41),
    "gaussian_corr": np.linspace(0.01, 0.99, 41),
}


# check the values are acceptable
for generation in generations:
    assert len([x for x in data_vals[generation] if x in noise_vals[generation]]) == 0, \
        f"There are overlapping noise and data parameters for the {generation} model."


# Run things in parallel with cartesian product on all parameters
# results is a list of dictionaries, each a line for a dataframe
results = Parallel(n_jobs=N_JOBS, verbose=1, backend='loky')(
    delayed(run_mse_quadrature)(
        estimation=estimation,
        generation=generation,
        data_val=data_val, noise_val=noise_val,
        prop_noise=prop_noise, noise_type=noise_type,
        n_samples=4000)
    for generation, estimation in product(generations, estimations)
    for noise_type in noise_types[estimation]
    for data_val, prop_noise in product(data_vals[generation], prop_noises[generation])  # nested for loop
    for noise_val in noise_vals[generation][estimation][noise_type]
)

filename = RESULTS_FOLDER / "expe_quadrature.th"
torch.save(results, filename)

"""Optimal noise for the generative models.
"""

"""MSE vs. n_samples for the generative models.
"""

import matplotlib

matplotlib.use('Agg')


from joblib.parallel import Parallel, delayed
import torch
from itertools import product

from nlica.runner import run_optimal_noise
from nlica.defaults import RESULTS_FOLDER

N_JOBS = 15

# loop over
generations = ["gaussian_mean", "gaussian_var", "gaussian_corr"]
n_comps = [1, 1, 2]
data_vals = {
    "gaussian_var": [1.],                           # vals for data variance
    "gaussian_corr": [0., 0.3, 0.8],     # vals for data correlation
    "gaussian_mean": [0.]     # vals for data mean
}
prop_noises = [0.1, 0.5, 0.9]
noise_bins = 20

generations = generations[:-1]
n_comps = n_comps[:-1]


# define wrapper inside loop
def run_optimal_noise_and_save(
    estimation, generation,
    n_comp, n_samples, seed,
    prop_noise, verbose, data_val,
    noise_bins, n_epochs, n_samples_data_montecarlo
):
    out = run_optimal_noise(
        estimation=estimation,
        generation=generation,
        n_comp=n_comp, n_samples=n_samples,
        n_samples_data_montecarlo=n_samples_data_montecarlo,
        seed=seed,
        prop_noise=prop_noise, verbose=verbose,
        noise_bins=noise_bins, data_val=data_val,
        n_epochs=n_epochs
    )
    results_folder = RESULTS_FOLDER / "optimal_noise" / generation
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = results_folder / f"expe_optimal_noise_data_val_{data_val}_prop_noise_{prop_noise}_newer.th"
    torch.save(out, filename)


# Run things in parallel with cartesian product on all parameters
# results is a list of dictionaries, each a line for a dataframe

results = Parallel(n_jobs=N_JOBS, verbose=1, backend='loky')(
    delayed(run_optimal_noise_and_save)(
        estimation='singlence', generation=generation,
        n_comp=n_comp, n_samples=10000, seed=0, n_samples_data_montecarlo=10000,
        prop_noise=prop_noise, verbose=True, data_val=data_val,
        noise_bins=noise_bins, n_epochs=30)
    for (generation, n_comp), prop_noise in product(zip(generations, n_comps), prop_noises)
    for data_val in data_vals[generation]
)

"""Test script for the run_model function of the runner script.
"""
import time
import torch
import pytest

from nlica.runner import run_model
from nlica.defaults import DEFAULTS
from nlica.utils import get_name_from_dict


_models_params = [
    ("gaussian_var", 1, 0.5),
    ("gaussian_corr", 2, 0.)
]


@pytest.mark.parametrize(
    "estimation",
    ["singlemle", "singlence"]
)
@pytest.mark.parametrize(
    "generation,n_comp,noise_val", _models_params
)
def test_convergence(estimation, generation, n_comp, noise_val, verbose=False):
    """Check that estimation methods (singlemle, singlence) recover
    the true data parameter."""
    # get default arguments
    kwargs = DEFAULTS.copy()

    # overwrite with specified arguments
    kwargs.update({
        "estimation": estimation,
        "generation": generation,
        "n_comp": n_comp,
        #
        "n_samples": 4000,
        "train_data": True,
        "noise_type": "likedata",
        "noise_val": noise_val,
        "noise_to_data": 1,
        "experiment": "sanity_check",
        "hue": generation,
        "predict_mse": False
    })

    # Rename output directory
    name = get_name_from_dict(kwargs)
    kwargs['out'] = kwargs['out'] / kwargs['experiment'] / kwargs['estimation'] / kwargs['hue'] / name

    # Run estimation
    t0 = time.time()
    metrics = run_model(**kwargs)
    computation_time = time.time() - t0

    for param_true, param_estim in zip(metrics.test.data_marginal_truth_params,
                                       metrics.test.data_marginal_params):
        if verbose:
            print("param_true: ", param_true)
            print("param_estim: ", param_estim)
            print("computation time: ", computation_time)
        # Check dtype
        assert param_true.dtype == torch.float64, "Tensor does not have dtype float64."
        assert param_estim.dtype == torch.float64, "Tensor does not have dtype float64."
        assert torch.allclose(param_true, param_estim, atol=0.1), \
            "The estimated parameter does not match the truth (for given tolerance)."

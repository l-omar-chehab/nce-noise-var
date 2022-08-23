"""Test script for the main script.
"""
import time
import sys
from io import StringIO
import numpy as np
import torch
import pytest

from main import (
    get_parser,
    run_model
)


_models_params = [
    ("gaussian_var", 1),
    ("gaussian_corr", 2),
    ("ica_rot", 2)
]


class ClosingStringIO(StringIO):
    """StringIO that closes after getvalue()."""

    def getvalue(self, close=True):
        """Get the value."""
        out = super().getvalue()
        if close:
            self.close()
        return out


class ArgvSetter(object):
    """Temporarily set sys.argv."""

    def __init__(self, args=(), disable_stdout=True,
                 disable_stderr=True):  # noqa: D102
        self.argv = list(('python',) + args)
        self.stdout = ClosingStringIO() if disable_stdout else sys.stdout
        self.stderr = ClosingStringIO() if disable_stderr else sys.stderr

    def __enter__(self):  # noqa: D105
        self.orig_argv = sys.argv
        sys.argv = self.argv
        self.orig_stdout = sys.stdout
        sys.stdout = self.stdout
        self.orig_stderr = sys.stderr
        sys.stderr = self.stderr
        return self

    def __exit__(self, *args):  # noqa: D105
        sys.argv = self.orig_argv
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr


@pytest.mark.parametrize(
    "estimation",
    ["singlemle", "singlence"]
)
@pytest.mark.parametrize(
    "generation,n_comp", _models_params
)
def test_main_convergence(estimation, generation, n_comp):
    """Check that estimation methods (singlemle, singlence) recover
    the true data parameter."""
    # get default parser values

    with ArgvSetter((
            '--estimation', estimation,
            '--n-samples', "4000",
            '--generation', generation,
            '--n-comp', str(n_comp),
            '--noise-type', "likedata",
            '--noise-scale', "0.5",
            '--noise-to-data', "1",
            '--train-data',
            '--experiment', "sanity_check",
            '--hue', generation,
    )):
        parser = get_parser()
        args = parser.parse_args()

        # collect results
        start = time.time()
        metrics = run_model(args)

        end = time.time()
        duration = np.round(end - start, 2)

        for param_name in metrics.test.data_marginal_truth.keys():
            param_true = metrics.test.data_marginal_truth[param_name]
            param_estim = metrics.test.data_marginal[param_name]
            print("param_true: ", param_true)
            print("param_estim: ", param_estim)
            print("duration: ", duration)
            # Check dtype
            assert param_true.dtype == torch.float64, "Tensor does not have dtype float64."
            assert param_estim.dtype == torch.float64, "Tensor does not have dtype float64."
            # TODO: for ica rot, allow difference of 2 pi between angles
            assert torch.allclose(param_true, param_estim, atol=0.1), \
                f"The {args.estimation} estimated parameter does not match the truth (for given tolerance)."


# @pytest.mark.slowtest  # cpu-intensive test
# @pytest.mark.parametrize(
#     "estimation", ["singlemle", "singlence"]
# )
# @pytest.mark.parametrize(
#     "generation,n_comp", _models_params
# )
# def test_mse_prediction(estimation, generation, n_comp):

#     print("WARNING: this is CPU-intensive!")

#     with ArgvSetter():
#         mse_empirical, robustmse_empirical = get_mse_empirical(
#             estimation=estimation, generation=generation,
#             n_comp=n_comp, n_samples=10_000,
#             prop_noise=0.5, noise_scale=0.5,
#             n_seeds=200, n_workers=2
#         )

#         mse_predicted = get_mse_predicted(
#             estimation=estimation, generation=generation,
#             n_comp=n_comp, n_samples=10_000,
#             prop_noise=0.5, noise_scale=0.5,
#             seed=0
#         )

#         # take log scale
#         logmse_empirical = torch.log(mse_empirical)
#         logrobustmse_empirical = torch.log(robustmse_empirical)
#         logmse_predicted = torch.log(mse_predicted)

#         print("logmse_empirical: ", logmse_empirical)
#         print("logrobustmse_empirical: ", logrobustmse_empirical)
#         print("logmse_predicted: ", logmse_predicted)
#         assert torch.allclose(logmse_empirical, logmse_predicted, atol=0.2), \
#             "The empirical and predicted (asymptotic) MSE do not match for " \
#             f"the generative model {generation} estimated with {estimation}."

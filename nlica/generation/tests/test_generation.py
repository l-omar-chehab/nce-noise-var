"""Test script for generative models.
"""
import pytest

import torch
from torch.distributions import MultivariateNormal as TorchMultivariateNormal
from scipy.stats import pearsonr

from nlica.generation.singleparam.gaussian_var import get_model as get_model_gaussian_var
from nlica.generation.singleparam.ica_rot import get_model as get_model_ica_rot
from nlica.generation.singleparam.gaussian_corr import get_model as get_model_gaussian_corr


# Tests over all generative models

_models_params = [
    (get_model_gaussian_var, 1),
    (get_model_gaussian_corr, 2),
    (get_model_ica_rot, 2)
]


@pytest.mark.parametrize(
    "get_model,n_comp", _models_params
)
def test_seeds(get_model, n_comp):
    """General sanity check for any generative model."""
    n_samples = 10000

    # Check 1 : same seeds yields same result
    out = get_model(n_samples=n_samples, seed=0, n_comp=n_comp)
    data_marginal_1, source_1 = out  # unpack
    obs_1 = data_marginal_1.generator_true(source_1)

    out = get_model(n_samples=n_samples, seed=0, n_comp=n_comp)
    data_marginal_2, source_2 = out  # unpack
    obs_2 = data_marginal_2.generator_true(source_2)

    assert torch.allclose(source_1, source_2), \
        "The same seed does not produce the same sources for " + str(get_model)
    assert torch.allclose(obs_1, obs_2), \
        "The same seed does not produce the same observations for " + + str(get_model)

    # Check 2 : different seeds yields different results
    out = get_model(n_samples=n_samples, seed=1, n_comp=n_comp)
    data_marginal_3, source_3 = out  # unpack
    obs_3 = data_marginal_3.generator_true(source_3)

    assert not torch.allclose(source_1, source_3), \
        "Different seeds do not produce different sources for " + str(get_model)
    assert not torch.allclose(obs_1, obs_3), \
        "Different seeds do not produce different observations for " + str(get_model)

    # Check 3 : Tensors have type float64
    assert source_1.dtype == torch.float64, "Tensor does not have dtype float64."
    assert source_2.dtype == torch.float64, "Tensor does not have dtype float64."
    assert source_3.dtype == torch.float64, "Tensor does not have dtype float64."
    for p in data_marginal_1.parameters():
        assert p.dtype == torch.float64, "Tensor does not have dtype float64."


# Tests over individual generative models

def test_gaussian_var():
    n_samples, n_comp = 1000, 1

    # Instantiate generative model
    data_marginal, _ = get_model_gaussian_var(n_samples=n_samples, seed=0, n_comp=n_comp)

    # Instantiate torch multivariate normal with same parameters
    loc = data_marginal.prior_marginal.loc
    covariance_matrix = data_marginal.prior_marginal.covariance_matrix

    torchnormal = TorchMultivariateNormal(
        loc=loc,
        covariance_matrix=covariance_matrix
    )

    # Check type
    for p in data_marginal.parameters():
        assert p.dtype == torch.float64, "Tensor does not have dtype float64."

    # Draw samples
    samples = data_marginal.sample((n_samples,))

    # Check that it matches results with a Pytorch Standard Multivariate Normal
    logprob = data_marginal.log_prob(samples)
    logprob_torch = torchnormal.log_prob(samples)
    assert torch.allclose(logprob, logprob_torch), \
        "Gaussian (var) generative model does not evaluate log_prob as does Pytorch."

    # Check that the only Parameter is the variance
    named_params = list(data_marginal.named_parameters())
    assert (len(named_params) == 1), \
        "Gaussian (var) generative model has more than one trainable parameter."
    assert (named_params[0][1].shape == torch.Size([])), \
        "Gaussian (var) generative model should have a scalar trainable parameter."
    assert (named_params[0][0] == "prior_marginal.cov_diag"), \
        "Gaussian (var) generative model has a trainable parameter that is not named cov_diag."


def test_gaussian_corr():
    n_samples, n_comp = 1000, 2

    # Instantiate generative model
    out = get_model_gaussian_corr(n_samples=n_samples, seed=0, n_comp=n_comp)
    data_marginal, source = out  # unpack

    # Instantiate torch multivariate normal with same parameters
    loc = data_marginal.prior_marginal.loc
    covariance_matrix = data_marginal.prior_marginal.covariance_matrix

    torchnormal = TorchMultivariateNormal(
        loc=loc,
        covariance_matrix=covariance_matrix
    )

    # Check type
    for p in data_marginal.parameters():
        assert p.dtype == torch.float64, "Tensor does not have dtype float64."

    # Draw samples
    samples = data_marginal.sample((n_samples,))

    # Check that it matches results with a Pytorch Standard Multivariate Normal
    logprob = data_marginal.log_prob(samples)
    logprob_torch = torchnormal.log_prob(samples)
    assert torch.allclose(logprob, logprob_torch), \
        "Gaussian (var) generative model does not evaluate log_prob as does Pytorch."

    # Check that the only Parameter is the variance
    named_params = list(data_marginal.named_parameters())
    assert (len(named_params) == 1), \
        "Gaussian (corr) generative model has more than one trainable parameter."
    assert (named_params[0][1].shape == torch.Size([])), \
        "Gaussian (corr) generative model should have a scalar trainable parameter."
    assert (named_params[0][0] == "prior_marginal.cov_offdiag"), \
        "Gaussian (corr) generative model has a trainable parameter that is not named cov_offdiag."


def test_ica_rot():
    n_samples, n_comp = 10000, 2

    # Instantiate generative model
    out = get_model_ica_rot(n_samples=n_samples, seed=0, n_comp=n_comp)
    data_marginal, source = out  # unpack
    obs = data_marginal.generator_true(source)

    # Check type
    for p in data_marginal.parameters():
        assert p.dtype == torch.float64, "Tensor does not have dtype float64."

    # Check that sources are independent (via PearsonR proxy)
    correlations = []
    for comp_1 in range(n_comp):
        for comp_2 in range(comp_1):
            samples_1 = source[..., comp_1].flatten().numpy()
            samples_2 = source[..., comp_2].flatten().numpy()
            corr = pearsonr(samples_1, samples_2)[0]
            correlations.append(corr)
    correlations = torch.Tensor(correlations)

    assert torch.allclose(correlations, torch.zeros_like(correlations), atol=1e-1), \
        "Some dimensions of the ICA model source are linearly correlated."

    # Check that observations are not independent (via PearsonR proxy)
    correlations = []
    for comp_1 in range(n_comp):
        for comp_2 in range(comp_1):
            samples_1 = obs[..., comp_1].flatten().numpy()
            samples_2 = obs[..., comp_2].flatten().numpy()
            corr = pearsonr(samples_1, samples_2)[0]
            correlations.append(corr)
    correlations = torch.Tensor(correlations)

    assert ~torch.allclose(correlations, torch.zeros_like(correlations), atol=1e-2), \
        "All dimensions of the ICA model observations are linearly decorrelated."

    # Check that the only Parameter is the embedding rotation angle
    named_params = list(data_marginal.named_parameters())
    assert (len(named_params) == 1), \
        "Orthogonal ICA generative model has more than one trainable parameter."
    assert (named_params[0][1].shape == torch.Size([])), \
        "Orthogonal ICA generative model should have a scalar trainable parameter."
    assert (named_params[0][0] == "embedder.angle"), \
        "Orthogonal ICA generative model has a trainable parameter that is not named angle."

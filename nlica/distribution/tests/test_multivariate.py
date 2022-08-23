"""Test script for multivariate.py
Distributions used in this project follow the format of torch.distributions,
but wrapped within a torch.nn.Module.
"""

import pytest

import numpy as np
from scipy.stats import pearsonr

from nlica.distribution.multivariate import MultivariateNormal, MultivariateLaplace
from nlica.distribution.multivariate import Histogram
from nlica.distribution.base import BaseFlow
from nlica.utils import hasmeth

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal as TorchMultivariateNormal
from torch import optim
import time

torch.set_default_dtype(torch.float64)


# Tests over all distributions

@pytest.mark.parametrize(
    "n_comp", [1, 2, 3, 5],
)
@pytest.mark.parametrize(
    "n_samples", [1, 10, 100]
)
@pytest.mark.parametrize(
    "Distribution",
    [MultivariateNormal, MultivariateLaplace, Histogram, TorchMultivariateNormal]
)
def test_dimensions(n_samples, n_comp, Distribution):
    """Tests dimensionality of samples according to pytorch API.
    """
    # Instantiate distributions
    if ("Histogram" in Distribution.__name__) or ("Laplace" in Distribution.__name__):
        distribution = Distribution(n_comp=n_comp)
    elif "Normal" in Distribution.__name__:
        distribution = Distribution(loc=torch.zeros(n_comp), covariance_matrix=torch.eye(n_comp))

    # Sampling
    samples = distribution.sample((n_samples,))
    assert (samples.shape == (n_samples, n_comp)), \
        f"Samples of {distribution} do not have expected shape ({n_samples}, {n_comp})."

    # Log-Probability
    logprob = distribution.log_prob(samples)
    assert (logprob.shape == (n_samples,)), \
        f"Log-probabilities of {distribution} do not have expected shape ({n_samples},)."

    # Check type
    assert samples.dtype == torch.float64, "Samples do not have dtype float64."
    assert logprob.dtype == torch.float64, "Log-probabilities does not have dtype float64."
    if hasmeth(distribution, "parameters"):
        for p in distribution.parameters():
            assert p.dtype == torch.float64, "A Parameter does not have dtype float64."


@pytest.mark.parametrize(
    "Distribution", [MultivariateNormal, MultivariateLaplace, Histogram]
)
def test_autograd(Distribution):
    """Tests that autograd does work for:
        -- 'log_prob' method
        -- 'score' method
    and does not work for:
        -- 'sample' method
    """
    n_samples, n_comp = 100, 2
    # Instantiate distributions
    if ("Histogram" in Distribution.__name__) or ("Laplace" in Distribution.__name__):
        distribution = Distribution(n_comp=n_comp)
    else:
        distribution = Distribution(
            loc=nn.Parameter(torch.zeros(n_comp), requires_grad=True),
            covariance_matrix=nn.Parameter(torch.eye(n_comp), requires_grad=True)
        )

    # Sampling
    distribution.zero_grad()
    samples = distribution.sample((n_samples,))
    loss = samples.mean()
    assert ~loss.requires_grad, f"Gradients are backpropagated to parameters for {distribution} 'sample' method."

    # Log-Probability
    distribution.zero_grad()
    logprob = distribution.log_prob(samples)
    loss = logprob.mean()
    loss.backward()
    is_grad = torch.prod(
        torch.Tensor([p.grad is not None for p in distribution.parameters()])
    )
    assert is_grad, f"Gradients are not backpropagated to parameters for {distribution} 'log_prob' method."

    # Score
    distribution.zero_grad()
    score = distribution.score(samples)
    loss = score.mean()
    loss.backward()
    is_grad = torch.prod(
        torch.Tensor([p.grad is not None for p in distribution.parameters()])
    )
    assert is_grad, f"Gradients are not backpropagated to parameters for {distribution} 'score' method."
    assert (len(list(distribution.named_parameters())) != 0), \
        "Score method leaves parameter list empty."


def test_flow():
    """Test if the Flow object is able to output without an error message.
    """
    # prior marginal is gaussian
    prior_marginal = MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2))

    # embedder is identity
    embedder = nn.Linear(2, 2, bias=None)
    embedder.weight.data = torch.eye(2)

    # generator is identity
    generator = nn.Linear(2, 2, bias=None)
    generator.weight.data = torch.eye(2)

    # build flow
    flow = BaseFlow(
        prior_marginal=prior_marginal,
        embedder=embedder,
        generator_true=generator
    )

    # check log-probabilities
    samples = flow.sample((100,))
    logprob_prior = prior_marginal.log_prob(samples)
    logprob_flow = flow.log_prob(samples)
    assert torch.allclose(logprob_flow, logprob_prior), \
           "Log-probabilities are different between a prior and an identity_flow."


# Tests over individual distributions

def test_multivariate_normal():
    """Tests values of outputs,
    between Normal (ours)
    and MultivariateNormal (torch.distributions).
    """
    # Define trainable parameters
    mean = nn.Parameter(torch.zeros(2), requires_grad=True)
    cov = nn.Parameter(torch.eye(2), requires_grad=True)

    # Instantiate distributions (ours vs. torch)
    normal = MultivariateNormal(loc=mean, covariance_matrix=cov)
    normal_torch = TorchMultivariateNormal(loc=mean, covariance_matrix=cov)

    samples = normal.sample((100,))

    # Log-Probability: check values
    logprob = normal.log_prob(samples)
    logprob_torch = normal_torch.log_prob(samples)
    assert torch.allclose(logprob, logprob_torch), \
           "Log-Probabilities have different shapes depending on whether they are computed via " \
           "Normal (ours) or MultivariateNormal (torch)."

    # Score: check values
    score = normal.score(samples)
    score_analytical = normal._score_analytical(samples)
    assert torch.allclose(score, score_analytical), \
           "Log-Probabilities have different shapes depending on whether they are computed via " \
           "autograd or analytically."


def test_multivariate_laplace():
    """Tests independence across dimensions.
    """
    n_comp = 3

    # Instantiate MultivariateLaplace
    laplace = MultivariateLaplace(n_comp=n_comp)

    # Draw samples
    samples = laplace.sample((10000,))

    # Compute correlations between sample components
    correlations = []
    for comp_1 in range(n_comp):
        for comp_2 in range(comp_1):
            samples_1 = samples[..., comp_1].flatten().numpy()
            samples_2 = samples[..., comp_2].flatten().numpy()
            corr = pearsonr(samples_1, samples_2)[0]
            correlations.append(corr)
    correlations = torch.Tensor(correlations)

    # Check the correlations are all null (as approx proxy for independence)
    assert torch.allclose(correlations, torch.zeros_like(correlations), atol=1e-1), \
        "Some sample dimensions of MultivariateLaplace are linearly correlated."

    # Check that samples per dimension have loc zero and scale one
    # which translates as mean 0 and variance 2
    assert torch.allclose(samples.mean(0), torch.zeros(n_comp), atol=1e-1), \
        "Some sample dimensions have mean different than zero."
    assert torch.allclose(samples.var(0), 2. * torch.ones(n_comp), atol=1e-1), \
        "Some sample dimensions have variance different than two."


def test_histogram(verbose=False):
    """Tests that the histogram
    -- is able to handle samples outside its domain
    -- converges to a uniform
       when maximizing entropy.
    """
    # Instantiate histogram
    hist = Histogram(n_comp=2, low=-1, high=1, n_bins=2)

    # Initialize with random-normal weights over bins
    hist.masses = nn.Parameter(torch.randn(hist.n_intervals), requires_grad=True)

    # Evaluate sample from outside the support
    x = 2 * torch.ones((1, 2))  # (*sample_size, n_comp)
    try:
        hist.log_prob(x)
    except Exception as exception:
        assert False, f"The histogram cannot evaluate a sample from outside its support and throws: {exception}."

    # TestCase.assertFalse("The histogram cannot evaluate a sample from outside its domain." in context.exception)

    # Draw samples from a uniform
    # TODO : make this a standalone function
    samples = hist._sample_uniform(
        sample_size=1000,
        n_comp=2,
        low=-1,
        high=1
    )

    # Maximize likelihood of uniform samples
    n_epochs, lr = 200, 1e-1
    optimizer = optim.AdamW(hist.parameters(), lr=lr)

    for epoch in range(n_epochs):
        start = time.time()

        logp = hist.log_prob(samples)
        loss = -logp.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        end = time.time()
        duration = np.round(end - start, 2)
        print(f"epoch {epoch} took {duration} seconds")

    probas = hist.softmax(hist.masses)
    probas_truth = torch.ones_like(probas) / len(probas)
    assert torch.allclose(probas, probas_truth, atol=0.05), \
        "Histogram does not converge to a uniform when maximizing entropy."

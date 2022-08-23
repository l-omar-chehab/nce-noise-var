"""Quadrature for gaussian_mean generative model."""

import numpy as np
from scipy.integrate import quad as integrator
from scipy.stats import norm
# import bigfloat
from scipy.special import expit

from nlica.runner import mem

tol = 1e-8
verbose = False


# data distribution
def data_logpdf(x, param):
    """1D Normal with Mean parameter."""
    out = norm.logpdf(x, loc=param, scale=1.)
    return out


def _data_pdf(x, param):
    out = np.exp(data_logpdf(x, param=param))
    return out


def data_score(x, param):
    out = param - x
    return out


# data-dependent integrands
def _score_mean_integrand_mle(x, param_data):
    out = data_score(x, param=param_data) * _data_pdf(x, param=param_data)
    return out


def _score_var_integrand_mle(x, param_data):
    out = data_score(x, param=param_data)**2 * _data_pdf(x, param=param_data)
    return out


def _optimalnoise_unnormalized_integrand(x, param_data, fisher):
    out = _data_pdf(x, param=param_data) * np.abs(data_score(x, param=param_data) / fisher)
    return out


# data-dependent integrals
@mem.cache  # cache to avoid losing computations (checkpointing)
def _score_var_mle(param_data):
    # compute fisher information
    fisher, abserr = integrator(
        _score_var_integrand_mle,
        -np.inf, np.inf,
        args=(param_data,),
        epsabs=tol
    )
    if verbose:
        print(f"Integration: fisher information computed with error {abserr}")
    return fisher


@mem.cache  # cache to avoid losing computations (checkpointing)
def _optimalnoise_normalization(param_data):
    # required: fisher information
    fisher = _score_var_mle(param_data=param_data)

    # compute normalization constant of the optimal noise
    normalization, abserr = integrator(
        _optimalnoise_unnormalized_integrand,
        -np.inf, np.inf,
        args=(param_data, fisher),
        epsabs=tol
    )
    if verbose:
        print(f"Integration: optimal noise normalization computed with error {abserr}")

    return normalization


# noise distribution
def noise_logpdf(
    x,
    param_data,
    noise_type,
    param_noise,  # for noise_type == "data"
    fisher=None, normalization=None  # for noise_type == "optimal"
):
    """Noise distribution.
    noise_type = "data" uses the data distribution as noise.
    noise_type = "optimal" uses Pihlaja's distribution as noise.
    """
    if noise_type == "data":
        # out = norm.logpdf(x, loc=param_noise, scale=2.)
        out = data_logpdf(x, param=param_noise)
    elif noise_type == "optimal":
        out = data_logpdf(x, param=param_data) + \
            np.log(np.abs(data_score(x, param=param_data) / fisher)) - \
            np.log(normalization)
    return out


def _noise_pdf(
    x,
    param_data,
    noise_type,
    param_noise,
    fisher, normalization
):
    out = np.exp(noise_logpdf(
        x,
        param_data=param_data,
        noise_type=noise_type,
        param_noise=param_noise,
        fisher=fisher, normalization=normalization)
    )
    return out


# data and noise-dependent integrands
def noise_discrim(
    x, param_data,
    noise_type,
    param_noise,  # for noise_type == "data"
    fisher, normalization,  # for noise_type == "optimal"
    noise_to_data
):
    """Binary discriminator for the noise (class = 0).
    Uses the density ratio of noise and data.

    How to deal with overflow in exp:
    https://stackoverflow.com/questions/9559346/deal-with-overflow-in-exp-using-numpy
    """
    logpd = data_logpdf(x, param=param_data)
    logpn = noise_logpdf(
        x,
        param_data=param_data,
        noise_type=noise_type,
        param_noise=param_noise,
        fisher=fisher, normalization=normalization
    )
    lognu = np.log(noise_to_data)
    out = expit(lognu + logpn - logpd)
    return out


def _score_mean_integrand_nce(
    x,
    param_data,
    noise_type,
    param_noise,
    fisher, normalization,
    noise_to_data
):
    out = _score_mean_integrand_mle(x, param_data=param_data) * \
        noise_discrim(
            x,
            param_data=param_data,
            noise_type=noise_type,
            param_noise=param_noise,
            fisher=fisher, normalization=normalization,
            noise_to_data=noise_to_data
    )
    return out


def _score_var_integrand_nce(
    x,
    param_data,
    noise_type,
    param_noise,
    fisher, normalization,
    noise_to_data
):
    out = _score_var_integrand_mle(x, param_data=param_data) * \
        noise_discrim(
            x,
            param_data=param_data,
            noise_type=noise_type,
            param_noise=param_noise,
            fisher=fisher, normalization=normalization,
            noise_to_data=noise_to_data
    )
    return out


# data and noise-dependent integrals
@mem.cache  # cache to avoid losing computations (checkpointing)
def _score_mean_nce(
    param_data,
    noise_type,
    param_noise,
    noise_to_data
):
    if noise_type == 'optimal':
        # required: fisher information
        fisher = _score_var_mle(param_data=param_data)
        # required: normalization of the optimal noise
        normalization = _optimalnoise_normalization(param_data=param_data)
    elif noise_type == 'data':
        fisher, normalization = None, None

    # compute score mean
    score_mean_nce, abserr = integrator(
        _score_mean_integrand_nce,
        -np.inf, np.inf,
        args=(param_data, noise_type, param_noise, fisher, normalization, noise_to_data),
        epsabs=tol
    )
    if verbose:
        print(f"Integration: score mean (nce) computed with error {abserr}")
    return score_mean_nce


@mem.cache  # cache to avoid losing computations (checkpointing)
def _score_var_nce(
    param_data,
    noise_type,
    param_noise,
    noise_to_data
):
    # required: fisher information
    fisher = _score_var_mle(param_data=param_data)

    # required: normalization of the optimal noise
    normalization = _optimalnoise_normalization(param_data=param_data)

    # compute score mean
    score_var_nce, abserr = integrator(
        _score_var_integrand_nce,
        -np.inf, np.inf,
        args=(param_data, noise_type, param_noise, fisher, normalization, noise_to_data),
        epsabs=tol
    )
    if verbose:
        print(f"Integration: score var (nce) computed with error {abserr}")
    return score_var_nce


# asymptotic mse
@mem.cache  # cache to avoid losing computations (checkpointing)
def get_asymptotic_mse_mle(
    param_data,
    noise_to_data,  # new
    n_samples  # new
):
    prop_noise = noise_to_data / (1 + noise_to_data)
    n_samples_data = (1 - prop_noise) * n_samples

    """This corresponds to the Cramer-Rao lower bound."""
    # compute score covar
    score_var = _score_var_mle(param_data=param_data)

    asympvar = 1. / score_var

    mse = (1.0 / n_samples_data) * asympvar

    return mse


@mem.cache  # cache to avoid losing computations (checkpointing)
def get_asymptotic_mse_nce(
    param_data,
    noise_type,
    param_noise,
    noise_to_data,
    n_samples  # new
):
    prop_noise = noise_to_data / (1 + noise_to_data)
    n_samples_data = (1 - prop_noise) * n_samples

    # compute score mean
    score_mean = _score_mean_nce(
        param_data=param_data,
        noise_type=noise_type,
        param_noise=param_noise,
        noise_to_data=noise_to_data)

    # compute score covar
    score_var = _score_var_nce(
        param_data=param_data,
        noise_type=noise_type,
        param_noise=param_noise,
        noise_to_data=noise_to_data)

    inverse_score_var = 1. / score_var

    first_term = inverse_score_var
    second_term = (1 + (1.0 / noise_to_data)) * (inverse_score_var * score_mean)**2
    asympvar = first_term - second_term

    mse = (1.0 / n_samples_data) * asympvar

    return mse


def get_mse_quadrature(n_samples=100, prop_noise=0.5, param_data=0.,
                       param_noise=0.5, estimation='singlence',
                       noise_type='data'):

    noise_to_data = prop_noise / (1 - prop_noise)

    if estimation == 'singlence':
        out = get_asymptotic_mse_nce(
            param_data=param_data,
            noise_type=noise_type,
            param_noise=param_noise,
            noise_to_data=noise_to_data,
            n_samples=n_samples
        )
    if estimation == 'singlemle':
        out = get_asymptotic_mse_mle(
            param_data=param_data,
            noise_to_data=noise_to_data,
            n_samples=n_samples
        )

    return out

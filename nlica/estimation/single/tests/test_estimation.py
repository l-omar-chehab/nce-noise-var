"""Test script for the SingleNCE estimation.
"""
import torch
import pytest

from nlica.generation.singleparam.gaussian_var import get_model as get_model_gaussian_var
from nlica.generation.singleparam.ica_rot import get_model as get_model_ica_rot
from nlica.generation.singleparam.gaussian_corr import get_model as get_model_gaussian_corr
from nlica.estimation.single.singlence import Model as SingleNCE
from nlica.estimation.single.singlemle import Model as SingleMLE
from nlica.utils import freeze

# from nlica.utils import freeze

_models_params = [
    (SingleMLE, get_model_gaussian_var, 1, 1),
    (SingleMLE, get_model_gaussian_corr, 2, 1),
    (SingleMLE, get_model_ica_rot, 2, 1),
    (SingleNCE, get_model_gaussian_var, 1, 1),
    (SingleNCE, get_model_gaussian_corr, 2, 1),
    (SingleNCE, get_model_ica_rot, 2, 1)
]


def _check_model_params(model, n_param, do_freeze_data_marginal_truth):
    n_grads = sum([p.grad is not None for p in model.parameters()])

    # Check type
    for p in model.parameters():
        assert p.dtype == torch.float64, "Tensor does not have dtype float64."

    if do_freeze_data_marginal_truth:
        if isinstance(model, SingleMLE):
            assert n_grads == 0, \
                "When data_marginal_true is frozen, " \
                "there should be no gradients. "
        if isinstance(model, SingleNCE):
            assert n_grads == n_param, \
                "When data_marginal_true is frozen, " \
                "there should be gradients only for noise_marginal. "

    else:  # gradients should flow
        if isinstance(model, SingleMLE):
            assert n_grads == n_param, \
                "Gradients should be backpropagated " \
                "to data_marginal_true params only."
        if isinstance(model, SingleNCE):
            assert n_grads == 2 * n_param, \
                "Gradients should be backpropagated " \
                "to data_marginal_true and noise_marginal params only."


# Test SingleMLE
@pytest.mark.parametrize(
    "Estimator,get_model,n_comp,n_param", _models_params
)
@pytest.mark.parametrize(
    "do_freeze_data_marginal_truth", [True, False]
)
def test_autograd(Estimator, get_model, n_comp, n_param, do_freeze_data_marginal_truth):
    n_samples = 1000

    # Instantiate *true* data distribution
    data_marginal_truth, source = get_model(n_samples=n_samples, seed=0, n_comp=n_comp)
    obs = data_marginal_truth.generator_true(source)

    # Instantiate *trainable* data distribution
    data_marginal, _ = get_model(n_samples=n_samples, seed=0, n_comp=n_comp)

    # Instantiate noise distribution (equal to data distribution)
    if Estimator == SingleMLE:
        noise_marginal = None  # MLE does not require a noise distribution for contrast
    elif Estimator == SingleNCE:
        noise_marginal, _ = get_model(n_samples=n_samples, seed=0, n_comp=n_comp)

    # Comment : there are 3 x n_param until now

    # Instantiate SingleMLE estimation
    model = Estimator(
        n_comp=n_comp,
        data_marginal=data_marginal,
        data_marginal_truth=data_marginal_truth,
        noise_marginal=noise_marginal,
        noise_to_data=1
    )

    if do_freeze_data_marginal_truth:
        freeze(data_marginal_truth)

    # Test: 'predict_mse' method
    model.zero_grad()
    mse_predicted = model.predict_mse(x=obs, n_samples=n_samples)
    loss = mse_predicted.mean()
    loss.backward()

    _check_model_params(model, n_param, do_freeze_data_marginal_truth)

    if Estimator == SingleNCE:

        # Test: '_discriminator_optimal' method
        model.zero_grad()
        y_pred = model._discriminator_optimal(obs)
        loss = y_pred.mean()
        loss.backward()

        _check_model_params(model, n_param, do_freeze_data_marginal_truth)


# Test SingleNCE

_models_params = [
    (get_model_gaussian_var, 1),
    (get_model_gaussian_corr, 2),
    (get_model_ica_rot, 2)
]


@pytest.mark.parametrize(
    "get_model,n_comp", _models_params
)
def test_singlence(get_model, n_comp):
    n_samples = 1000

    # Instantiate *true* data distribution
    out = get_model(n_samples=n_samples, seed=0, n_comp=n_comp)
    data_marginal_true, source = out  # unpack
    obs = data_marginal_true.generator_true(source)

    # Instantiate *trainable* data distribution
    out = get_model(n_samples=n_samples, seed=0, n_comp=n_comp)
    data_marginal, _ = out  # unpack

    # Instantiate noise distribution (equal to data distribution)
    out = get_model(n_samples=n_samples, seed=0, n_comp=n_comp)
    noise_marginal, _ = out  # unpack

    # Comment : there are 3 x n_param until now

    # Instantiate SingleNCE estimation
    model = SingleNCE(
        n_comp=n_comp,
        data_marginal=data_marginal,
        data_marginal_truth=data_marginal_true,
        noise_marginal=noise_marginal,
        noise_to_data=1
    )

    # Test method: optimal_discriminator
    # we expect the output to be constant at 0.5
    # because here noise and data have same distributions and proportion
    softpred = model._discriminator_optimal(obs)
    softpred_theory = (1. / 2) * torch.ones(n_samples)
    assert torch.allclose(softpred, softpred_theory), \
        "The optimal discriminator of the SingleNCE estimation " \
        "does not always predict 0.5 when data = noise."

    # Check type
    assert softpred.dtype == torch.float64, "Tensor does not have dtype float64."
    assert softpred_theory.dtype == torch.float64, "Tensor does not have dtype float64."

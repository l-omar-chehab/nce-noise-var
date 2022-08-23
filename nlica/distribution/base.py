"""Classes to sample (from) and evaluate distributions.

Unlike Pytorch's 'distributions' class, this inherits from Pytorch's 'module' class (like neural networks)
allowing (unconstrained) variable of type 'Parameter' to be learnt.

References:
https://github.com/pytorch/pytorch/issues/31009
https://github.com/bayesiains/nflows/tree/master/nflows/distributions
"""

from nlica.utils import (
    extract_weights,
    load_weights,
    delete_weights,
    hasmeth
)

import torch
from torch import nn
from torch.autograd.functional import jacobian

from copy import deepcopy


# Multivariate distributions (factorial)

class BaseMultivariate(nn.Module):
    """Multivariate distribution class.
    Inspired from torch.distributions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        raise RuntimeError("forward method is not implemented for a BaseMultivariate object.")

    def sample(self, sample_shape=()):
        """Samples x from pdf(x).

        Parameters
        ----------
        sample_shape : tuple of iid realizations, e.g. (n_samples,)

        Returns
        -------
        samples : torch.tensor of shape (*sample_shape, n_comp)
                  Samples from the pdf.
        """
        # alternative to abstract method
        raise NotImplementedError("You must implement a sample method.")

    def log_prob(self, x):
        """Computes logpdf(x).

        Parameters
        ----------
        x : torch tensor of shape (*sample_shape, n_comp)

        Returns
        -------
        logprob : torch.tensor of shape (*sample_shape)
        """
        # alternative to abstract method
        raise NotImplementedError("You must implement a log_prob method.")

    def score(self, x):
        """Computes the gradient of logpdf(x) near its current
        parameters, e.g. mean, covariance for a Gaussian.

        WARNING bis:
        in the edge case of a single, scalar parameter, it is internally considered
        as an array so that we can perform vector calculus.

        Uses PyTorch's autograd to do automatic differentiation.

        Parameters
        ----------
        x : torch tensor of shape (n_samples, n_comp)

        Returns
        -------
        score : torch tensor of shape (n_samples, n_params_flattened)

        Based on: https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240/5
        """
        # Extract weights (as params with requires_grad=True)
        weights, names = extract_weights(self, as_param=True)

        # Create deepcopy of the model
        # so I can delete / load weights without tampering with the original model
        model_copy = deepcopy(self)

        # Delete parameters of model copy
        delete_weights(model_copy)

        def eval_log_prob(*new_weights):
            # Replace parameters with tensors
            load_weights(model_copy, names, new_weights)
            out = model_copy.log_prob(x)
            return out

        # Use autograd w.r.t. tensors
        score = jacobian(func=eval_log_prob, inputs=tuple(weights), create_graph=True)
        # tuple of tensors, each of size (dim_ouput, *dims_param)
        score = [s.flatten(start_dim=1) if len(s.shape) > 1 else s
                 for s in score]
        score = torch.cat(score, dim=-1)  # (dim_output, dims_param_flattened)
        if len(score.shape) == 1:  # flat, there is a unique scalar parameter
            score = score.reshape(-1, 1)

        # Reset parameters
        load_weights(self, names, weights)

        return score


class BaseFlow(BaseMultivariate):
    """Flow distribution class.
    Inspired from https://github.com/bayesiains/nflows/blob/master/nflows/flows/base.py
    """
    def __init__(self, prior_marginal, embedder, generator_true):
        super().__init__()
        self.prior_marginal = prior_marginal
        self.embedder = embedder
        self.generator_true = generator_true

    def forward(self, *args):
        raise RuntimeError("forward method is not implemented for a BaseFlow object.")

    def sample(self, sample_shape=()):
        assert (self.generator_true is not None), "Cannot sample if generator_true is None."
        s = self.prior_marginal.sample(sample_shape)
        x = self.generator_true(s)
        return x

    def log_prob(self, x):
        s = self.embedder(x)

        logmarginal = self.prior_marginal.log_prob(s)   # (B,)

        if hasmeth(self.embedder, "get_weight"):
            logdet = self.embedder.get_weight().det().abs().log()   # scalar tensor
        else:
            logdet = self.embedder.weight.det().abs().log()   # scalar tensor

        logprob = logmarginal + logdet

        return logprob

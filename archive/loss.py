'''Functions that evaluate downstream loss.'''
import torch
from torch.distributions.cauchy import Cauchy

from nlica.distributions import Factorial


def loss_mle(model, x, eval=True):
    """Computes the maximum-likelihood loss for an Invertible DenseNet.
    This loss has 3 parts, as described in https://arxiv.org/abs/2006.15090
    Input:
        model (torch.nn.Module subclass): instance of a DenseNet
        type: default 'train', otherwise 'eval'.
              If 'train', loss 3 is not included,
              it is already in the relative grad.
        x (torch tensor): observation, of shape (B, C)
    Output:
        loss (torch scalar tensor): maximum-likelihood loss over a batch
        z (torch tensor): encoding, of shape (B, C)

    Remark on loss_2:
        `grad()` returns a tuple that we need to unpack:
        for each scalar output, returns output sum of (L, B, C) gradients
        w.r.t. inputs (i.e. Jacobian-vector product)
        here: the nonlinearity is applied pointwise over (L, B, C)
        so for a given output, its gradients w.r.t. inputs are all null except
        one so each output is indeed the pointwise gradient
    """
    # dim, device = x.shape[-1], x.device
    z = model(x)  # (B, C)
    loss = torch.tensor(0.0)  # autograd loss

    # part 1 : log prior (Cauchy)
    # TODO: harcoded atm
    base = Cauchy(loc=0, scale=1)
    prior = Factorial(distributions=[base] * 2)
    logprior = prior.log_prob(z)  # (B,)
    loss_1 = -logprior.mean()
    loss += loss_1

    # part 2 : log activation derivatives
    if model.encoder.n_layers > 1:
        ys = torch.stack(model.encoder.linear_outputs)  # (L, B, C)
        zs = model.encoder.activation(ys)  # (L, B, C)
        activation_grads = torch.autograd.grad(zs, ys,
                                               torch.ones_like(zs),
                                               create_graph=True)[0]  # (L, B, C)
        logactivdiff = torch.log(torch.abs(activation_grads)).sum(dim=(0, -1))  # (B,)
        loss_2 = -logactivdiff.mean()
        loss += loss_2
    else:
        loss_2 = torch.tensor(0.0)

    # part 3 : log determinant
    logabsdet = sum(
        [
            layer.weight.data.det().abs().log()  # without.data to track grad
            for layer in model.encoder.linear_layers
        ]
    )
    loss_3 = -logabsdet

    return loss, loss_1, loss_2, loss_3, z


def hsic(X):
    '''Computes HS Independence Criterion between features.

    This corresponds to the MMD(joint, product-of-marginals):
    it is null when there is total independence.

    Refs:
    https://papers.nips.cc/paper/2013/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf
    http://www.cmap.polytechnique.fr/~zoltan.szabo/talks/invited_talk/Zoltan_Szabo_invited_talk_Rennes_Statistical_Seminar_26_10_2018_slides.pdf

    Args:
        X (torch.Tensor): shape (n_samples, n_features)

    Returns:
        (scalar): measure of dependency
    '''
    n_sample, n_feat = X.shape

    # # linear kernel
    # def kernel(a, b):
    #     return a * b

    # gaussian kernel
    def kernel(a, b):
        scale = 0.2
        return torch.exp(-((a - b) ** 2) / (2 * scale ** 2))

    K = torch.stack(
        [kernel(*torch.meshgrid(X[:, feat], X[:, feat])) for feat in range(n_feat)],
        axis=-1,
    )

    scale_1 = 1.0 / n_sample ** 2
    scale_2 = 1.0 / n_sample ** (n_feat + 1)
    scale_3 = 1.0 / n_sample ** (2 * n_feat)

    # scale every coefficient with the weight so the sum does not explode
    term_1 = (scale_1 ** (1 / n_feat) * K).prod(axis=2).sum(axis=1).sum(axis=0)
    term_2 = (scale_2 ** (1 / n_feat) * K).sum(axis=1).prod(axis=-1).sum(axis=0)
    term_3 = (scale_3 ** (1 / n_feat) * K).sum(axis=0).sum(axis=0).prod(axis=-1)

    hsic = 1.0 * term_1 - 2.0 * term_2 + 1.0 * term_3

    return hsic

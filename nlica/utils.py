"""Useful functions."""

import numpy as np
import argparse
import functools
import torch
from torch import nn
import shutil
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from nlica.defaults import DEFAULTS


# Generic functions
def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)


def is_numpy(x):
    return type(x).__module__ == np.__name__


def set_torch_seed(seed):
    """Set torch's seed."""
    # THE FOLLOWING WILL CAUSE AN ISSUE WITH MULTIPROCESSING...!!
    # if torch.cuda.is_available():
    #   torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def hasmeth(instance, method):
    """Checks if a module has a given method.
    Inspiration: the hasattr function."""
    out = getattr(instance, method, None)
    return callable(out)  # boolean


# to compute nested attributes
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# Generic functions for Pytorch neural networks


def freeze(net):
    """Freezes the parameters (weights and attributes)
    of a Pytorch Neural Network."""
    for parameter in net.parameters():
        parameter.requires_grad = False


def unfreeze(net):
    """Unfreezes the parameters (weights and attributes)
    of a Pytorch Neural Network."""
    for parameter in net.parameters():
        parameter.requires_grad = True


def initialize(net, seed=None, init="default"):
    """Initializes a Pytorch Neural Network,
    sampling the weight matrices from a given distribution (e.g. Uniform)
    and setting the bias vectors to zero."""

    # fix seed
    if seed is not None:
        set_torch_seed(seed)

    # helper function: applies to a submodule of the network
    def initialize_module(m, init="default"):
        # if isinstance(m, nn.Bilinear):
        if init == "default":
            if hasmeth(m, "reset_parameters"):
                m.reset_parameters()
        else:
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight"):
                if init == "normal":
                    nn.init.kaiming_normal_(m.weight)
                elif init == "orthogonal":
                    nn.init.orthogonal_(m.weight)  # , gain=4)
                elif init == "identity":
                    nn.init.eye_(m.weight)
                elif init == "uniform":
                    nn.init.xavier_uniform_(m.weight)

    # freeze other args of the helper function
    def initialize_module_partial(m):
        return initialize_module(m, init=init)

    # func = lambda m: reset_module_weights(m, init=init)

    # apply method: the helper function applies recursively over submodules
    # of the network
    net.apply(initialize_module_partial)


def collect_weights(net):
    """Collects the weight matrices of a Pytorch Neural Network,
    in no particular order."""
    weights = list()

    def collect_weights_module(module):
        if hasattr(module, "weight"):
            weights.append(module.weight.data)

    net.apply(collect_weights_module)

    return weights


def collect_biases(net):
    """Collects the bias vectors of a Pytorch Neural Network,
    in no particular order."""
    biases = list()

    def get_biases_module(module):
        if hasattr(module, "bias"):
            biases.append(module.bias.data)

    net.apply(get_biases_module)

    return biases


def single_one_hot_encode(label, n_labels=10):
    """
    Transforms numeric labels to 1-hot encoded labels.
    Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """
    assert label >= 0 and label < n_labels, "label is out of range"
    y = torch.zeros(n_labels).float()
    y[label] = 1

    return y


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def cross_correl(A, B):
    """Computes correlation matrix between A and B.
    Args:
        A (np.ndarray): shape (n_samples, n_dims_A)
        B (np.ndarray): shape (n_samples, n_dims_B)

    Returns:
        (np.ndarray): correlation matrix of shape (n_dims_A, n_dims_B)
    """
    n_samples = A.shape[0]

    # center
    A = A - A.mean(0)
    B = B - B.mean(0)

    # scale
    A = A / A.std(0)
    B = B / B.std(0)

    return A.T @ B / n_samples


def compute_correl_score(sources_true, sources_estim):
    """Transforms sources_estim using permutation and sign change
    to better match sources_true.
    Args:
        sources_true (torch.Tensor): shape (n_samples, n_sources)
        sources_estim (torch.Tensor): shape (n_samples, n_sources)

    Returns:
        scores (np.array): pearson r recovery sources
        sources_estim (torch.Tensor): transformed estimated sources
    """
    # Torch to numpy
    sources_true = sources_true.data.numpy()
    sources_estim = sources_estim.data.numpy()

    # Find permutation
    corr_mat = cross_correl(sources_true, sources_estim)
    _, new_order = linear_sum_assignment(-np.abs(corr_mat))
    sources_estim = sources_estim[:, new_order]

    # Find sign change
    corr_mat = cross_correl(sources_true, sources_estim)
    sel = np.diag(corr_mat) < 0
    sources_estim[:, sel] = -sources_estim[:, sel]

    # Compute score after assignment
    corr_mat = cross_correl(sources_true, sources_estim)
    scores = np.diag(np.abs(corr_mat))

    return scores, torch.Tensor(sources_estim)


def correct_unmixing_matrix(unmixing, sources_true, sources_estim):
    """Corrects unmixing matrix for intractable indeterminacies
    of the ICA problem (permutation, sign, scale).
    Args:
        unmixing (torch.Tensor): shape (n_sources, n_sources)
        sources_true (torch.Tensor): shape (n_samples, n_sources)
        sources_estim (torch.Tensor): shape (n_samples, n_sources)

    The unmixing is such that source_estim = obs @ unmixing.

    Returns:
        scores (np.array): pearson r recovery sources
        sources_estim (torch.Tensor): transformed estimated sources
    """
    # Torch to numpy
    sources_true = sources_true.data.numpy()
    sources_estim = sources_estim.data.numpy()
    unmixing = unmixing.data.numpy().T

    # Find permutation
    corr_mat = cross_correl(sources_true, sources_estim)
    _, new_order = linear_sum_assignment(-np.abs(corr_mat))
    sources_estim = sources_estim[:, new_order]
    unmixing = unmixing[:, new_order]

    # Find sign change
    corr_mat = cross_correl(sources_true, sources_estim)
    sel = np.diag(corr_mat) < 0
    sources_estim[:, sel] = -sources_estim[:, sel]
    unmixing[:, sel] = -unmixing[:, sel]

    # Find scaling
    norms_estim = np.linalg.norm(sources_estim, axis=0, keepdims=True)  # (1, 4)
    norms_true = np.linalg.norm(sources_true, axis=0, keepdims=True)  # (1, 4)
    sources_estim = sources_estim / norms_estim * norms_true
    unmixing = unmixing / norms_estim * norms_true

    return torch.Tensor(unmixing.T)


def create_directory(path, overwrite=False):
    """
    Args:
        path (pathlib object)
    """
    # if it is there
    if path.exists():

        # overwrite: remove then recreate
        if overwrite:
            shutil.rmtree(path)

        # do not overwrite: record as bis
        else:
            # while path.exists():
            path = Path(str(path) + "_bis")

    path.mkdir(parents=True)

    return path


def get_classif_upperbound(
    dataset_train, dataset_test=None, n_estimators=500, max_depth=5, metrics=None,
    verbose=False
):
    """In practice, Random Forest performance is much more impacted by
    n_samples than max_depth (keep it at 10).
    To estimate the Bayes accuracy with a Random Forest, this means
    scaling n_samples by n_classes (therefore n_neg) for CPC."""
    # random forest as off-the-shelf nonlinear model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,  # 100
        max_depth=max_depth,
        random_state=0,  # ADDED
        n_jobs=20,
    )  # ADDED
    # parameters = {'max_depth':[5, 10]}
    # clf = GridSearchCV(rf, parameters)
    clf = rf

    X_train, X_test = torch_to_numpy(dataset_train.X), torch_to_numpy(dataset_test.X)
    y_train, y_test = (
        torch_to_numpy(dataset_train.y).astype(int),
        torch_to_numpy(dataset_test.y).astype(int),
    )
    if hasattr(dataset_train, "U"):
        U_train, U_test = torch_to_numpy(dataset_train.U), torch_to_numpy(
            dataset_test.U
        )
        X_train = np.concatenate([X_train, U_train], axis=-1)
        X_test = np.concatenate([X_test, U_test], axis=-1)
    n_train, n_test = len(dataset_train), len(dataset_test)

    # accuracy lower-bound (Chance)
    metrics.train.acc_lb = y_train.mean()
    metrics.test.acc_lb = y_test.mean()

    # flatten features if need be
    X_train, X_test = X_train.reshape(n_train, -1), X_test.reshape(n_test, -1)

    # classify
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # accuracy upper-bound (Random Forest)
    acc_train = accuracy_score(y_pred_train, y_train)
    acc_test = accuracy_score(y_pred_test, y_test)
    metrics.train.acc_ub = acc_train
    metrics.test.acc_ub = acc_test

    if verbose:
        print(f"Random Forest train acc: {acc_train}")
        print(f"Random Forest test acc: {acc_test}")

    return acc_train, acc_test


def torch_to_numpy(x):
    return x.data.cpu().numpy()


def gen_ortho(dim=3, seed=0):
    random_state = np.random.RandomState(seed)
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum()
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


def get_name_from_dict(kwargs, include=['estimation'], ignore=['out']):
    """Characterizes an experiment by its parameters
    given by a dictionary (e.g. from a parser).

    We used this to generate a folder specific to each experiment, where to save KPIs.
    Now instead, we use joblib memory to cache.

    Args:
        kwargs (dict): dictionary defining the experiment
        ignore (list of str): args to include in the summary name.
                              Otherwise will choose those will value
                              different from default.

    Returns:
        name (str): summary name of an experiment
    """
    parts = []
    # name_args = dict(args.__dict__)  # replace with kwargs

    kwargs = DEFAULTS.copy()

    # remove args to ignore
    for key in ignore:
        kwargs.pop(key, None)

    # start with names to include
    for key in include:
        value = kwargs[key]
        if isinstance(value, Path):
            value = value.name
        elif isinstance(value, list):
            value = ",".join(map(str, value))
        parts.append(f"{key}={value}")
        kwargs.pop(key, None)  # remove after using

    # then include names with values different than default
    for name, value in kwargs.items():
        if value != DEFAULTS[name]:
            if isinstance(value, Path):
                value = value.name
            elif isinstance(value, list):
                value = ",".join(map(str, value))
            parts.append(f"{name}={value}")

    if parts:
        name = " ".join(parts)
    else:
        name = "default"

    return name


def amari_distance(W, A):
    """
    Computes the Amari distance between two matrices W and A.
    It cancels when WA is a permutation and scale matrix.
    copied from https://github.com/pierreablin/picard/blob/master/picard/_tools.py

    Parameters
    ----------
    W : ndarray, shape (n_features, n_features)
        Input matrix
    A : ndarray, shape (n_features, n_features)
        Input matrix

    Returns
    -------
    d : float
        The Amari distance
    """
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)
    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs."""References:
"""https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240/5
https://github.com/pytorch/pytorch/blob/f65ab89eddc5cf449b94fdbddb4715fb8f57d2d6/benchmarks/functional_autograd_benchmark/vision_models.py#L19-L24
https://github.com/pytorch/pytorch/blob/f65ab89eddc5cf449b94fdbddb4715fb8f57d2d6/benchmarks/functional_autograd_benchmark/utils.py
"""


def _del_nested_attr(obj, names):
    """
    obj is a nn.Module
    names is a list of str
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj, names, value):
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def _get_nested_attr(obj, names):
    """
    Get the attribute specified by the given list of names to value.
    For example, to get the attribute obj.conv.weight,
    use _get_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return _get_nested_attr(getattr(obj, names[0]), names[1:])


def extract_weights(mod, as_param=True, requires_grad_only=False):
    """
    This function extracts all the Parameters from the model and
    return them as a tuple of tensors or Parameters as well as their
    original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    names, weights = [], []

    if requires_grad_only:
        for (name, param) in mod.named_parameters():
            if param.requires_grad:
                names.append(name)
                weight = param if as_param else param.detach().requires_grad_()
                weights.append(weight)
    else:
        for (name, param) in mod.named_parameters():
            names.append(name)
            weight = param if as_param else param.detach().requires_grad_()
            weights.append(weight)

    return weights, names


def delete_weights(mod, requires_grad_only=False):
    """
    This function removes all the Parameters from the model.
    """
    names, params = [], []

    if requires_grad_only:
        for (name, param) in mod.named_parameters():
            if param.requires_grad:
                names.append(name)
                params.append(param)
    else:
        for (name, param) in mod.named_parameters():
            names.append(name)
            params.append(param)

    # otherwise 'OrderedDict mutated during iteration'
    for (name, param) in zip(names, params):
        _del_nested_attr(mod, name.split("."))


def load_weights(mod, names, weights):
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    """
    for name, weight in zip(names, weights):
        _set_nested_attr(mod, name.split("."), weight)

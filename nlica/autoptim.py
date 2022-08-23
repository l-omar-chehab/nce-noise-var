import numpy as np
import torch

from scipy.optimize import minimize as minimize_

torch.set_default_dtype(torch.float64)


def _scipy_func(objective_function, x, shapes, args=()):
    """There are two additions to the original code:
    -- we modify the dtype of torch_vars list to torch Parameters (instead of Tensors)
    -- objective_function takes as input a torch_vars list that is not unpacked with *
    """
    # import ipdb; ipdb.set_trace()
    optim_vars = _split(x, shapes)

    torch_vars = [torch.tensor(var, requires_grad=True) for var in optim_vars]
    # Modify list to contain Parameters (instead of tensors)
    dtype = torch.get_default_dtype()
    for t in torch_vars:
        t.data = t.data.type(dtype)
    torch_vars = [torch.nn.Parameter(t) for t in torch_vars]

    del optim_vars

    obj = objective_function(torch_vars, *args)
    obj.backward(retain_graph=True)

    gradients = [var.grad.numpy() for var in torch_vars]
    g_vectorized, _ = _vectorize(gradients)
    return obj.item(), g_vectorized


def minimize(objective_function, optim_vars, args=(), **kwargs):
    """A wrapper to call scipy.optimize.minimize while computing the gradients
       using pytorch's auto-differentiation.
        Parameters
        ----------
        objective_function : callable
            The objective function to be minimized.
                ``fun(optim_vars, *args) -> float``
            or
                ``fun(*optim_vars, *args) -> float``
            where optim_vars is either a torch tensor or a list of torch
            tensors and `args` is a tuple of the fixed parameters needed to
            completely specify the function. The function should be written
            in pytorch.
        optim_vars : ndarray or list of ndarrays
            Initial guess.
        args : tuple, optional
            Extra arguments passed to the objective function.
        kwargs : dict, optional
            Extra arguments passed to scipy.optimize.minimize. See
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            for the full list of available keywords.
        Returns
        -------
        output : ndarray or list of ndarrays
            The solution, of same shape as the input `optim_vars`.
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.
        """
    # Convert args to torch
    args_torch = [torch.tensor(arg).double() for arg in args]

    # Check if there are bounds:
    bounds = kwargs.get('bounds')
    bounds_in_kwargs = bounds is not None

    # Convert input to a list if it is a single array
    if type(optim_vars) is np.ndarray:
        input_is_array = True
        optim_vars = (optim_vars,)
        if bounds_in_kwargs:
            bounds = (bounds,)
    else:
        input_is_array = False

    # Vectorize optimization variables
    x0, shapes = _vectorize(optim_vars)

    # Convert bounds to the correct format
    if bounds_in_kwargs:
        bounds = _convert_bounds(bounds, shapes)
        kwargs['bounds'] = bounds

    # Define the scipy optimized function and run scipy.minimize
    def func(x):
        return _scipy_func(objective_function, x, shapes, args_torch)
    res = minimize_(func, x0, jac=True, **kwargs)

    # Convert output to the input format
    output = _split(res['x'], shapes)
    if input_is_array:
        output = output[0]
    return output, res


def _convert_bounds(bounds, shapes):
    output_bounds = []
    for shape, bound in zip(shapes, bounds):
        # Check is the bound is already parsable by scipy.optimize
        b = bound[0]
        if isinstance(b, (list, tuple, np.ndarray)):
            output_bounds += bound
        else:
            output_bounds += [bound, ] * np.prod(shape)
    return output_bounds


def _vectorize(optim_vars):
    shapes = [var.shape for var in optim_vars]
    x = np.concatenate([var.ravel() for var in optim_vars])
    return x, shapes


def _split(x, shapes):
    """There are two additions to the original code:
    -- we enforce type int in the definition of x_split
    -- in the definition of optim_vars, we cover the case where a variable is a
       0-dimensional numpy array, using an if / else condition
    """
    x_split = np.split(x, np.cumsum([np.prod(shape) for shape in shapes[:-1]]).astype(int))
    optim_vars = [var.reshape(*shape) if len(shape) != 0 else var.squeeze()
                  for (var, shape) in zip(x_split, shapes)]
    return optim_vars

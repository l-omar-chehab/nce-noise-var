import time
from functools import partial

import numpy as np
# import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from joblib import Memory


from .defaults import DEFAULTS, CACHE_FOLDER
from .train import run_epoch as run_epoch_extended
from .train import Metrics, eval_downstream, callback_extended
from .autoptim import minimize
from .distribution.multivariate import MultivariateNormal, Histogram
from .utils import (
    get_classif_upperbound,
    extract_weights,
    delete_weights,
    freeze,
    load_weights,
    get_name_from_dict,
    set_torch_seed
)
# from nlica.vis import (
#     visualize_timeseries,
#     visualize_distribution,
#     plot_kpis
# )


torch.set_default_dtype(torch.float64)


N_SAMPLES_TEST = 4000
mem = Memory(CACHE_FOLDER)


def perturb_init_inplace(xlist, seed=None):
    '''xlist is a list of parameters'''
    # Control a factor of variation
    if seed is not None:
        set_torch_seed(seed)
    for x in xlist:
        # Define initialization noise
        init_noise_scale = 0.2
        init_noise = init_noise_scale * torch.randn_like(x.data)

        # Clamp unacceptable values (a bit barbaric)
        sel_too_big = init_noise > 0.5
        sel_too_small = init_noise < -0.5
        init_noise[sel_too_big] = 0.5
        init_noise[sel_too_small] = -0.5

        # Update the initial parameter
        x.data += init_noise  # add gaussian noise


def run_model(**kwargs):
    torch.autograd.set_detect_anomaly(True)

    # WARNING: hardcoded
    # nb samples used to generate observation is different from
    # nb samples used in the predict_mse formula
    prop_noise = kwargs['noise_to_data'] / (1 + kwargs['noise_to_data'])
    n_samples_data_test = int((1 - prop_noise) * N_SAMPLES_TEST)

    if kwargs['predict_mse'] or kwargs['cost'] == "asympmse":
        n_samples_data = kwargs['n_samples_data_montecarlo']
    else:
        n_samples_data = int((1 - prop_noise) * kwargs['n_samples'])

    # Device
    device = "cpu"
    if kwargs['verbose']:
        print("device: ", device)

    # create_directory(kwargs['out'], overwrite=True)
    kwargs['out'].mkdir(exist_ok=True, parents=True)

    metrics = Metrics(path=kwargs['out'], verbose=kwargs['verbose'])

    # Add parameters of the experiment to metrics
    metrics.train.experiment.update(kwargs)

    # Estimatation procedure
    if kwargs['verbose']:
        print('Using :', kwargs["estimation"])

    if kwargs["estimation"] == "singlence":
        from nlica.estimation.single.singlence import Model, supervised, make_dataset
    elif kwargs["estimation"] == "singlemle":
        from nlica.estimation.single.singlemle import Model, supervised, make_dataset
    run_epoch = partial(run_epoch_extended, supervised=supervised)

    # Generative model :
    if kwargs['verbose']:
        print('Using :', kwargs['generation'])
    if kwargs['generation'] == 'gaussian_mean':
        from nlica.generation.singleparam.gaussian_mean import get_model
    if kwargs['generation'] == 'gaussian_var':
        from nlica.generation.singleparam.gaussian_var import get_model
    elif kwargs['generation'] == 'gaussian_corr':
        from nlica.generation.singleparam.gaussian_corr import get_model

    data_marginal, source = get_model(
        n_comp=kwargs['n_comp'], n_samples=n_samples_data,
        seed=kwargs['seed_data_gen'], data_val=kwargs['data_val']
    )

    _, source_test = get_model(
        n_comp=kwargs['n_comp'], n_samples=n_samples_data_test,
        seed=kwargs['seed_data_testset'], data_val=kwargs['data_val']
    )

    data_marginal_truth, _ = get_model(
        n_comp=kwargs['n_comp'], n_samples=1,
        seed=kwargs['seed_data_gen'], data_val=kwargs['data_val']
    )

    obs = data_marginal.generator_true(source)
    obs_test = data_marginal.generator_true(source_test)

    # # Generative model : visualize sources and observations
    # if kwargs['with_plot']:
    #     visualize_timeseries(source, figpath=kwargs['out'] / "source_timeseries.pdf")
    #     visualize_distribution(source, figpath=kwargs['out'] / "source_distribution.pdf")
    #     plt.close("all")
    #     visualize_timeseries(obs, figpath=kwargs['out'] / "observation_timeseries.pdf")
    #     visualize_distribution(obs, figpath=kwargs['out'] / "observation_distribution.pdf")
    #     plt.close("all")

    # Choose contrastive noise

    # noise marginal
    if kwargs['noise_type'] == "gaussian":
        data_cov = torch.Tensor(np.cov(obs.data.numpy(), rowvar=False)).float()
        if kwargs['n_comp'] == 1:
            data_cov = data_cov * torch.eye(kwargs['n_comp'])  # use 2-dim tensor instead of 0-dim
        noise_marginal = MultivariateNormal(loc=torch.zeros(kwargs['n_comp']),
                                            covariance_matrix=data_cov)
        del data_cov  # not used afterwards
    elif kwargs['noise_type'] == "likedata":
        noise_marginal, _ = get_model(n_comp=kwargs['n_comp'], n_samples=1,
                                      seed=kwargs['seed_data_gen'], data_val=kwargs['data_val'])
        for param in noise_marginal.parameters():
            param.data = param.data * 0. + kwargs['noise_val']
    elif kwargs['noise_type'] == "flexible":
        noise_marginal = Histogram(n_bins=kwargs['noise_bins'], n_comp=kwargs['n_comp'])
    elif str(kwargs['noise_type'])[-3:] == ".th":  # a filename is entered
        temp = torch.load(kwargs['noise_type'])
        noise_marginal = Histogram(n_bins=temp['n_bins'], n_comp=temp['n_comp'])
        noise_marginal.load_state_dict(temp['noise_marginal_statedict'])

    # Formatting data (train/test datasets, dataloaders)
    metrics.train.experiment['n_samples'] = kwargs['n_samples']
    metrics.test.experiment['n_samples'] = N_SAMPLES_TEST
    metrics.train.experiment['n_samples_data'] = n_samples_data
    metrics.test.experiment['n_samples_data'] = n_samples_data_test

    # Estimation method (e.g. SingleMLE)
    model = Model(
        n_comp=kwargs['n_comp'],
        data_marginal_truth=data_marginal_truth,
        data_marginal=data_marginal,
        noise_marginal=noise_marginal,
        noise_to_data=kwargs['noise_to_data']
    )
    if kwargs['verbose']:
        print(model)

    # # Visualize estimated sources before training
    # if kwargs['with_plot']:
    #     source_init = data_marginal.embedder(obs)
    #     visualize_timeseries(source_init, figpath=kwargs['out'] / "source_init_timeseries.pdf")
    #     visualize_distribution(source_init, figpath=kwargs['out'] / "source_init_distribution.pdf")
    #     plt.close("all")

    # Freeze parameters you do not want to train
    freeze(data_marginal_truth)
    if not kwargs['train_data']:
        freeze(data_marginal)
    if not kwargs['train_noise']:
        freeze(noise_marginal)
        # add gaussian noise to the data approximation we will optimize,
        # given it is initialized with the truth
        if kwargs['perturb_init']:
            perturb_init_inplace(list(data_marginal.parameters()), seed=kwargs['seed_data_gen'])

    trainable_params_names = [
        name for (name, param) in model.named_parameters() if param.requires_grad
    ]
    trainable_params_sel = [
        param.requires_grad for (_, param) in model.named_parameters()
    ]
    trainable_params_init_numpy = [
        param.data.numpy() for (sel, param) in zip(trainable_params_sel, model.parameters())
        if sel
    ]
    if kwargs['verbose']:
        print("Optimized parameters: ", trainable_params_names)

    if not kwargs['predict_mse']:

        if kwargs['cost'] == 'default':

            dataset_train = make_dataset(
                obs, noise_to_data=kwargs['noise_to_data'],
                noise_marginal=noise_marginal, seed=kwargs['seed_dataset'])
            dataset_test = make_dataset(
                obs_test, noise_to_data=kwargs['noise_to_data'],
                noise_marginal=noise_marginal, seed=kwargs['seed_dataset'] + 1)

            dataloaded_train = DataLoader(
                dataset_train, batch_size=kwargs['batch_size'],  # len(dataset_train)
                shuffle=True)  # , num_workers=0)
            dataloaded_test = DataLoader(
                dataset_test, batch_size=kwargs['batch_size'],
                shuffle=False)

            # Estimate Pretext Task difficulty
            if supervised:
                if kwargs['verbose']:
                    print("Pretext Task : Random Forest... \n")
                acc_train, acc_test = get_classif_upperbound(
                    dataset_train, dataset_test, max_depth=10, metrics=metrics, verbose=kwargs['verbose']
                )

            # Optimizer
            callback = partial(
                callback_extended,
                model=model,
                dataloaded_train=dataloaded_train,
                dataloaded_test=dataloaded_test,
                run_epoch=run_epoch,
                eval_downstream=eval_downstream,
                metrics=metrics,
                downstream_obs=obs_test,
                downstream_source=source_test,
                mixing_net=data_marginal.generator_true
            )

            def compute_cost(torch_vars):
                '''This is based on autoptim v.2., based on pytorch's autograd.'''
                _, names = extract_weights(model, requires_grad_only=True)
                delete_weights(model, requires_grad_only=True)
                load_weights(model, names, torch_vars)
                model.zero_grad()

                loss_train = run_epoch(
                    model,
                    dataloaded_train,
                    optimizer=None,
                    train=True,
                    evaluat=False,
                    metrics=metrics,
                )

                return loss_train

            objective_function = compute_cost

        elif kwargs['cost'] == 'asympmse':

            def compute_asympmse(torch_vars):
                '''This is based on autoptim v.2., based on pytorch's autograd.'''
                _, names = extract_weights(model, requires_grad_only=True)
                delete_weights(model, requires_grad_only=True)
                load_weights(model, names, torch_vars)
                model.zero_grad()

                # forward pass
                start = time.time()
                loss_train = model.predict_mse(x=obs, n_samples=kwargs['n_samples'], verbose=kwargs['verbose'])
                end = time.time()
                duration = np.round(end - start, 2)
                if kwargs['verbose']:
                    print(f"Forward pass takes {duration} seconds")

                return loss_train

            objective_function = compute_asympmse

            callback = None

        # import ipdb; ipdb.set_trace()
        # select only non-frozen parameters
        solution, _ = minimize(
            objective_function=objective_function,
            optim_vars=trainable_params_init_numpy,
            method='CG',
            callback=callback,
            tol=1e-20,
            options={"maxiter": kwargs['n_epochs'], "disp": kwargs['verbose']}
        )
        assert solution  # same as available params?

        # # Visualize estimated sources post training
        # if kwargs['with_plot']:
        #     source_estim = data_marginal.embedder(obs)
        #     visualize_timeseries(source_estim, figpath=kwargs['out'] / "source_final_timeseries.pdf")
        #     visualize_distribution(source_estim, figpath=kwargs['out'] / "source_final_distribution.pdf")
        #     plt.close("all")

        # Save parameters of interest
        metrics.test.data_marginal_params = [param.data for param in data_marginal.parameters()]
        metrics.test.data_marginal_truth_params = [param.data for param in data_marginal_truth.parameters()]

        metrics.test.data_marginal_truth_statedict = data_marginal_truth.state_dict()
        metrics.test.data_marginal_statedict = data_marginal.state_dict()
        metrics.test.noise_marginal_statedict = noise_marginal.state_dict()

    else:
        # Compute predicted mse
        mse = model.predict_mse(
            x=obs, n_samples=kwargs['n_samples'],
            # important : keep this one args.n_samples and not n_samples
            verbose=kwargs['verbose']
        )
        metrics.mse_predicted = mse

    # if kwargs['with_plot']:
    #     plot_kpis(metrics, figpath=kwargs['out'] / "kpis.pdf")

    return metrics


# Wrap your experiment in one function
@mem.cache  # cache to avoid losing computations (checkpointing)
def run_mse_empirical(
        estimation="singlence", generation="gaussian_var",
        n_comp=1, n_samples=1000,
        noise_type="likedata",
        prop_noise=0.5, noise_val=0.5,
        data_val="default",
        seed=0, verbose=False,
        perturb_init=False,
        out=DEFAULTS['out']
):
    t0 = time.time()
    # renaming variables
    noise_to_data = prop_noise / (1 - prop_noise)

    # get default arguments
    kwargs = DEFAULTS.copy()

    # overwrite with specified arguments
    kwargs.update({
        "estimation": estimation,
        "generation": generation,
        "n_comp": n_comp,
        "n_samples": n_samples,
        "prop_noise": prop_noise,
        "noise_type": noise_type,
        "noise_val": noise_val,
        "seed_data_gen": seed,
        "seed_dataset": seed,
        "verbose": verbose,
        "noise_to_data": noise_to_data,
        "out": out,
        "data_val": data_val,
        "perturb_init": perturb_init,
        #
        "train_data": True,
        "experiment": "mse_empirical",
        "hue": str(n_samples),
        "predict_mse": False
    })

    # Rename output directory
    name = get_name_from_dict(kwargs)
    kwargs['out'] = kwargs['out'] / kwargs['experiment'] / kwargs['estimation'] / kwargs['hue'] / name

    # Run estimation
    metrics = run_model(**kwargs)

    # Collect and format results: flatten and concatenate the parameters into one big vector
    params_estim = torch.cat([param.flatten() for param in metrics.test.data_marginal_params])
    params_true = torch.cat([param.flatten() for param in metrics.test.data_marginal_truth_params])

    # Compute squared error
    se = torch.pow(params_estim - params_true, 2).mean().item()

    # return a dict to get a DataFrame down the road
    output = {
        "estimation": estimation,
        "generation": generation,
        "n_samples": n_samples,
        "prop_noise": prop_noise,
        "data_val": data_val,
        "noise_val": noise_val,
        "seed": seed,
        "params_estim": params_estim,
        "params_true": params_true,
        "se": se,
        "noise_type": noise_type,
        "computation_time": time.time() - t0
    }

    return output   # return a dict to get a DataFrame down the road


# Wrap your experiment in one function
@mem.cache  # cache to avoid losing computations (checkpointing)
def run_mse_montecarlo(
        estimation="singlence", generation="gaussian_var",
        n_comp=1, n_samples=1000,
        prop_noise=0.5, noise_val=0.5,
        noise_type="likedata",
        data_val="default",
        seed=0, verbose=False,
        perturb_init=False,
        out=DEFAULTS['out'],
        n_samples_data_montecarlo=10000
):
    t0 = time.time()
    # renaming variables
    noise_to_data = prop_noise / (1 - prop_noise)

    # get default arguments
    kwargs = DEFAULTS.copy()

    # overwrite with specified arguments
    kwargs.update({
        "estimation": estimation,
        "generation": generation,
        "n_comp": n_comp,
        "n_samples": n_samples,
        "prop_noise": prop_noise,
        "data_val": data_val,
        "noise_type": noise_type,
        "noise_val": noise_val,
        "seed_data_gen": seed,
        "seed_dataset": seed,
        "verbose": verbose,
        "noise_to_data": noise_to_data,
        "out": out,
        "perturb_init": perturb_init,
        "n_samples_data_montecarlo": n_samples_data_montecarlo,
        #
        "train_data": True,
        "experiment": "mse_prediction",
        "hue": str(n_samples),
        "predict_mse": True
    })

    # Rename output directory
    name = get_name_from_dict(kwargs)
    kwargs['out'] = kwargs['out'] / kwargs['experiment'] / kwargs['estimation'] / kwargs['hue'] / name

    # Run estimation
    metrics = run_model(**kwargs)

    # Collect results
    mse = metrics.mse_predicted

    # return a dict to get a DataFrame down the road
    output = {
        "estimation": estimation,
        "generation": generation,
        "n_samples": n_samples,
        "n_samples_data_montecarlo": n_samples_data_montecarlo,
        "prop_noise": prop_noise,
        "noise_val": noise_val,
        "data_val": data_val,
        "seed": seed,
        "mse_pred": mse.item(),
        "noise_type": noise_type,
        "computation_time": time.time() - t0
    }

    return output


# Wrap your experiment in one function
@mem.cache  # cache to avoid losing computations (checkpointing)
def run_mse_quadrature(
    generation='gaussian_var', prop_noise=0.5,
    data_val=1., noise_val=0, n_samples=400, estimation='singlence', noise_type='data',
):
    if generation == "gaussian_var":
        from nlica.quadrature.singleparam.gaussian_var import get_mse_quadrature
    elif generation == "gaussian_mean":
        from nlica.quadrature.singleparam.gaussian_mean import get_mse_quadrature
    elif generation == "gaussian_corr":
        from nlica.quadrature.singleparam.gaussian_corr import get_mse_quadrature
    mse = get_mse_quadrature(
        estimation=estimation,
        n_samples=n_samples,
        prop_noise=prop_noise,
        param_data=data_val,
        param_noise=noise_val,
        noise_type=noise_type
    )
    results = {
        "mse": mse,
        "generation": generation,
        "prop_noise": prop_noise,
        "data_val": data_val, "noise_val": noise_val,
        "n_samples": n_samples,
        "estimation": estimation,
        "noise_type": noise_type
    }

    return results


# Wrap your experiment in one function
@mem.cache  # cache to avoid losing computations (checkpointing)
def run_optimal_noise(
        estimation="singlence", generation="gaussian_var",
        n_comp=1, n_samples=10000,
        n_samples_data_montecarlo=10000,
        prop_noise=0.5,
        data_val="default",
        seed=0, verbose=True, noise_bins=20,
        out=DEFAULTS['out'], n_epochs=100,
):
    t0 = time.time()
    # renaming variables
    noise_to_data = prop_noise / (1 - prop_noise)

    # get default arguments
    kwargs = DEFAULTS.copy()

    # overwrite with specified arguments
    kwargs.update({
        "estimation": estimation,
        "generation": generation,
        "n_comp": n_comp,
        "n_samples": n_samples,
        "n_samples_data_montecarlo": n_samples_data_montecarlo,
        "prop_noise": prop_noise,
        "seed_data_gen": seed,
        "seed_dataset": seed,
        "verbose": verbose,
        "noise_to_data": noise_to_data,
        "noise_bins": noise_bins,
        "out": out,
        "data_val": data_val,
        "n_epochs": n_epochs,
        #
        "train_noise": True,
        "perturb_init": False,
        "noise_type": "flexible",
        "experiment": "optimalnoise",
        "hue": generation,
        "cost": "asympmse"
    })

    # Rename output directory
    name = get_name_from_dict(kwargs)
    kwargs['out'] = kwargs['out'] / kwargs['experiment'] / kwargs['estimation'] / kwargs['hue'] / name

    # Run estimation
    metrics = run_model(**kwargs)

    # return a dict to get a DataFrame down the road
    output = {
        "estimation": estimation,
        "generation": generation,
        "n_samples": n_samples,
        "n_samples_data_montecarlo": n_samples_data_montecarlo,
        "prop_noise": prop_noise,
        "data_val": data_val,
        "seed": seed,
        "n_comp": n_comp,
        "n_bins": noise_bins,
        "data_marginal_statedict": metrics.test.data_marginal_statedict,
        "data_marginal_truth_statedict": metrics.test.data_marginal_truth_statedict,
        "noise_marginal_statedict": metrics.test.noise_marginal_statedict,
        "computation_time": time.time() - t0
    }

    return output

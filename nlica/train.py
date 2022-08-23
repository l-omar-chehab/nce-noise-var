"""Functions that train a PyTorch neural network."""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd.functional import jacobian

from treetable import treetable, table, group, leaf
from pathlib import Path
import time

import numpy as np

from nlica.utils import amari_distance, compute_correl_score, cross_correl

# Classes to store KPIs


class Pretext:
    def __init__(self):
        self.epoch = list()
        self.loss = list()
        self.acc = list()
        self.acc_lb = None  # lower bound: chance
        self.acc_ub = None  # upper bound: random forest
        self.time = list()

        # specific to mle
        self.loss_prior = list()
        self.loss_activation = list()
        self.loss_invertibility = list()

        # parameters of the experiment
        self.experiment = dict()


class Downstream:
    def __init__(self):
        self.epoch = list()
        self.pearson_r = list()  # linear dep score
        self.pearson_r_dep = list()
        self.hsic = list()  # stat dep score
        self.amari = list()
        self.invertibility_mean = list()
        self.invertibility_std = list()


class Metrics:
    def __init__(self, path=Path("./dump"), verbose=False):
        # where to store
        self.logfile_path = path / "log.txt"

        # what to store: KPIs
        self.train = Pretext()
        self.test = Pretext()
        self.downstream = Downstream()
        self.verbose = verbose

        # put here until better place
        self.asymp_var_theory = None

        # what to store: model weights
        self.best_model_path = path / "model_best.th"

        # how to display: treetable format
        self.mytable = table(
            [
                group("train", groups=[leaf("loss", ".2f"), leaf("acc", ".2%")]),
                group("test", groups=[leaf("loss", ".2f"), leaf("acc", ".2%")]),
            ]
        )

    def record(self):
        log = open(self.logfile_path, "a")  # TODO: optimize this

        epoch = self.train.epoch[-1]
        duration_train = self.train.time[-1]
        duration_test = self.test.time[-1]

        log.write(
            f"Epoch {epoch}: "
            f"train =  {duration_train} seconds | "
            f"test =  {duration_test} seconds \n"
        )
        log.flush()
        log.close()

    def display(self):

        if len(self.train.acc) == 0:
            lines = [
                {
                    "train": {"loss": self.train.loss[-1]},
                    "test": {"loss": self.test.loss[-1]},
                },
            ]
        else:
            lines = [
                {
                    "train": {"loss": self.train.loss[-1], "acc": self.train.acc[-1]},
                    "test": {"loss": self.test.loss[-1], "acc": self.test.acc[-1]},
                },
            ]

        if self.verbose:
            print(treetable(lines, self.mytable))

    def save(self, path):

        # Log initial and final source recovery scores in text file
        # TODO: use best not final model score
        log = open(self.logfile_path, "a")

        try:
            log.write("\n \n Pearson R (init): ")
            for r in self.downstream.pearson_r[0]:
                log.write(f"{r:.3f} ")
            log.write("\n Pearson R (final): ")
            for r in self.downstream.pearson_r[-1]:
                log.write(f"{r:.3f} ")

            log.flush()
            log.close()

            # Print initial and final source recovery scores
            if self.verbose:
                print("Pearson R (init): ", self.downstream.pearson_r[0])
                print("Pearson R (final): ", self.downstream.pearson_r[-1])
        except Exception:
            pass

        torch.save(self, path)


# Training over an epoch (different methods)
# supervised = True : should work for SingleNCE
# supervised = Flase : should work for SingleMLE

def run_epoch(
    model,
    dataloaded,
    optimizer=None,
    train=True,
    evaluat=False,
    progress=False,
    metrics=None,
    supervised=True
):
    if metrics is not None:
        if train:
            metrics_actual = metrics.train
        else:
            metrics_actual = metrics.test

    if evaluat:
        model.eval()
        desc = "Evaluate (train or test set)"
    else:
        model.train()
        desc = "Train (train set)"

    # outside the loop
    criterion = nn.BCELoss()

    # with is_autograd:
    with torch.enable_grad():
        running_loss, running_acc, n_samples = 0, 0, 0

        dataloaded_iter = iter(dataloaded)  # make iterable
        if progress:
            dataloaded_iter = tqdm(
                dataloaded_iter,
                leave=False,
                ncols=120,
                total=len(dataloaded),
                desc=desc,
            )

        start = time.time()
        for batch in dataloaded_iter:
            x, y = batch  # unpack, format (B, C)
            batch_size = len(x)
            n_samples += batch_size

            if supervised:
                y_pred = model(x)
                loss = criterion(y_pred, y)
            else:  # foward model is loss
                neglik = model(x)
                loss = neglik.mean()
            running_loss += loss * batch_size

            # Compute accuracy for which the loss is a proxy
            if supervised:
                cond = (y_pred.detach() > 0.5) == y
                running_acc += cond.sum().item()

            # Backward then optimize
            if (train) and (not evaluat) and (optimizer is not None):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        end = time.time()
        duration = np.round(end - start, 2)
        running_loss /= n_samples
        if supervised:
            running_acc = running_acc / n_samples

        # record
        if metrics is not None:
            metrics_actual.loss.append(running_loss.item())
            metrics_actual.time.append(duration)
            if supervised:
                metrics_actual.acc.append(running_acc)

    return running_loss

# def run_epoch_mle(model,
#                   dataloaded,
#                   optimizer=None,
#                   train=True,
#                   evaluat=False,
#                   progress=False,
#                   metrics=None):
#     if train:
#         metrics_actual = metrics.train
#     else:
#         metrics_actual = metrics.test

#     if evaluat:
#         model.eval()
#         is_autograd = torch.no_grad()
#         desc = "Evaluate (train or test set)"
#     else:
#         model.train()
#         is_autograd = torch.enable_grad()
#         desc = "Train (train set)"

#     with is_autograd:
#         running_loss, n_samples = 0, 0
#         running_loss_1, running_loss_2, running_loss_3 = 0, 0, 0

#         dataloaded_iter = iter(dataloaded)  # make iterable
#         if progress:
#             dataloaded_iter = tqdm(dataloaded_iter, leave=False, ncols=120,
#                                 total=len(dataloaded), desc=desc)

#         start = time.time()
#         for batch_idx, batch in enumerate(dataloaded_iter):
#             x, y = batch  # unpack, format (B, C)
#             batch_size = len(x)
#             n_samples += batch_size

#             loss, loss_1, loss_2, loss_3, s = loss_mle(model, x)
#             print("s.grad: ", s.grad)

#             running_loss_1 += loss_1.item() * batch_size
#             running_loss_2 += loss_2.item() * batch_size
#             running_loss_3 += loss_3.item() * batch_size
#             running_loss += loss.item() * batch_size

#             # Backward then optimize
#             if train and not evaluat:
#                 loss.backward()
#                 print("s.grad: ", s.grad)
#                 use_relative_grad(model, x=x, z=s)  # the trick is here
#                 optimizer.step()
#                 optimizer.zero_grad()

#         end = time.time()
#         duration = np.round(end - start, 2)
#         running_loss /= n_samples
#         running_loss_1 /= n_samples
#         running_loss_2 /= n_samples
#         running_loss_3 /= n_samples

#         # record
#         if evaluat:
#             metrics_actual.loss.append(running_loss)
#             metrics_actual.loss_prior.append(running_loss_1)
#             metrics_actual.loss_activation.append(running_loss_2)
#             metrics_actual.loss_invertibility.append(running_loss_3)
#             metrics_actual.time.append(duration)

#     return running_loss


# Evaluating downstream task


def get_lower_triang_values(A):
    if A.shape != (1, 1):
        vals = torch.cat([torch.diag(A, idx) for idx in range(1, A.shape[0])])
        vals = vals.flatten().tolist()
    else:
        vals = [0]  # default, lower triangular does not exist for (1, 1) tensor
    return vals


def eval_downstream(embedder, obs, source, metrics, mixing_net):
    """Encoder is a torch net. obs and source are torch tensors."""

    # Source recovery (via Pearson R)
    source_estim = embedder(obs)
    r_score, _ = compute_correl_score(source, source_estim)
    corr_mat = cross_correl(source_estim, source_estim)
    r_dep = get_lower_triang_values(corr_mat.abs())

    # Independence of sources
    # hsic_score = hsic(source_estim)
    hsic_score = 0  # placeholder, as computation explodes with n_data

    # Invertibility of embedder (via logdetjacobian)
    # this requires autograd : be careful not to use 'with torch.no_grad()'
    if hasattr(embedder, "weight"):  # linear encoder, jacob indep from data
        jacobs = [jacobian(func=embedder, inputs=obs[0])]
    else:  # nonlinear encoder
        jacobs = [jacobian(func=embedder, inputs=datapoint) for datapoint in obs]
    logdets = [jacob.det().abs().log() for jacob in jacobs]
    logdets_mean = torch.Tensor(logdets).mean().item()
    logdets_std = torch.Tensor(logdets).std().item()

    # Forward operator recovery
    if hasattr(embedder, "weight"):  # linear encoder
        amari_score = amari_distance(embedder.weight.data, mixing_net.weight.data)
        if metrics is not None:
            metrics.downstream.amari.append(amari_score)

    # Update Metrics
    if metrics is not None:
        metrics.downstream.pearson_r.append(r_score)
        metrics.downstream.pearson_r_dep.append(r_dep)
        metrics.downstream.hsic.append(hsic_score)
        metrics.downstream.invertibility_mean.append(logdets_mean)
        metrics.downstream.invertibility_std.append(logdets_std)

    return r_score


# Training over all epochs


def train_eval_model(
    model,
    dataloaded_train,
    dataloaded_test,
    optimizer=None,
    scheduler=None,
    n_epochs=100,
    eval_downstream_every=5,
    run_epoch=None,
    eval_downstream=eval_downstream,
    metrics=Metrics(),
    downstream_obs=None,
    downstream_source=None,
    mixing_net=None,
):

    model_best = model.state_dict()
    loss_train_best = np.inf

    # Loop over epochs
    # epochs = trange(n_epochs + 1, desc='Training Model', leave=True)
    epochs = list(range(n_epochs + 1))
    for epoch in epochs:
        # epochs.set_description(f'epoch {epoch}')
        # epochs.refresh()

        if epoch == 0:
            pass
        elif epoch > 0:
            # Train set
            _ = run_epoch(
                model,
                dataloaded_train,
                optimizer=optimizer,
                train=True,
                evaluat=False,
                metrics=None,
            )

        # Train set (model frozen)
        loss_train = run_epoch(
            model,
            dataloaded_train,
            optimizer=optimizer,
            train=True,
            evaluat=True,
            metrics=metrics,
        )

        # Test set
        loss_test = run_epoch(
            model,
            dataloaded_test,
            optimizer=optimizer,
            train=False,
            evaluat=True,
            metrics=metrics,
        )

        # Optimization trick
        if scheduler:
            scheduler.step(loss_test)

        # Update best model
        if loss_train < loss_train_best:
            model_best = model.state_dict()
            loss_train_best = loss_train

        # Update metrics, log and display
        metrics.train.epoch.append(epoch)
        metrics.test.epoch.append(epoch)
        metrics.record()
        metrics.display()

        # Evaluate Downstream Task (*during* Pretext Task)
        if eval_downstream is not None:
            if epoch % eval_downstream_every == 0:
                metrics.downstream.epoch.append(epoch)
                _ = eval_downstream(
                    encoder=model.encoder,
                    obs=downstream_obs,
                    source=downstream_source,
                    metrics=metrics,
                    mixing_net=mixing_net,
                )

    # After training
    torch.save(model_best, metrics.best_model_path)


# with scipy
# global iteration
iteration = 0


def callback_extended(xk,
                      model,
                      dataloaded_train,
                      dataloaded_test,
                      eval_downstream_every=1,
                      run_epoch=None,
                      eval_downstream=eval_downstream,
                      metrics=Metrics(),
                      downstream_obs=None,
                      downstream_source=None,
                      mixing_net=None,
                      verbose=False):
    global iteration  # looks for a global variable (outside the function)
    if verbose:
        print("ITERATION: ", iteration)

    # if encoder is trainable, update the model with its value
    # isn't this already done in objective func?
    # param_names = [param_name for (param_name, param) in model.named_parameters()
    #                if param.requires_grad == True]

    # shapes = [param.shape for (_, param) in model.named_parameters()
    #           if param.requires_grad == True]
    # print("SHAPES: ", shapes)
    # sel_encoder = [idx for idx in range(len(param_names)) if "encoder" in param_names[idx]]
    # print("SEL ENCODER: ", sel_encoder)
    # idx_encoder = sel_encoder[0]
    # xk_split = np.split(xk, np.cumsum([np.prod(shape) for shape in shapes[:-1]]).astype(int))
    # xk_encoder = xk_split[idx_encoder]

    # # set model to current weight
    # encoder_mat = xk_encoder.reshape(model.n_comp, model.n_comp)  # harcoded
    # model.encoder.weight.data = torch.from_numpy(encoder_mat).float()

    run_epoch(model,
              dataloaded_train,
              optimizer=None,
              train=True,
              evaluat=True,
              metrics=metrics)
    run_epoch(model,
              dataloaded_test,
              optimizer=None,
              train=False,
              evaluat=True,
              metrics=metrics)

    # display current iterates
    if verbose:
        print([(name, param) for (name, param) in model.named_parameters() if param.requires_grad])

    # record current iteration
    metrics.train.epoch.append(iteration)
    metrics.test.epoch.append(iteration)
    metrics.record()
    metrics.display()

    # evaluate and record downstream metrics
    if eval_downstream is not None:
        if iteration % eval_downstream_every == 0:
            metrics.downstream.epoch.append(iteration)
            eval_downstream(model.data_marginal.embedder, downstream_obs, downstream_source,
                            metrics, mixing_net)

    iteration += 1

    return False

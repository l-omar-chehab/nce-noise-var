"""Functions for visualization"""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa import stattools
import seaborn as sns
import pandas as pd

from nlica.utils import is_numpy


def visualize_timeseries(x, fig=None, ax=None, figpath=None, noise=None):
    """Plot exploratory view of data:
    the sampling structure per component.

    Args:
        x (np.array): of shape (n_data, n_comp)
    """
    n_data, n_comp = x.shape

    if not (fig and ax):
        plt.close("all")
        fig, axes = plt.subplots(nrows=n_comp + 1, ncols=1, figsize=(10, 10))

    # Convert to numpy
    if not is_numpy(x):
        # assume in torch, otherwise crudely broadcast with np.array()
        x = x.data.numpy()
    if (noise is not None) and (not is_numpy(noise)):
        noise = noise.data.numpy()

    # fig, axes = plt.subplots(nrows=n_comp + 1, ncols=1, figsize=(10, 10))

    # Timeseries per component
    for comp, ax in enumerate(axes.flatten()[:-1]):
        ax.plot(x[:, comp])
        ax.set(ylabel=f"component {comp}", xlim=(0, 50))

    # Autocorrelation per component
    ax = axes.flatten()[-1]
    for comp in range(n_comp):
        acf = stattools.acf(x[:, comp], fft=True, nlags=120)
        ax.plot(acf)
    ax.grid("on")
    ax.set(ylim=(-0.5, 1), xlim=(0, 50), ylabel="Autocorrelation")

    fig.suptitle("Sampling structure per component")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    if figpath is not None:
        plt.savefig(figpath)

    return fig, ax


def visualize_distribution(x, fig=None, ax=None, figpath=None, noise=None):
    """Plot exploratory view of data;
    the sampling structure between components.

    To include within a larger figure, check this ref:
    https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot

    Args:
        x (np.array): of shape (n_data, n_comp)
    """
    if not (fig and ax):
        plt.close("all")

    # Convert to numpy
    if not is_numpy(x):
        # assume in torch, otherwise crudely broadcast with np.array()
        x = x.data.numpy()
    if (noise is not None) and (not is_numpy(noise)):
        noise = noise.data.numpy()

    n_data, n_comp = x.shape

    fig, axes = plt.subplots(nrows=n_comp + 1, ncols=1, figsize=(10, 10))

    x_dataframe = pd.DataFrame(x)
    g = sns.PairGrid(x_dataframe)

    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)

    g.fig.set_size_inches(10, 10)
    g.fig.suptitle("Sampling structure between components")
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    if figpath is not None:
        plt.savefig(figpath)

    return g.fig, g.axes


def plot_kpis(metrics, figpath=None):
    """Show KPIs tracked during training of Pretext Task."""
    plt.close("all")
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 8), sharex=True)

    color_1 = "blue"
    color_2 = "gray"
    xmax = max(metrics.train.epoch) if len(metrics.train.epoch) != 0 else 10

    # Pretext
    ax = axes[0]
    if len(metrics.test.acc) == 0:
        # unsupervised task (e.g. MLE)
        ax.plot(metrics.test.epoch, metrics.test.loss, color=color_1)
        ylim = None
        ylabel = "-Loglik"
    else:
        # self-supervised task (e.g. PCL)
        # plot acc for which crossent-loss is a proxy
        ax.plot(metrics.test.epoch, metrics.test.acc, color=color_1)
        ylim = (0.5, 1.0)
        ylabel = "Accuracy"

    if metrics.test.acc_ub is not None:
        # self-supervised task (e.g. PCL)
        ax.axhline(
            metrics.test.acc_ub,
            lw=0.5,
            ls="--",
            color=color_1,
            label="Upper-Bound (Rand. Forest)",
        )

    ax.set(
        ylim=ylim, xlim=(0, xmax), xlabel="Epochs", ylabel=ylabel, title="Pretext Task"
    )
    ax.legend()

    # Downstream (source recovery)
    ax = axes[1]
    if len(metrics.downstream.pearson_r) != 0:
        ax.plot(
            metrics.downstream.epoch,
            np.array(metrics.downstream.pearson_r),
            ls="--",
            alpha=0.5,
            color=color_2,
        )
    ax.set(
        ylim=(0.5, 1.0),
        xlim=(0, xmax),
        xlabel="Epochs",
        ylabel="Pearson R",
        title="Source Recovery",
    )

    # Downstream (source independence)
    ax = axes[2]
    if len(metrics.downstream.hsic) != 0:
        ax.plot(metrics.downstream.epoch, np.array(metrics.downstream.hsic), ls="--")
    ax.set(xlim=(0, xmax), xlabel="Epochs", ylabel="HSIC", title="Source Dependence")

    ax = axes[3]
    if len(metrics.downstream.pearson_r_dep) != 0:
        n_comp = len(metrics.downstream.pearson_r_dep[0])
        for comp in range(n_comp):
            ax.plot(
                metrics.downstream.epoch,
                np.array(metrics.downstream.pearson_r_dep)[:, comp],
                ls="--",
                color=color_2,
            )
    ax.set(xlim=(0, xmax), ylim=(0.0, 1.01), xlabel="Epochs", ylabel="Pearson R")

    # Downstream (operator recovery)
    ax = axes[4]
    if len(metrics.downstream.amari) != 0:
        ax.plot(metrics.downstream.epoch, np.array(metrics.downstream.amari), ls="--")
    ax.set(xlim=(0.0, xmax), xlabel="Epochs", ylabel="Amari", title="Operator Recovery")

    # Tracking invertibility of the encoder
    ax = axes[5]
    if len(metrics.downstream.invertibility_mean) != 0:
        mean = np.array(metrics.downstream.invertibility_mean)
        error = np.array(metrics.downstream.invertibility_std)
        ax.plot(metrics.downstream.epoch, mean)
        ax.fill_between(metrics.downstream.epoch, mean + error, mean - error)
    ax.set(
        title="Invertibility \n" + r"$-\infty$ = non-invertible, $0$ = orthogonal",
        ylabel="LogdetJacob",
        xlim=(0, xmax),
    )

    fig.tight_layout()

    # Save figure
    if figpath is not None:
        plt.savefig(figpath)

    return fig, axes

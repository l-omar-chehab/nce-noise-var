"""Plots for the paper.
We plot the MSE vs. hyperparameters {data parameter, nb samples, noise parameter, noise proportion}."""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

from nlica.utils import trunc
from nlica.defaults import RESULTS_FOLDER, IMAGE_FOLDER

# %%
# import data
results_quadrature = torch.load(RESULTS_FOLDER / "expe_quadrature_checkpoint.th")
results_quadrature = pd.DataFrame(results_quadrature)

# %%
# reformat data
results_quadrature['mse'] = results_quadrature['mse'].apply(lambda x: np.log10(x))
results_quadrature.rename(columns={'mse': 'logmse'}, inplace=True)

# take log of variance
sel = results_quadrature['generation'] == 'gaussian_var'
results_quadrature.loc[sel, 'data_val'] = results_quadrature.loc[sel, 'data_val'].apply(lambda x: np.log10(x))
results_quadrature.loc[sel, 'noise_val'] = results_quadrature.loc[sel, 'noise_val'].apply(lambda x: np.log10(x))

results_quadrature.drop(['n_samples'], axis=1, inplace=True)


# %%
# Figure: MSE vs. noise parameter

generations = ["gaussian_mean", "gaussian_var", "gaussian_corr"]
data_vals_desired = {
    "gaussian_mean": [0.],
    "gaussian_var": [np.log10(1.)],
    "gaussian_corr": [0., 0.9],
}
param_names = ["Mean", "Variance (log10)", "Correlation"]

# figure
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14.4, 5))
fontsize = 16
params = {'font.size': fontsize}
plt.rcParams.update(params)

for ax, generation, param_name in zip(axes.flatten(), generations, param_names):

    ymin = np.inf
    ymax = -np.inf

    for data_val_desired in data_vals_desired[generation]:

        # data
        estimation = "singlence"
        prop_noise_desired = 0.5
        color = 'blue'

        sel = results_quadrature.generation == generation
        all_data_vals = results_quadrature.loc[sel, "data_val"].unique()
        idx_data_val = np.abs(all_data_vals - data_val_desired).argmin()
        data_val = all_data_vals[idx_data_val]

        all_prop_noise = results_quadrature.loc[sel, "prop_noise"].unique()
        idx_prop_noise = np.abs(all_prop_noise - prop_noise_desired).argmin()
        prop_noise = all_prop_noise[idx_prop_noise]

        # noise : gaussian family
        sel = (results_quadrature['estimation'] == estimation) & \
            (results_quadrature['generation'] == generation) & \
            (results_quadrature['data_val'] == data_val) & \
            (results_quadrature['prop_noise'] == prop_noise) & \
            (results_quadrature['noise_type'] == 'data')
        xs = results_quadrature[sel].noise_val.to_numpy()
        ys = results_quadrature[sel].logmse.to_numpy()

        y_bestnoise, idx_bestnoise = ys.min(), ys.argmin()
        x_bestnoise = xs[idx_bestnoise]

        print("Best noise param: ", x_bestnoise)

        idx_data = np.argmin(np.abs(xs - data_val))
        x_data, y_data = xs[idx_data], ys[idx_data]

        ax.plot(xs, ys, color=color, label='Parametric Noise')
        ax.scatter(
            x_bestnoise, y_bestnoise, marker='x', s=150, lw=2, color=color,
            label='Parametric Noise (optimal)'
        )
        ax.scatter(
            x_data, y_data, s=150, lw=2, facecolors='none', edgecolors=color,
            label='Parametric Noise (noise matches data)'
        )

        ymin, ymax = min([ymin, ys.min()]), max([ymax, ys.max()])

        # noise : pihlaja
        sel = (results_quadrature['estimation'] == estimation) & \
            (results_quadrature['generation'] == generation) & \
            (results_quadrature['data_val'] == data_val) & \
            (results_quadrature['prop_noise'] == prop_noise) & \
            (results_quadrature['noise_type'] == 'optimal')
        y = results_quadrature.loc[sel, "logmse"].item()

        ymin, ymax = min([ymin, y]), max([ymax, y])

        ax.axhline(y, color='red', label='Optimal Noise (all-noise limit)')

        # cramer-rao lower bound
        sel = (results_quadrature['estimation'] == 'singlemle') & \
            (results_quadrature['generation'] == generation) & \
            (results_quadrature['data_val'] == data_val) & \
            (results_quadrature['prop_noise'] == prop_noise)
        xs = results_quadrature[sel].noise_val.to_numpy()
        y = results_quadrature[sel].logmse.item()

        ymin, ymax = min([ymin, y]), max([ymax, y])

        ax.axhline(y, color='black', ls='dotted', label='Lower Bound (Cramer-Rao)')

        # esthetics
        ax.set(
            yticks=trunc(np.array([ymin, ymax]), decs=1),
            ylabel="MSE (log10)",
            xlabel=param_name
        )


handles, labels = axes.flatten()[0].get_legend_handles_labels()
# fig.legend(
#     handles, labels, loc='upper center',
#     bbox_to_anchor=(0.48, 1.16),
#     ncol=1)

axes.flatten()[-1].axis('off')
axes.flatten()[-1].legend(
    handles, labels, loc='upper center',
    bbox_to_anchor=(0.48, 1.16),
    ncol=1
)

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / "mse_vs_noiseval.pdf")


# %%
# Figure: MSE vs. noise proportion

generations = ["gaussian_mean", "gaussian_var", "gaussian_corr"]
data_vals_desired = {
    "gaussian_mean": 0.,
    "gaussian_var": np.log10(1.),
    "gaussian_corr": 0.
}
# hardcoded
noise_vals_desired = {
    "gaussian_mean": -1.5,
    "gaussian_var": 0.64,
    "gaussian_corr": 0.
}
param_names = ["Mean", "Variance", "Correlation"]

# figure
fig, axes = plt.subplots(nrows=len(generations), figsize=(7.2, 7), sharex=True)
fontsize = 16
params = {'font.size': fontsize}
plt.rcParams.update(params)

for ax, generation, param_name in zip(axes, generations, param_names):

    ymin = np.inf
    ymax = -np.inf

    estimation = "singlence"
    generation = generation
    data_val_desired = data_vals_desired[generation]
    color = 'blue'
    noise_val_desired = noise_vals_desired[generation]

    sel = (results_quadrature['generation'] == generation) & \
        (results_quadrature['estimation'] == estimation) & \
        (results_quadrature['noise_type'] == 'data')
    all_data_vals = results_quadrature.loc[sel, "data_val"].unique()
    idx_data_val = np.abs(all_data_vals - data_val_desired).argmin()
    data_val = all_data_vals[idx_data_val]

    all_noise_vals = results_quadrature.loc[sel, "noise_val"].unique()
    idx_noise_val = np.abs(all_noise_vals - noise_val_desired).argmin()
    noise_val = all_noise_vals[idx_noise_val]
    idx_noise_val = np.abs(all_noise_vals - data_val_desired).argmin()
    noise_val_data = all_noise_vals[idx_noise_val]

    # noise : data
    sel = (results_quadrature['estimation'] == estimation) & \
        (results_quadrature['generation'] == generation) & \
        (results_quadrature['data_val'] == data_val) & \
        (results_quadrature['noise_type'] == 'data') & \
        (results_quadrature['noise_val'] == noise_val_data)
    xs = results_quadrature[sel].prop_noise.to_numpy()
    ys = results_quadrature[sel].logmse.to_numpy()

    y_bestnoise, idx_bestnoise = ys.min(), ys.argmin()
    x_bestnoise = xs[idx_bestnoise]

    ax.plot(xs, ys, color='grey', label='Parametric Noise (data)')
    ax.scatter(x_bestnoise, y_bestnoise, marker='x', s=150, lw=2, color='grey')

    ymin, ymax = min([ymin, ys.min()]), max([ymax, ys.max()])

    # noise : gaussian family
    sel = (results_quadrature['estimation'] == estimation) & \
        (results_quadrature['generation'] == generation) & \
        (results_quadrature['data_val'] == data_val) & \
        (results_quadrature['noise_type'] == 'data') & \
        (results_quadrature['noise_val'] == noise_val)
    xs = results_quadrature[sel].prop_noise.to_numpy()
    ys = results_quadrature[sel].logmse.to_numpy()

    y_bestnoise, idx_bestnoise = ys.min(), ys.argmin()
    x_bestnoise = xs[idx_bestnoise]

    ax.plot(xs, ys, color=color, label=r'Parametric Noise (optimal for $\nu=1$)')
    ax.scatter(x_bestnoise, y_bestnoise, marker='x', s=150, lw=2, color=color)

    ymin, ymax = min([ymin, ys.min()]), max([ymax, ys.max()])

    # noise : pihlaja
    sel = (results_quadrature['estimation'] == estimation) & \
        (results_quadrature['generation'] == generation) & \
        (results_quadrature['data_val'] == data_val) & \
        (results_quadrature['noise_type'] == 'optimal')
    xs = results_quadrature[sel].prop_noise.to_numpy()
    ys = results_quadrature[sel].logmse.to_numpy()

    y_bestnoise, idx_bestnoise = ys.min(), ys.argmin()
    x_bestnoise = xs[idx_bestnoise]

    ax.plot(xs, ys, color='red', label='Optimal Noise (all-noise limit)')
    ax.scatter(x_bestnoise, y_bestnoise, marker='x', s=150, lw=2, color='red')

    ymin, ymax = min([ymin, ys.min()]), max([ymax, ys.max()])

    # cramer-rao lower bound
    sel = (results_quadrature['estimation'] == 'singlemle') & \
        (results_quadrature['generation'] == generation) & \
        (results_quadrature['data_val'] == data_val) & \
        (results_quadrature['noise_type'] == 'data')
    xs = results_quadrature[sel].prop_noise.to_numpy()
    ys = results_quadrature[sel].logmse.to_numpy()

    ax.plot(xs, ys, color='black', ls='--', label='Lower Bound (Cramer-Rao)')

    ymin, ymax = min([ymin, ys.min()]), max([ymax, ys.max()])

    ax.set(
        yticks=trunc(np.array([ymin, ymax]), decs=1),
        ylabel="MSE (log10)",
    )
    ax.set_title(f"Gaussian {param_name} Model", fontsize=fontsize)


axes[-1].set(
    xlabel="Noise Proportion",
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc='upper center',
    bbox_to_anchor=(0.48, 1.21),
    ncol=1)

fig.tight_layout()
plt.savefig(
    IMAGE_FOLDER / "mse_vs_noiseprop.pdf",
    bbox_inches='tight'
)


# %%
# Figure: noise parameter (best) vs. data parameter

generations = ["gaussian_mean", "gaussian_var", "gaussian_corr"]
param_names = {
    "gaussian_mean": "Mean",
    "gaussian_var": "Var (log10)",
    "gaussian_corr": "Correlation"
}
estimation = "singlence"
prop_noise_desired = 0.5
color = 'blue'
globalopt = True

# figure
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.2, 6))
fontsize = 16
params = {'font.size': fontsize}
plt.rcParams.update(params)

for ax, generation in zip(axes.flatten(), generations):

    param_name = param_names[generation]

    sel = results_quadrature.generation == generation
    all_prop_noise = results_quadrature.loc[sel, "prop_noise"].unique()
    idx_prop_noise = np.abs(all_prop_noise - prop_noise_desired).argmin()
    prop_noise = all_prop_noise[idx_prop_noise]

    xs = results_quadrature.loc[sel, "data_val"].unique()

    # for fixed noise prop
    # do this better using groupby
    ys = []
    for x in xs:
        sel = (results_quadrature['estimation'] == estimation) & \
            (results_quadrature['generation'] == generation) & \
            (results_quadrature['data_val'] == x) & \
            (results_quadrature['noise_type'] == 'data') & \
            (results_quadrature['prop_noise'] == prop_noise)
        results_quadrature_sel = results_quadrature[sel]
        y = results_quadrature_sel.loc[results_quadrature_sel['logmse'].idxmin()]['noise_val']
        ys.append(y)

    lim_bottom = min([min(xs), min(ys)])
    lim_top = max([max(xs), max(ys)])

    xs = np.array(xs)
    ys = np.array(ys)

    if generation == "gaussian_mean":
        ax.scatter(xs, ys, color=color)
        sel_up = ys > xs
        sel_bottom = ys < xs
        ax.plot(xs[sel_up], ys[sel_up], color=color, label=r'Optimal Noise ($\nu=1$)')
        ax.plot(xs[sel_bottom], ys[sel_bottom], color=color)
    else:
        ax.plot(xs, ys, color=color, label=r'Optimal Noise ($\nu=1$)')

    # global optimization over noise prop
    # do this better using groupby
    if globalopt:
        ys = []
        for x in xs:
            sel = (results_quadrature['estimation'] == estimation) & \
                (results_quadrature['generation'] == generation) & \
                (results_quadrature['data_val'] == x) & \
                (results_quadrature['noise_type'] == 'data')
            results_quadrature_sel = results_quadrature[sel]
            y = results_quadrature_sel.loc[results_quadrature_sel['logmse'].idxmin()]['noise_val']
            ys.append(y)

        lim_bottom = min([min(xs), min(ys)])
        lim_top = max([max(xs), max(ys)])

        xs = np.array(xs)
        ys = np.array(ys)

        if generation == "gaussian_mean":
            # ax.scatter(xs, ys, color='purple')
            sel_up = ys > xs
            sel_bottom = ys < xs
            ax.plot(xs[sel_up], ys[sel_up], color='purple', label=r'Optimal Noise ($\nu^{\mathrm{opt}}$)')
            ax.plot(xs[sel_bottom], ys[sel_bottom], color='purple')
        else:
            ax.plot(xs, ys, color='purple', label=r'Optimal Noise ($\nu^{\mathrm{opt}}$)')

    # compare with identity function
    ax.plot(np.linspace(lim_bottom, lim_top, 100),
            np.linspace(lim_bottom, lim_top, 100),
            color='grey', ls='--', alpha=0.5,
            label='Noise matches Data')

    ax.set(
        xlabel=f"Data {param_name}",
        ylabel=f"Optimal \n Noise {param_name}",
        xlim=(lim_bottom, lim_top),
        ylim=(lim_bottom, lim_top),
        xticks=np.round(np.linspace(lim_bottom, lim_top, 3), 1),
        yticks=np.round(np.linspace(lim_bottom, lim_top, 3), 1)
        # yticks=[np.round(ys.min(), 1), np.round(ys.max(), 1)]
    )
    ax.set_aspect('equal', 'box')
    # ax.grid("on")

handles, labels = axes.flatten()[0].get_legend_handles_labels()
# fig.legend(
#     handles, labels, loc='upper center',
#     bbox_to_anchor=(0.48, 1.2),
#     ncol=2)
axes.flatten()[-1].axis('off')
axes.flatten()[-1].legend(
    handles, labels, loc='center',
    # bbox_to_anchor=(0.48, 1.2),
    ncol=1)

fig.tight_layout(pad=1.)

plt.savefig(
    IMAGE_FOLDER / "noiseval_vs_dataval.pdf",
    # bbox_inches='tight'
)


# %%
# Image

estimation = "singlence"
# generation = "gaussian_var"
generation = "gaussian_corr"
# data_val_desired = np.log10(1.0)
data_val_desired = 0.7
param_name = "Variance (log10)" if generation == "gaussian_var" else "Correlation"

sel = results_quadrature.generation == generation
all_data_vals = results_quadrature.loc[sel, "data_val"].unique()
idx_data_val = np.abs(all_data_vals - data_val_desired).argmin()
data_val = all_data_vals[idx_data_val]

sel = (results_quadrature['estimation'] == estimation) & \
    (results_quadrature['generation'] == generation) & \
    (results_quadrature['data_val'] == data_val) & \
    (results_quadrature['noise_type'] == 'data')
prop_noises = results_quadrature.loc[sel, "prop_noise"].unique()
noise_vals = results_quadrature.loc[sel, "noise_val"].unique()

data = results_quadrature[sel][['noise_val', 'prop_noise', 'logmse']].set_index(['noise_val', 'prop_noise'])
logmse = data.unstack().values[:, :]  # shape (noise_vals, prop_noises)

noise_vals_best_idx = np.argmin(logmse, axis=0)   # shape (prop_noises)
prop_noises_best_idx = np.argmin(logmse, axis=1)  # shape (noise_vals)

noise_vals_best = noise_vals[noise_vals_best_idx]   # shape (prop_noises)
prop_noises_best = prop_noises[prop_noises_best_idx]   # shape (prop_noises)

# Figure
fig, ax = plt.subplots()
img = ax.imshow(logmse)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.15)
cbar = fig.colorbar(img, cax=cax, orientation='vertical', ticks=[-5., -1.])
cbar.ax.set_ylabel('MSE (log10)', rotation=270, labelpad=0)

ax.scatter(noise_vals_best_idx, range(len(prop_noises)), color='red')
ax.scatter(range(len(noise_vals)), prop_noises_best_idx, color='red')

# %%

# Figure
fig, ax = plt.subplots(figsize=(10, 3))

# Noise parameter vs. Noise proportion
ax.scatter(prop_noises, noise_vals_best)
# ax.axhline(data_val, ls='dotted', label='Data Variance (log10)')
ax.set(
    xlabel='Noise Proportion',
    ylabel=f'Best Noise {param_name}'
)
# ax.legend(loc='lower right')

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / "quadrature_noiseval_vs_noiseprop.pdf")
# %%
# Noise Proportion vs. Noise Parameter

estimation = "singlence"
generations = ["gaussian_mean", "gaussian_var", "gaussian_corr"]
param_names = ["Mean", "Variance (log10)", "Correlation"]
data_vals_desired = [0., np.log10(1.), 0.3]
subsample_factor = 4

# Figure
fig, axes = plt.subplots(nrows=len(generations), figsize=(7.2, 7))
fontsize = 16
params = {'font.size': fontsize}
plt.rcParams.update(params)

color = 'blue'


for generation, param_name, data_val_desired, ax in zip(generations, param_names, data_vals_desired, axes):

    sel = results_quadrature.generation == generation
    all_data_vals = results_quadrature.loc[sel, "data_val"].unique()
    idx_data_val = np.abs(all_data_vals - data_val_desired).argmin()
    data_val = all_data_vals[idx_data_val]

    sel = (results_quadrature['estimation'] == estimation) & \
        (results_quadrature['generation'] == generation) & \
        (results_quadrature['data_val'] == data_val) & \
        (results_quadrature['noise_type'] == 'data')
    prop_noises = results_quadrature.loc[sel, "prop_noise"].unique()
    noise_vals = results_quadrature.loc[sel, "noise_val"].unique()

    data = results_quadrature[sel][['noise_val', 'prop_noise', 'logmse']].set_index(['noise_val', 'prop_noise'])
    logmse = data.unstack().values[:, :]  # shape (noise_vals, prop_noises)

    noise_vals_best_idx = np.argmin(logmse, axis=0)   # shape (prop_noises)
    prop_noises_best_idx = np.argmin(logmse, axis=1)  # shape (noise_vals)

    noise_vals_best = noise_vals[noise_vals_best_idx]   # shape (prop_noises)
    prop_noises_best = prop_noises[prop_noises_best_idx]   # shape (prop_noises)

    # Noise proportion vs. Noise parameter
    ax.plot(noise_vals[::subsample_factor], prop_noises_best[::subsample_factor], color=color)
    ax.axvline(data_val, color=color, ls='dotted', label='Data parameter')
    ax.axhline(0.5, ls='-', color='grey', lw=0.5)
    ax.set(
        xlabel=f'Noise {param_name}',
        ylabel='Optimal \n Noise Prop.',
        # yticks=[0., 0.5],
        # xticks=[-1., -0.5, 0., 0.5, 1.]
    )
    # ax.grid("on")

    # # Prediction
    # if generation == "gaussian_var":
    #     ax.plot(np.log10(param_noises), noiseprops, color='black')

axes[0].set_xlim(-10, 10)
axes[0].legend(loc='upper left')
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(
#     handles, labels, loc='upper center',
#     bbox_to_anchor=(0.48, 1.05),
#     ncol=1)


fig.tight_layout()
plt.savefig(
    IMAGE_FOLDER / "noiseprop_vs_noiseval.pdf",
    bbox_inches='tight'
)
# %%
from nlica.quadrature.singleparam.gaussian_var import get_best_noiseprop

param_noises = np.logspace(-3, 2, 20)
noiseprops = []
for param_noise in param_noises:
    noiseprop = get_best_noiseprop(param_noise=param_noise)
    noiseprops.append(noiseprop)

# %%

# %%

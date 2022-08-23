# %%
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import norm

from nlica.distribution.multivariate import Histogram
from nlica.generation.singleparam.gaussian_mean import get_model as get_model_gaussian_mean
from nlica.generation.singleparam.gaussian_var import get_model as get_model_gaussian_var
from nlica.generation.singleparam.gaussian_corr import get_model as get_model_gaussian_corr
from nlica.defaults import RESULTS_FOLDER, IMAGE_FOLDER
# from nlica.quadrature_gaussian_mean_new import _noise_pdf as noise_pihlaja_mean
# from nlica.quadrature_gaussian_var_new import _noise_pdf as noise_pihlaja_var


# %%
# Get uniform samples
xs = torch.linspace(start=-5, end=5, steps=1000)[:, None]


# Noise pihlaja
@np.vectorize
def noise_pihlaja_mean(x):
    score = -x
    invfisher = 1.  # don't know but just a scaling
    return norm.pdf(x) * np.abs(score * invfisher)


@np.vectorize
def noise_pihlaja_var(x):
    score = (x**2 - 1) / 2
    invfisher = 2.
    return norm.pdf(x) * np.abs(score * invfisher)


all_noise_pihlaja = {
    "gaussian_mean": noise_pihlaja_mean,
    "gaussian_var": noise_pihlaja_var
}

param_names = {
    "gaussian_mean": "Mean",
    "gaussian_var": "Variance"
}

# %%
# 1D Gaussian Mean or Variance
generation = "gaussian_mean"
data_val = 0.
prop_noise = 0.9
noise_pihlaja = all_noise_pihlaja[generation]
param_name = param_names[generation]

# Load data and noise distributions
results = torch.load(
    RESULTS_FOLDER / "optimal_noise" / generation /
    f"expe_optimal_noise_data_val_{data_val}_prop_noise_{prop_noise}.th"
)
n_bins, n_comp = results['n_bins'], results['n_comp']

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 3))

# Data
xs = torch.linspace(start=-5, end=5, steps=1000)[:, None]  # uniform samples
data_marginal, _ = get_model_gaussian_mean(n_comp=n_comp, n_samples=5, seed=0)
ys_data = torch.exp(data_marginal.log_prob(xs))

# Noise
noise_marginal = Histogram(n_bins=n_bins, n_comp=n_comp)
noise_marginal.load_state_dict(results['noise_marginal_statedict'])
ys_noise = torch.exp(noise_marginal.log_prob(xs))

# Reformat
xs = torch.linspace(start=-5, end=5, steps=1000)[:, None]  # uniform
xs = xs.flatten().data.detach().numpy()
ys_data = ys_data.data.detach().numpy()
ys_noise = ys_noise.data.detach().numpy()
ys_zeros = np.zeros_like(ys_noise)

# Pihlaja noise
ys_noise_pihalaja = noise_pihlaja(xs)

# normalize
delta_x = xs[1] - xs[0]
ys_noise = ys_noise / (ys_noise.sum() * delta_x)
ys_noise_pihalaja = ys_noise_pihalaja / (ys_noise_pihalaja.sum() * delta_x)

# Plot
ax.plot(xs, ys_data, color='blue', alpha=1, label='data', lw=2)
ax.fill_between(xs, ys_noise, ys_zeros, label='noise', color='red', alpha=0.15)
ax.plot(xs, ys_noise_pihalaja, label='noise (predicted)', color='black', alpha=1, ls='--', lw=2)

ax.set(
    xlabel="",
    ylabel=f"1D Gaussian ({param_name})" + "\n \n" + "Probability",
    yticks=[0., 0.4]  # hardcoded
)

ax.legend()

# # create legend (manually)
# blue_patch = mpatches.Patch(color='blue', label='data')
# red_patch = mpatches.Patch(color='red', label='noise')
# purple_patch = mpatches.Patch(color='purple', label='noise (pihlaja)')
# ax.legend(handles=[blue_patch, red_patch, purple_patch], loc='upper left')

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / f"optimal_noise_{generation}_dataval_{data_val}_propnoise_{prop_noise}.pdf")


# %%
# 2D Gaussian Correlation
generation = "gaussian_corr"

data_vals = [0.] + [0.3] * 3
prop_noises = [0.5] + [0.1, 0.5, 0.9]


def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)


# Initialize plot
fig, axes = plt.subplots(
    nrows=len(prop_noises), ncols=2, figsize=(7.2, 9)
)
fontsize = 16
params = {'font.size': fontsize}
plt.rcParams.update(params)

for ax, (data_val, prop_noise) in zip(axes, zip(data_vals, prop_noises)):

    # Load data and noise distributions
    results = torch.load(
        RESULTS_FOLDER / "optimal_noise" / generation /
        f"expe_optimal_noise_data_val_{data_val}_prop_noise_{prop_noise}.th"
    )
    n_bins, n_comp = results['n_bins'], results['n_comp']

    # Get uniform samples
    x1s = torch.linspace(-5, 5, 100)
    x2s = torch.linspace(-5, 5, 100)
    x1s, x2s = torch.meshgrid(x1s, x2s)
    xs = torch.stack([x1s.flatten(), x2s.flatten()], dim=-1)

    # Noise
    noise_marginal = Histogram(n_bins=n_bins, n_comp=n_comp)
    noise_marginal.load_state_dict(results['noise_marginal_statedict'])
    ys_noise = torch.exp(noise_marginal.log_prob(xs))

    # Data
    data_marginal, _ = get_model_gaussian_corr(n_comp=n_comp, n_samples=5, seed=0,
                                               data_val=data_val)
    ys_data = torch.exp(data_marginal.log_prob(xs))

    # Reformat
    xs = xs.data.detach().numpy()
    ys_data = ys_data.data.detach().numpy()
    ys_noise = ys_noise.data.detach().numpy()
    y_data_max = trunc(ys_data.max(), 2)
    y_noise_max = trunc(ys_noise.max(), 2)

    # Plot
    im = ax[0].scatter(xs[:, 0], xs[:, 1], c=ys_data, cmap='Blues', alpha=0.3, label='data')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[0., y_data_max])

    im = ax[1].scatter(xs[:, 0], xs[:, 1], c=ys_noise, cmap='Reds', alpha=0.3, label='noise')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[0., y_noise_max])
    cbar.ax.set_ylabel('Probability', rotation=270, labelpad=20)

    # esthetics
    ax[0].set(
        ylabel=rf"${int(prop_noise * 100)}\%$ Noise"
    )
    ax[0].set_aspect('equal', 'box')
    ax[1].set_aspect('equal', 'box')


# # ylabel = "2D Gaussian (Correlation)" + "\n\n" + ""
# ylabel = ""
# axes[0].set_ylabel(ylabel)

# create legend (manually)
blue_patch = mpatches.Patch(color='blue', label='data')
red_patch = mpatches.Patch(color='red', label='noise')

fig.legend(
    handles=[blue_patch, red_patch], loc='upper center',
    bbox_to_anchor=(0.48, 1.05),
    ncol=2)


# # esthetics
# for ax in axes.flatten():
#     if ax is not axes[0]:
#         ax.set_yticklabels([])
#     if ax is not axes[-1]:
#         ax.set_xticklabels([])

fig.tight_layout(pad=1.)
plt.savefig(
    IMAGE_FOLDER / "optimal_noise_correlation.pdf",
    bbox_inches='tight',
)


# %%
# Panel figure
generation = "gaussian_var"
data_val = 1.
prop_noises = [0.1, 0.5, 0.9]
noise_pihlaja = all_noise_pihlaja[generation]
param_name = param_names[generation]

# Initialize plot
fig, axes = plt.subplots(nrows=len(prop_noises), figsize=(7.2, 6), sharex=True)
fontsize = 16
params = {
    'font.size': fontsize,
}
plt.rcParams.update(params)


for ax, prop_noise in zip(axes.flatten(), prop_noises):

    # nu = np.round(prop_noise / (1 - prop_noise), 1)

    # Load data and noise distributions
    results = torch.load(
        RESULTS_FOLDER / "optimal_noise" / generation /
        f"expe_optimal_noise_data_val_{data_val}_prop_noise_{prop_noise}.th"
    )
    n_bins, n_comp = results['n_bins'], results['n_comp']

    # Data
    xs = torch.linspace(start=-5, end=5, steps=1000)[:, None]  # uniform samples
    data_marginal, _ = get_model_gaussian_mean(n_comp=n_comp, n_samples=5, seed=0)
    ys_data = torch.exp(data_marginal.log_prob(xs))

    # Noise
    noise_marginal = Histogram(n_bins=n_bins, n_comp=n_comp)
    noise_marginal.load_state_dict(results['noise_marginal_statedict'])
    ys_noise = torch.exp(noise_marginal.log_prob(xs))

    # Reformat
    xs = torch.linspace(start=-5, end=5, steps=1000)[:, None]  # uniform
    xs = xs.flatten().data.detach().numpy()
    ys_data = ys_data.data.detach().numpy()
    ys_noise = ys_noise.data.detach().numpy()
    ys_zeros = np.zeros_like(ys_noise)

    # Pihlaja noise
    ys_noise_pihalaja = noise_pihlaja(xs)

    # normalize
    delta_x = xs[1] - xs[0]
    ys_noise = ys_noise / (ys_noise.sum() * delta_x)
    ys_noise_pihalaja = ys_noise_pihalaja / (ys_noise_pihalaja.sum() * delta_x)

    # Plot
    ax.plot(xs, ys_data, color='blue', alpha=1, label='data', lw=2)
    ax.fill_between(xs, ys_noise, ys_zeros, label='noise', color='red', alpha=0.15)

    # esthetics
    ax.set(
        xlabel="",
        ylabel=rf"${int(prop_noise * 100)}\%$ Noise",
    )

# esthetics
for ax in axes:
    if ax == axes[-1]:
        ax.plot(xs, ys_noise_pihalaja, label='noise (predicted)', color='black', alpha=1, ls='--', lw=2)
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
                  fancybox=False, shadow=False, ncol=5)
    if ax == axes[0]:
        ax.axvline(np.sqrt(5), label='noise (predicted)', color='black', alpha=1, ls='--', lw=2)
        ax.axvline(-np.sqrt(5), color='black', alpha=1, ls='--', lw=2)

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / f"optimal_noise_{generation}_dataval_{data_val}_panel.pdf")


# %%
# Panel figure 2
generation = "gaussian_mean"
data_val = 0.
prop_noises = [0.1, 0.5, 0.9]
noise_pihlaja = all_noise_pihlaja[generation]
param_name = param_names[generation]

# Initialize plot
fig, axes = plt.subplots(nrows=len(prop_noises), figsize=(7.2, 6), sharex=True)
fontsize = 16
params = {
    'font.size': fontsize,
}
plt.rcParams.update(params)


for ax, prop_noise in zip(axes.flatten(), prop_noises):

    # Load data and noise distributions
    results = torch.load(
        RESULTS_FOLDER / "optimal_noise" / generation /
        f"expe_optimal_noise_data_val_{data_val}_prop_noise_{prop_noise}.th"
    )
    n_bins, n_comp = results['n_bins'], results['n_comp']

    # Data
    xs = torch.linspace(start=-5, end=5, steps=1000)[:, None]  # uniform samples
    data_marginal, _ = get_model_gaussian_mean(n_comp=n_comp, n_samples=5, seed=0)
    ys_data = torch.exp(data_marginal.log_prob(xs))

    # Noise
    noise_marginal = Histogram(n_bins=n_bins, n_comp=n_comp)
    noise_marginal.load_state_dict(results['noise_marginal_statedict'])
    ys_noise = torch.exp(noise_marginal.log_prob(xs))

    # Reformat
    xs = torch.linspace(start=-5, end=5, steps=1000)[:, None]  # uniform
    xs = xs.flatten().data.detach().numpy()
    ys_data = ys_data.data.detach().numpy()
    ys_noise = ys_noise.data.detach().numpy()
    ys_zeros = np.zeros_like(ys_noise)

    # Pihlaja noise
    ys_noise_pihalaja = noise_pihlaja(xs)

    # normalize
    delta_x = xs[1] - xs[0]
    ys_noise = ys_noise / (ys_noise.sum() * delta_x)
    ys_noise_pihalaja = ys_noise_pihalaja / (ys_noise_pihalaja.sum() * delta_x)

    # Plot
    ax.plot(xs, ys_data, color='blue', alpha=1, label='data', lw=2)
    ax.fill_between(xs, ys_noise, ys_zeros, label='noise', color='red', alpha=0.15)

    # esthetics
    ax.set(
        xlabel="",
        ylabel=rf"${int(prop_noise * 100)}\%$ Noise",
    )

for ax in axes:
    if ax == axes[-1]:
        ax.plot(xs, ys_noise_pihalaja, label='noise (predicted)', color='black', alpha=1, ls='--', lw=2)
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
                  fancybox=False, shadow=False, ncol=5)
    if ax == axes[0]:
        ax.axvline(np.sqrt(2), label='noise (predicted)', color='black', alpha=1, ls='--', lw=2)
        ax.axvline(-np.sqrt(2), color='black', alpha=1, ls='--', lw=2)

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / f"optimal_noise_{generation}_dataval_{data_val}_panel.pdf")

# %%

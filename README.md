# The Optimal Noise in Noise-Contrastive Learning Is Not What You Think

### How to install

From within your local repository, run

```bash
# Create and activate an environment
conda env create -f environment.yml
conda activate nce

# Install the package
python setup.py develop
```

### How to replicate Figures 1, 4, and 5
With current parameters, this experiment takes about 2 days
and a maximum of 20GB of RAM to replicate.

```bash
# Simulate the Mean-Squared Error (MSE) using quadrature
python experiments/run_quadrature.py

# Plot the Mean-Squared Error's (MSE) dependency on hyperparameters
python experiments/plot_quadrature.py
```

### How to replicate Figures 2 and 3
With current parameters, this experiment takes about 2 days
and 300GB of RAM to replicate.

```bash
# Numerically optimize the Mean-Squared Error (MSE)
python experiments/run_optimal_noise.py

# Plot the numerically optimal noise
python experiments/plot_optimal_noise.py
```

### Reference

If you use this code in your project, please cite:

```bib
@misc{chehab2022ncenoisevar,
  author = {Chehab, Omar and Gramfort, Alexandre and Hyvarinen, Aapo},
  title = {The Optimal Noise in Noise-Contrastive Learning Is Not What You Think},
  publisher = {arXiv},
  year = {2022},
}
```


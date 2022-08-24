# The Optimal Noise in Noise-Contrastive Learning Is Not What You Think

Code for the paper <a href="https://arxiv.org/abs/2203.01110" target="_blank">The Optimal Noise in Noise-Contrastive Learning Is Not What You Think</a>.

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
@InProceedings{chehab2022ncenoisevar,
  title = 	 {The optimal noise in noise-contrastive learning is not what you think},
  author =       {Chehab, Omar and Gramfort, Alexandre and Hyv{\"a}rinen, Aapo},
  booktitle = 	 {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {307--316},
  year = 	 {2022},
  editor = 	 {Cussens, James and Zhang, Kun},
  volume = 	 {180},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {01--05 Aug},
  publisher =    {PMLR},
}
```


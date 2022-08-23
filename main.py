"""Main script."""

from pathlib import Path
import torch
import argparse

from nlica.runner import run_model
from nlica.defaults import DEFAULTS
from nlica.utils import (
    get_name_from_dict,
)

torch.set_default_dtype(torch.float64)


def get_parser():
    """Set parameters for the experiment."""

    parser = argparse.ArgumentParser("nce",
                                     description="Do Inference with \
                                        Noise-Contrastive Learning.")
    # Experiment Settings
    parser.add_argument("--seed-data-gen", type=int, default=DEFAULTS['seed_data_gen'])
    parser.add_argument("--seed-dataset", type=int, default=DEFAULTS['seed_dataset'])
    parser.add_argument("--seed_data_testset", type=int, default=DEFAULTS['seed_data_testset'])

    # Repositories
    parser.add_argument("-o", "--out", type=Path, default=DEFAULTS['out'])

    # Data Generation
    parser.add_argument("--noise-to-data", type=float, default=DEFAULTS['noise_to_data'])
    parser.add_argument("--n-samples", type=int, default=DEFAULTS['n_samples'])
    parser.add_argument("--n_samples_data_montecarlo", type=int, default=DEFAULTS['n_samples_data_montecarlo'])
    parser.add_argument("--n-comp", type=int, default=DEFAULTS['n_comp'])

    parser.add_argument("--data-val", type=float, default=DEFAULTS['data_val'])  # TODO: solve type pb here

    parser.add_argument("--noise-type", default=DEFAULTS['noise_type'],
                        help="Other options are: gaussian, flexible. "
                             "if gaussian, the noise is a Gaussian with same cov as the observed data "
                             "if flexible, the noise is a learnable histogram.")
    parser.add_argument("--noise-val", type=float, default=DEFAULTS["noise_val"],
                        help="Effective only when noise-type is set to likedata.")
    parser.add_argument("--noise-bins", type=int, default=DEFAULTS["noise_bins"],
                        help="Number of bins for the noise distribution when it is a learnable histogram.")

    # Training
    parser.add_argument("--supervised", action='store_true',
                        help="Computes a (RandomForest) estimate of the Bayes-optimal "
                             "accuracy, as an upper-bound on a supervised task.")
    parser.add_argument("-e", "--n-epochs", type=int, default=DEFAULTS['n_epochs'])
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULTS['batch_size'])
    parser.add_argument("--lr", type=float, default=DEFAULTS['lr'])
    parser.add_argument("--track", type=int, default=DEFAULTS['track'],
                        help='Track downstream task performance '
                             '(e.g. source recovery) every <track> epochs.')
    parser.add_argument("--perturb_init", action='store_true')
    # parser.add_argument("--optimizer", type=str, default="conjugategrad",
    #                     help="Other choices: sgd, adam.")

    # Sampling and Model
    parser.add_argument("--estimation", type=str, default=DEFAULTS['estimation'],
                        help="Other choices: singlemle.")
    parser.add_argument("--cost", type=str, default=DEFAULTS['cost'],
                        help="Other choices: asympmse.")
    parser.add_argument("--train-data", action='store_true')
    parser.add_argument("--train-noise", action='store_true')

    # Asymptotic variance: theoretical
    parser.add_argument("--predict-mse", action='store_true',
                        help="Compute (theoretical) asymptotic variance of the unmixing estimator.")

    # Generative model
    parser.add_argument("--generation", type=str, default=DEFAULTS['generation'],
                        help="The other options are 'ica_rot' and 'gaussian_corr'.")

    # Additional description
    parser.add_argument("--experiment", type=str, default=DEFAULTS['experiment'],
                        help="Enter a sup-repo name for an experiment.")
    parser.add_argument("--hue", type=str, default=DEFAULTS['hue'],
                        help="Enter a sub-repo name for an experiment.")
    parser.add_argument("--describe", type=str, default=DEFAULTS['describe'],
                        help="Any additional information you might "
                             "want to add.")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--with-plot", action='store_true')
    return parser


if __name__ == "__main__":

    # Experiment name
    parser = get_parser()
    args = parser.parse_args()  # you can modify this namespace for quick iterations

    # Rename output directory
    name = get_name_from_dict(args.__dict__)
    args.out = args.out / args.experiment / args.estimation / args.hue / name

    run_model(**args.__dict__)

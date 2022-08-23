from pathlib import Path

# Create folders
RESULTS_FOLDER = Path(Path(__file__).parent.parent / "results")
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)

IMAGE_FOLDER = RESULTS_FOLDER / "images"
IMAGE_FOLDER.mkdir(exist_ok=True, parents=True)

CACHE_FOLDER = RESULTS_FOLDER / "cache"

DUMP_FOLDER = RESULTS_FOLDER / "dump"
DUMP_FOLDER.mkdir(exist_ok=True, parents=True)

# Dictionary of experiment hyperparameters
# WARNING: booleans are not registered in parser defaults
DEFAULTS = {
    # Experiment Settings
    "seed_data_gen": 0,
    "seed_dataset": 0,
    "seed_data_testset": 2000,
    # Repositories
    "out": DUMP_FOLDER,
    # Data Generation
    "noise_to_data": 1,
    "n_samples": 1000,
    "n_samples_data_montecarlo": 10000,
    "n_comp": 1,
    "data_val": "default",
    "noise_type": "likedata",
    "noise_val": 1.,
    "noise_bins": 20,
    # Training
    "supervised": False,
    "n_epochs": 100,
    "batch_size": 64,
    "lr": 1e-1,
    "track": 5,
    "perturb_init": False,
    # Sampling and Model
    "estimation": "singlence",
    "cost": "default",
    "train_data": False,
    "train_noise": False,
    # Asymptotic variance: theoretical
    "predict_mse": False,
    # Generative model
    "generation": "gaussian_var",
    # Additional description
    "experiment": "",
    "hue": "",
    "describe": "",
    "verbose": False,
    "with_plot": False
}

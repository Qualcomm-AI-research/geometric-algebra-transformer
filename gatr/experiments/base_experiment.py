#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm, trange

import gatr.primitives.attention
import gatr.utils.logger
from gatr.layers import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig
from gatr.utils.logger import logger
from gatr.utils.misc import NaNError, flatten_dict, frequency_check, get_device
from gatr.utils.mlflow import log_mlflow
from gatr.utils.plotting import MATPLOTLIB_PARAMS

cs = ConfigStore.instance()
cs.store(name="base_attention", node=SelfAttentionConfig)
cs.store(name="base_mlp", node=MLPConfig)


class BaseExperiment:
    """Base experiment manager class.

    To be subclassed by experiment-specific manager classes.

    Parameters
    ----------
    cfg : OmegaConf
        Experiment configuration. See the config folder in the repository for examples.
    """

    def __init__(self, cfg):
        # Store config
        self.cfg = cfg

        # Device, dtype, backend
        self.device, self.dtype = self._init_backend()

        # Initialize state
        self.model: Optional[nn.Module] = None
        self.ema = None
        self.metrics = {}
        self._best_state = None
        self._training_start_time: Optional[float] = None

        # Initialize folder and logger
        self._initialize_experiment_folder()
        self._initialize_logger()
        self._silence_the_lambs()

    def __call__(self, train=True, evaluate=True):
        """Performs experiment as outlined below.

        - initializes all the logistics
        - instantiates model (if necessary)
        - loads checkpoint (if applicable)
        - trains model (if `train`)
        - evaluates model (if `eval`)

        Parameters
        ----------
        train : bool
            Whether to train the model.
        evaluate : bool
            Whether to evaluate the model.
        """

        experiment_id = self._initialize_experiment()

        with mlflow.start_run(experiment_id=experiment_id, run_name=self.cfg.run_name):
            self._save_config()

            # Create / load model
            if self.model is None:
                self.create_model()
                self.load_model()

            # Train
            if train:
                self.train()
                # Save model
                self.save_model("model_final.pt")

            # Evaluate
            if evaluate:
                self.evaluate()

        logger.info("Anders nog iets?")
        return self.metrics

    def create_model(self):
        """Create self.model according to the specification in self.cfg."""

        # Create model
        self.model = self._create_model()
        assert self.model is not None

        # Report number of parameters
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log_mlflow("efficiency.num_parameters", float(num_parameters), step=0)
        logger.info(f"Model has {num_parameters / 1e6:.2f}M learnable parameters")

        # Create exponential moving average object
        if self.cfg.training.ema:
            logger.info("Using EMA for validation and eval")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            )
        else:
            logger.debug("Not using EMA")
            self.ema = None

    def load_model(self, checkpoint=None):
        """Loads a model checkpoint from disk.

        Parameters
        ----------
        checkpoint : None or str or pathlib.Path
            Path to checkpoint. If None, self.cfg.checkpoint is used.
        """

        # If model hasn't been created yet, do that first
        if self.model is None:
            self.create_model()
        assert self.model is not None

        if checkpoint is None:
            checkpoint = self.cfg.checkpoint

        if checkpoint is not None:
            logger.info(f"Loading model checkpoint from {checkpoint}")
            state_dict = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state_dict)

            if self.cfg.training.ema:
                ema_checkpoint = checkpoint.replace(".pt", "_ema.pt")
                logger.info(f"Loading EMA checkpoint from {ema_checkpoint}")
                state_dict = torch.load(ema_checkpoint, map_location="cpu")
                assert self.ema is not None
                self.ema.load_state_dict(state_dict)

    def train(self):
        """High-level training function."""
        assert self.model is not None

        logger.info("Starting training")

        # Prepare data
        train_data = self._load_dataset("train")
        train_loader = self._make_data_loader(
            train_data, batch_size=self.cfg.training.batchsize, shuffle=True
        )
        val_data = self._load_dataset("val")
        eval_batchsize = self.cfg.training.get("eval_batchsize", self.cfg.training.batchsize)
        val_loader = self._make_data_loader(val_data, batch_size=eval_batchsize, shuffle=False)

        # Training
        num_epochs = (self.cfg.training.steps - 1) // (
            (len(train_data) - 1) // self.cfg.training.batchsize + 1
        ) + 1
        logger.info(
            f"Training for {self.cfg.training.steps} steps, that is, {num_epochs} epochs on a "
            f"dataset of size {len(train_data)} with batchsize {self.cfg.training.batchsize}"
        )
        optim, scheduler = self._create_optimizer_and_scheduler()

        # Prepare book-keeping
        train_metrics: Dict[str, Any] = defaultdict(list)
        val_metrics: Dict[str, Any] = defaultdict(list)
        self._best_state = {"state_dict": None, "loss": None, "step": None}
        self._training_start_time = time.time()

        # GPU
        self.model = self.model.to(self.device)
        if self.ema:
            self.ema.to(self.device)

        # Loop over epochs
        step = 0
        for epoch in trange(
            num_epochs, desc="Training epochs", disable=not self.cfg.training.progressbar
        ):
            log_mlflow("train.epoch", epoch, step=step)
            epoch_start = time.perf_counter()
            self.model.train()

            # Loop over steps
            for data in train_loader:
                self._step(data, optim, scheduler, step, val_data, val_loader)
                if step >= self.cfg.training.steps:
                    break
                step += 1

            logger.debug(f"Finished epoch {epoch} in {time.perf_counter() - epoch_start} seconds")

        logger.debug("Finished training")

        # Final validation loop
        if (
            self.cfg.training.validate_every_n_steps is not None
            and self.cfg.training.validate_every_n_steps > 0
        ):
            self.validate(val_loader, step)

        # Wrap up early stopping
        if (
            self.cfg.training.early_stopping
            and self._best_state["step"] is not None
            and self._best_state["step"] < step
        ):
            logger.info(
                f'Early stopping after step {self._best_state["step"]} '
                f'with validation loss {self._best_state["loss"]}'
            )
            self.model.load_state_dict(self._best_state["state_dict"])
        else:
            logger.debug("Not using early stopping")

        return train_metrics, val_metrics

    def validate(self, dataloader, step):
        """Runs validation loop, logs results, and may store state dict for early stopping.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Validation dataloader.
        step : int
            Current step number.
        """
        assert self.model is not None

        # Compute validation metrics, using EMA if available
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics = self._compute_metrics(dataloader)
        else:
            metrics = self._compute_metrics(dataloader)

        # Log
        self.metrics["val"] = metrics
        logger.info(f"Validation loop at step {step}:")
        for key, value in metrics.items():
            log_mlflow(f"val.{key}", value, step=step)
            logger.info(f"    {key} = {value}")

        # Early stopping: compare val loss to last val loss
        new_val_loss = metrics["loss"]
        assert self._best_state is not None
        if self._best_state["loss"] is None or new_val_loss < self._best_state["loss"]:
            self._best_state["loss"] = new_val_loss
            self._best_state["state_dict"] = self.model.state_dict().copy()
            self._best_state["step"] = step

    def evaluate(self):
        """Evaluates self.model on all eval datasets and logs the results."""

        # Should we evaluate with EMA in addition to without?
        ema_values = [False]
        if self.ema is not None:
            ema_values.append(True)

        # Loop over evaluation datasets
        dfs = {}
        for tag in self._eval_dataset_tags:
            dataset = self._load_dataset(tag)
            eval_batchsize = self.cfg.training.get("eval_batchsize", self.cfg.training.batchsize)
            dataloader = self._make_data_loader(dataset, batch_size=eval_batchsize, shuffle=False)

            # Loop over EMA on / off
            for ema in ema_values:
                # Effective tag name
                full_tag = (tag + "_ema") if ema else tag

                # Run evaluation
                if ema:
                    with self.ema.average_parameters():
                        metrics = self._compute_metrics(dataloader)
                else:
                    metrics = self._compute_metrics(dataloader)

                # Log results
                self.metrics[full_tag] = metrics
                logger.info(f"Ran evaluation on dataset {full_tag}:")
                for key, val in metrics.items():
                    logger.info(f"    {key} = {val}")
                    log_mlflow(f"eval.{full_tag}.{key}", val)

                # Store results in csv file
                # Pandas does not like scalar values, have to be iterables
                test_metrics_ = {key: [val] for key, val in metrics.items()}
                df = pd.DataFrame.from_dict(test_metrics_)
                df.to_csv(Path(self.cfg.exp_dir) / "metrics" / f"eval_{full_tag}.csv", index=False)
                dfs[full_tag] = df
        return dfs

    def save_model(self, filename=None):
        """Save model in experiment folder.

        Parameters
        ----------
        filename : None or str
            Filename to save the model to.
        """

        assert self.model is not None
        if filename is None:
            filename = "model.pt"

        model_path = Path(self.cfg.exp_dir) / "models" / filename
        logger.info(f"Saving model at {model_path}")
        torch.save(self.model.state_dict(), model_path)

        if self.ema is not None:
            ema_path = Path(self.cfg.exp_dir) / "models" / filename.replace(".pt", "_ema.pt")
            logger.debug(f"Saving EMA model at {ema_path}")
            assert ema_path != model_path
            torch.save(self.ema.state_dict(), ema_path)

    def _init_backend(self):
        """Initializes device, dtype, and attention implementation."""
        device = get_device()
        if self.cfg.training.float16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("Training on bfloat16")
        elif self.cfg.training.float16:
            dtype = torch.float16
            logger.info("Training on float16 (bfloat16 is not supported by environment)")
        else:
            dtype = torch.float32
            logger.info("Training on float32")

        torch.backends.cuda.enable_flash_sdp(self.cfg.training.enable_flash_sdp)
        torch.backends.cuda.enable_math_sdp(self.cfg.training.enable_math_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(self.cfg.training.enable_mem_efficient_sdp)
        if self.cfg.training.force_xformers:
            logger.debug("Forcing use of xformers' attention implementation")
            gatr.primitives.attention.FORCE_XFORMERS = True

        return device, dtype

    def _initialize_experiment(self):
        """Initialize experiment folder (and plenty of other initialization thingies)."""

        # Initialize MLflow experiment
        experiment_id = self._initialize_mlflow()

        # Silence other loggers (thanks MLflow)
        self._silence_the_lambs()

        # Re-initialize logger - this is annoying, something in the way that MLflow interacts with
        # logging means that we otherwise get duplicate logging to stdout
        self._initialize_logger()

        # Print config to log
        logger.info(f"Running experiment at {self.cfg.exp_dir}")
        logger.info(f"Config: \n{OmegaConf.to_yaml(self.cfg)}")

        # Set random seed
        torch.random.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Initialize plotting
        self._init_plt()

        return experiment_id

    def _initialize_experiment_folder(self):
        """Creates experiment folder."""

        exp_dir = Path(self.cfg.exp_dir).resolve()
        subfolders = [
            exp_dir / "models",
            exp_dir / "figures",
            exp_dir / "metrics",
            exp_dir / "data",
        ]

        # Create experiment subfolders (main folder will be automatically created as well)
        for subdir in subfolders:
            if subdir.exists():
                logger.warning(f"Warning: directory {subdir} already exists!")
            subdir.mkdir(parents=True, exist_ok=True)

    def _initialize_logger(self):
        """Initializes logging."""

        # In sweeps (multiple experiments in one job) we don't want to set up the handlers again
        if gatr.utils.logger.logging_initialized:
            logger.info("Logger already initialized - hi again!")
            return

        logger.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)-19.19s %(levelname)-1.1s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(Path(self.cfg.exp_dir) / "output.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # This is important to avoid duplicate log outputs
        # See https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module # pylint:disable=line-too-long
        logger.propagate = False

        gatr.utils.logger.logging_initialized = True
        logger.info("Hoi.")

    def _initialize_mlflow(self):
        """Initializes all things related to MLflow tracking."""

        # Set up MLflow tracking location
        Path(self.cfg.mlflow.db).parent.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{Path(self.cfg.mlflow.db).resolve()}")

        # Set up MLflow experiment
        Path(self.cfg.mlflow.artifacts).mkdir(exist_ok=True)
        try:
            experiment_id = mlflow.create_experiment(
                self.cfg.exp_name,
                artifact_location=f"file:{Path(self.cfg.mlflow.artifacts).resolve()}",
            )
            logger.info(f"Created experiment {self.cfg.exp_name} with ID {experiment_id}")
        except mlflow.exceptions.MlflowException:
            pass  # Experiment exists already

        # Set MLflow experiment details
        experiment = mlflow.set_experiment(self.cfg.exp_name)
        experiment_id = experiment.experiment_id
        artifact_loc = experiment.artifact_location

        logger.info(
            f"Set experiment {self.cfg.exp_name} with ID {experiment_id},"
            f" artifact location {artifact_loc}"
        )

        return experiment_id

    @staticmethod
    def _silence_the_lambs():
        """Silences other loggers."""
        for name, other_logger in logging.root.manager.loggerDict.items():
            if not "gatr_experiment" in name:
                other_logger.level = logging.WARNING

    @staticmethod
    def _init_plt():
        """Initializes matplotlib's rcparams to look good."""

        sns.set_style("whitegrid")
        matplotlib.rcParams.update(MATPLOTLIB_PARAMS)

    def _save_config(self):
        """Stores the config in the experiment folder and tracks it with mlflow."""

        # Save config
        config_filename = Path(self.cfg.exp_dir) / "config.yml"
        logger.info(f"Saving config at {config_filename}")
        with open(config_filename, "w", encoding="utf-8") as file:
            file.write(OmegaConf.to_yaml(self.cfg))

        # Store config as MLflow params
        for key, value in flatten_dict(self.cfg).items():
            log_mlflow(key, value, kind="param")

    def _create_model(self):
        """Creates the model from the config.

        Returns
        -------
        model : torch.nn.Module
            A randomly initialized model, following the specification in self.cfg.model
        """

        # Hydra magic!
        return instantiate(self.cfg.model)

    def _create_optimizer_and_scheduler(self):
        """Creates optimizer and scheduler.

        Returns
        -------
        optim : torch.optim.Optimizer
            Adam optimizer for the parameters of self.model.
        sched : torch.optim.lr_scheduler.LRScheduler
            Exponential LR scheduler for optim.
        """

        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        num_scheduler_steps = self.cfg.training.steps // self.cfg.training.update_lr_every_n_steps
        if num_scheduler_steps > 0:
            gamma = self.cfg.training.lr_decay ** (1 / num_scheduler_steps)
        else:
            gamma = 1.0
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

        return optim, scheduler

    def _make_data_loader(self, dataset, batch_size, shuffle):
        """Creates a data loader.

        Parameters
        ----------
        dataset : torch.nn.utils.data.Dataset
            Dataset.
        batch_size : int
            Batch size.
        shuffle : bool
            Whether the dataset is shuffled.

        Returns
        -------
        dataloader
            Data loader.
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _prep_data(self, data, device=None):
        """Data preparation during training loop, e.g. to move data to correct device and dtype."""
        if isinstance(data, (tuple, list)):
            data = tuple(x.to(device or self.device) for x in data)
        else:
            data = (data.to(device or self.device),)

        return data

    def _step(self, data, optim, scheduler, step, val_data, val_loader):
        """Everything that that may happen per step."""

        # Move data to GPU, and other and other data prep stuff
        data = self._prep_data(data)

        # Forward pass
        with torch.autocast(
            device_type="cuda", dtype=self.dtype, enabled=self.cfg.training.float16
        ):
            ctx = torch.autograd.detect_anomaly if self.cfg.training.detect_anomaly else nullcontext
            with ctx():
                loss, metrics = self._forward(*data)

        # Optimizer step
        grad_norm = self._optimizer_step(loss, optim)

        # Log loss and metrics
        self._log(loss, metrics, grad_norm, step)

        # Debugging output
        if step == 0:
            logger.info(f"Finished first forward pass with loss {loss.item()}")

        # Validation loop
        if frequency_check(step, self.cfg.training.validate_every_n_steps, skip_initial=True):
            logger.info(f"Starting validation at step {step}")
            self.validate(val_loader, step)

        # Plotting
        if frequency_check(step, self.cfg.training.plot_every_n_steps, skip_initial=False):
            self.visualize(val_data, step)

        # Save model checkpoint
        if frequency_check(step, self.cfg.training.save_model_every_n_steps, skip_initial=True):
            self.save_model(f"model_step_{step}.pt")

        # LR scheduler
        if frequency_check(step, self.cfg.training.update_lr_every_n_steps, skip_initial=True):
            scheduler.step()
            logger.debug(f"Decaying LR to {scheduler.get_last_lr()[0]}")
            log_mlflow("train.lr", scheduler.get_last_lr()[0], step=step)

        return step

    def _optimizer_step(self, loss, optim):
        """Optimizer step and gradient norm clipping."""
        assert self.model is not None
        if not torch.isfinite(loss):
            raise NaNError("NaN in loss!")

        optim.zero_grad()
        loss.backward()

        # Grad norm clipping
        try:
            clip_norm = self.cfg.training.clip_grad_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), clip_norm, error_if_nonfinite=True
            )
            grad_norm = grad_norm.cpu().item()
        except RuntimeError as e:
            for n, p in self.model.named_parameters():
                if not torch.isfinite(p.grad.flatten().norm()):
                    print(f"Non-finite grad in {n}")
            raise e

        optim.step()
        if self.ema is not None:
            self.ema.update()

        return grad_norm

    def _compute_metrics(self, dataloader):
        """Given a dataloader, computes all relevant metrics. Can be adapted by subclasses.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader.

        Returns
        -------
        metrics : dict with str keys and float values
            Metrics computed from dataset.
        """

        # Move to eval mode and eval device
        assert self.model is not None
        self.model.eval()
        eval_device = torch.device(self.cfg.training.eval_device)
        self.model = self.model.to(eval_device)

        aggregate_metrics: Dict[str, float] = defaultdict(float)

        # Loop over dataset and compute error
        for data in tqdm(dataloader, disable=False, desc="Evaluating"):
            data = self._prep_data(data, device=eval_device)

            # Forward pass
            loss, metrics = self._forward(*data)

            # Weight for this batch (last batches may be smaller)
            batchsize = len(data[0])
            weight = batchsize / len(dataloader.dataset)

            # Book-keeping
            aggregate_metrics["loss"] += loss.item() * weight
            for key, val in metrics.items():
                aggregate_metrics[key] += val * weight

        # Move model back to training mode and training device
        self.model.train()
        self.model = self.model.to(self.device)

        # Return metrics
        return aggregate_metrics

    def _log(self, loss, metrics, grad_norm, step):
        """Log to MLflow.

        Parameters
        ----------
        loss : torch.Tensor
            Loss
        metrics : dict with str keys and float values
            Additional metrics for logging
        grad_norm : float
            Gradient norm
        """

        if not frequency_check(step, self.cfg.training.log_every_n_steps):
            return

        metrics["loss"] = loss.item()
        metrics["grad_norm"] = grad_norm
        metrics["step"] = step
        assert self._training_start_time is not None
        metrics["time_total_s"] = time.time() - self._training_start_time
        metrics["time_per_step_s"] = (time.time() - self._training_start_time) / (step + 1)

        for key, values in metrics.items():
            log_mlflow(f"train.{key}", values, step=step)

    def _load_dataset(self, tag):
        """Loads dataset. To be implemented by subclasses.

        Parameters
        ----------
        tag : str
            Dataset tag, like "train", "val", or one of self._eval_tags.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            Dataset.
        """
        raise NotImplementedError

    def _forward(self, *data):
        """Model forward pass. To be implemented by subclasses.

        Parameters
        ----------
        data : tuple of torch.Tensor
            Data batch.

        Returns
        -------
        loss : torch.Tensor
            Loss
        metrics : dict with str keys and float values
            Additional metrics for logging
        """
        raise NotImplementedError

    def visualize(self, dataset, step):
        """Visualization function.

        To be implemented by subclasses.
        """

    @property
    def _eval_dataset_tags(self):
        """Eval dataset tags, to be implemented by subclasses.

        Returns
        -------
        tags : iterable of str
            Eval dataset tags
        """
        return {"eval"}

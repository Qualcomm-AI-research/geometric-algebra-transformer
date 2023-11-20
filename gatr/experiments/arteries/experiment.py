# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from pathlib import Path

import pandas as pd
import torch
import torch_geometric
from coronary_mesh_convolution.transforms import InletGeodesics, RemoveFlowExtensions
from coronary_mesh_convolution.utils.metrics import Metrics

from gatr.experiments import BaseExperiment
from gatr.experiments.arteries.dataset import NumberedVesselDataset, RotationDataset
from gatr.utils.logger import logger

LABEL_SCALE = 53.8360  # the sqrt of average squared norm of training label vectors


class ArteryExperiment(BaseExperiment):
    """Experiment manager for wall-shear-stress estimation in arteries.

    The dataset was introduced and is described in Suk et al, "Mesh Convolutional Neural Networks
    for Wall Shear Stress Estimation in 3D Artery Models".

    References
    ----------
    J. Suk et al., "Mesh Convolutional Neural Networks for Wall Shear Stress Estimation in
        3D Artery Models", arXiv:2109.04797

    Parameters
    ----------
    cfg : OmegaConf
        Experiment configuration. See the config folder in the repository for examples.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._criterion = torch.nn.MSELoss()

        # Data transforms
        transforms = [
            torch_geometric.transforms.GenerateMeshNormals(),
            InletGeodesics(),
            RemoveFlowExtensions(factor=(4.0, 1.0)),
        ]
        self.transform = torch_geometric.transforms.Compose(transforms)
        self.rotations = RotationDataset(2000, special=False, seed=123)

    def rotate_sample(self, data):
        """Rotate data sample with pre-sampled rotation matrices."""
        r = self.rotations[data.shape_id[0]]
        data.pos = data.pos @ r.T
        data.norm = data.norm @ r.T
        data.y = data.y @ r.T
        return data

    def _load_dataset(self, tag):
        """Loads dataset.

        References
        ----------
        J. Suk et al., "Mesh Convolutional Neural Networks for Wall Shear Stress Estimation in
            3D Artery Models", arXiv:2109.04797

        Parameters
        ----------
        tag : str
            Dataset tag, like "train", "val", or one of self._eval_tags.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            Dataset.
        """

        # Train / validation / test splits from Suk et al
        split = {"train": [0, 1600], "val": [1600, 1799], "eval": [1799, 1999]}[tag]

        if (self.cfg.data.rotate_train and tag == "train") or (
            self.cfg.data.rotate_test and tag in ["val", "eval"]
        ):
            transform = self.rotate_sample
        else:
            transform = None

        # Load dataset
        dataset = NumberedVesselDataset(
            Path(self.cfg.data.data_dir),
            "*.hdf5",
            split,
            tag,
            pre_transform=self.transform,
            transform=transform,
        )

        # Subsample dataset if requested
        if tag == "train" and self.cfg.data.subsample is not None and self.cfg.data.subsample < 1.0:
            n = int(round(self.cfg.data.subsample * len(dataset)))
            dataset = dataset.index_select(slice(n))

        # Logging
        logger.info(f"Loaded dataset partition {tag} with length {len(dataset)}")

        return dataset

    def _prep_data(self, data, device=None):
        """Data preparation during training loop, e.g. to move data to correct device and dtype."""

        if device is None:
            device = self.device

        # Only move relevant features to GPU and float16
        data.pos = data.pos.to(device)
        data.norm = data.norm.to(device)
        data.geo = data.geo.to(device)
        data.y = data.y.to(device)

        # NaN debugging
        assert torch.all(torch.isfinite(data.pos))
        assert torch.all(torch.isfinite(data.norm))
        assert torch.all(torch.isfinite(data.geo))
        assert torch.all(torch.isfinite(data.y))

        # Mask should also be on GPU, but remain as bool
        data.mask = data.mask.to(device)
        data.batch = data.batch.to(device)

        return (data,)

    def _forward(self, *data):
        """Model forward pass.

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

        # Forward pass
        assert self.model is not None
        data = data[0]
        y_pred = self.model(data)

        # NaN debugging
        assert torch.all(torch.isfinite(y_pred))

        # Compute loss
        y = data.y.view(*y_pred.shape)
        y_normalized = y / LABEL_SCALE  # normalize labels
        y_pred_unnormalized = y_pred * LABEL_SCALE  # unnormalize predictions
        mse_normalized = self._criterion(y_pred[:, data.mask, :], y_normalized[:, data.mask, :])
        mse = self._criterion(y_pred_unnormalized[:, data.mask, :], y[:, data.mask, :])
        loss = mse_normalized

        # Additional metrics
        metrics = dict(mse=mse.item(), mse_normalized=mse_normalized.item())

        return loss, metrics

    @torch.no_grad()
    def _compute_metrics(self, dataloader):
        """Given a dataloader, computes all relevant metrics. To be implemented by subclasses.

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

        metrics_model = Metrics([dataloader])
        results = []

        # Loop over dataset and compute metrics
        for data in dataloader:
            data = self._prep_data(data)[0]
            model_output = self.model(data)
            y_pred_batch = model_output[0] * LABEL_SCALE  # Scale as ground truth, remove batch idx
            num_meshes = data.batch.max() + 1
            for mesh_i in range(num_meshes):
                batch_mask = data.batch == mesh_i
                # Mask out one particular mesh, remove the in/outlets
                prediction = y_pred_batch[batch_mask & data.mask]
                label = data.y[batch_mask & data.mask]
                _, maximum, mean, _ = metrics_model.absolute_differences(prediction, label)
                scale_max, scale_median = metrics_model.scale(label)
                approximation_error = metrics_model.approximation_error(prediction, label)
                mse = self._criterion(prediction, label)
                mesh_results = dict(
                    delta_max=maximum.item(),
                    delta_mean=mean.item(),
                    nmae=(mean / metrics_model.M).item(),
                    approximation_error=approximation_error.item(),
                    scale_max=scale_max.item(),
                    scale_median=scale_median.item(),
                    mse=mse.item(),
                )
                results.append(mesh_results)
        df = pd.DataFrame(results)
        metrics = {}
        for metric, series in df.items():
            metrics[f"{metric}"] = series.mean()
            metrics[f"{metric}_median"] = series.median()
            metrics[f"{metric}_q75"] = series.quantile(0.75)
        metrics["loss"] = metrics["mse"]

        # Move model back to training mode and training device
        self.model.train()
        self.model = self.model.to(self.device)

        return metrics

    @property
    def _eval_dataset_tags(self):
        """Eval dataset tags.

        Returns
        -------
        tags : iterable of str
            Eval dataset tags
        """
        return {"eval"}

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
        return torch_geometric.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False
        )

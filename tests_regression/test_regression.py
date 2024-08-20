# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""A regression test that tests regression."""


import pytest
import torch
import torch_geometric
from torch_geometric.data import Batch
from tqdm import trange

from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig
from gatr.nets import GATr
from gatr.utils.einsum import enable_cached_einsum
from tests_regression.regression_datasets.constants import BATCHSIZE, DEVICE, NUM_EPOCHS
from tests_regression.regression_datasets.points_distance import (
    PointsDistanceDataset,
    PointsDistanceWrapper,
)
from tests_regression.regression_helpers import XFormersDatasetWrapper, XFormersModelWrapper


def gatr_factory(wrapper_class):
    """Factory function for a GATr model."""
    print("Creating GATr")
    model = GATr(
        in_mv_channels=wrapper_class.mv_in_channels,
        out_mv_channels=wrapper_class.mv_out_channels,
        hidden_mv_channels=8,
        in_s_channels=wrapper_class.s_in_channels,
        out_s_channels=wrapper_class.s_out_channels,
        hidden_s_channels=16,
        attention=SelfAttentionConfig(
            num_heads=4,
            increase_hidden_channels=2,
            multi_query=True,
        ),
        num_blocks=10,
        mlp=MLPConfig(),
    )
    wrapped_model = wrapper_class(model)
    return wrapped_model


@pytest.mark.parametrize("model_factory", [gatr_factory], ids=["GATr"])
@pytest.mark.parametrize(
    "data,wrapper_class", [(PointsDistanceDataset(), PointsDistanceWrapper)], ids=["distance"]
)
@pytest.mark.parametrize("xformers", [True, False])
@pytest.mark.parametrize("torch_compile", [False])
def test_regression(
    model_factory, data, wrapper_class, xformers, torch_compile, lr=3e-4, target_loss=0.1
):
    """Test whether model can successfully regress on a dataset data to almost zero train error."""
    try:
        model = model_factory(wrapper_class)
        if xformers:
            model = XFormersModelWrapper(model)
            data = XFormersDatasetWrapper(data)
            dataloader = torch_geometric.data.DataLoader(data, batch_size=BATCHSIZE, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(data, batch_size=BATCHSIZE, shuffle=True)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        model = model.to(DEVICE)

        if torch_compile:
            enable_cached_einsum(False)
            # TODO: Once we can compile without graph breaks (expand_pairwise adapted),
            # we should compile with `fullgraph=True` in the non-xformer case.
            model = torch.compile(
                model,
                fullgraph=False,
                dynamic=True,
            )

        print("Starting training")
        for _ in (pbar := trange(NUM_EPOCHS)):
            epoch_loss = 0.0
            for batch in dataloader:
                if isinstance(batch, Batch):
                    batch.to(DEVICE)
                    x = batch
                    y = batch.y
                else:
                    x, y = batch
                    x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item() / len(dataloader)
                pbar.set_description(desc=f"Loss = {loss.item():.3f}", refresh=True)

        print(f"Training loss in last epoch: {epoch_loss}")
        assert epoch_loss < target_loss
    finally:
        # Restore to defaults unconditionally
        enable_cached_einsum(True)

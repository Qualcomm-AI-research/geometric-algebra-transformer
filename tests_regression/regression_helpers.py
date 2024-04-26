import torch
from torch_geometric.data import Batch, Data
from xformers.ops.fmha import BlockDiagonalMask


class XFormersDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper that turns a torch Dataset into a torch_geometric-compatible version.

    Assumes that the wrapped dataset returns tuples (x, y), and wraps them in a Data object.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return Data(x=x, y=y)

    def __len__(self):
        return len(self.dataset)


class XFormersModelWrapper(torch.nn.Module):
    """Wrapper that turns a "plain" GATr wrapper into one that uses xformers."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs: Batch):
        """Forward pass of the Xformers model."""
        multivector, scalars = self.model.embed_into_ga(inputs.x)
        attn_mask = BlockDiagonalMask.from_seqlens(torch.bincount(inputs.batch).tolist())
        multivector_outputs, scalar_outputs = self.model.net(
            multivector, scalars=scalars, attention_mask=attn_mask
        )
        outputs, other = self.model.extract_from_ga(multivector_outputs, scalar_outputs)
        outputs = self.model.postprocess_results(inputs, outputs)

        if self.model.return_other:
            return outputs, other

        return outputs

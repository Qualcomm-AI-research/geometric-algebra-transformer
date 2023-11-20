# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
from torch import nn


class BaseWrapper(nn.Module):
    """Base GATr wrapper.

    To be subclassed by experiment-specific wrapper classes.

    Parameters
    ----------
    net : torch.nn.Module
        GATr network.
    scalars : bool
        Whether the GATr model uses auxiliary scalars in its inputs and outputs. (In hidden
        representations, GATr uses auxiliary scalars always.)
    return_other : bool
        Whether the wrapper should return regularization terms in addition to the model predictions.
    """

    def __init__(self, net, scalars=True, return_other=True):
        super().__init__()
        self.net = net
        self.scalars = scalars
        self.return_other = return_other

    def build_attention_mask(
        self, inputs, mv=None, s=None
    ):  # pylint: disable=unused-argument,redundant-returns-doc
        """Construct attention mask.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs to wrapped network.
        mv : torch.Tensor
            Multivector embedding of inputs.
        s : torch.Tensor
            Auxiliary scalar embedding of inputs.

        Returns
        -------
        attention_mask : None or torch.Tensor or xformers.ops.fmha.BlockDiagonalMask
            Attention mask.
        """
        return None

    def forward(self, inputs: torch.Tensor):
        """Wrapped forward pass pass.

        Parses inputs into GA + scalar representation, calls the forward pass of the wrapped net,
        and extracts the outputs from the GA + scalar representation again.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        other : torch.Tensor
            Additional output data, e.g. required for regularization. Only returned if
            `self.return_other`.
        """

        multivector, scalars = self.embed_into_ga(inputs)
        mask = self.build_attention_mask(  # pylint: disable=assignment-from-none
            inputs, mv=multivector, s=scalars
        )
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        outputs, other = self.extract_from_ga(multivector_outputs, scalar_outputs)
        outputs = self.postprocess_results(inputs, outputs)

        if self.return_other:
            return outputs, other

        return outputs

    def embed_into_ga(self, inputs):
        """Embeds raw inputs into the geometric algebra (+ scalar) representation.

        To be implemented by subclasses.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        mv_inputs : torch.Tensor
            Multivector inputs, as expected by geometric network.
        scalar_inputs : torch.Tensor or None
            Scalar inputs, as expected by geometric network.
        """
        raise NotImplementedError

    def extract_from_ga(self, multivector, scalars):
        """Extracts raw outputs from the GATr multivector + scalar outputs.

        To be implemented by subclasses.

        Parameters
        ----------
        multivector : torch.Tensor
            Multivector outputs from GATr.
        scalars : torch.Tensor or None
            Scalar outputs from GATr.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        other : torch.Tensor
            Additional output data, e.g. required for regularization.
        """
        raise NotImplementedError

    def postprocess_results(self, inputs, outputs):  # pylint: disable=unused-argument
        """Postprocesses the outputs extracted from the GA representation.

        To be implemented by subclasses, optionally (by default, no postprocessing is applied).

        Parameters
        ----------
        inputs
            Raw inputs, pre embedding.
        outputs : torch.Tensor
            Raw outputs, pre postprocessing.

        Returns
        -------
        processed_outputs : torch.Tensor
            Raw outputs, after postprocessing.
        """
        return outputs

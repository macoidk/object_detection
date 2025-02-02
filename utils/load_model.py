import os
from typing import Type

import torch
import torch.nn as nn


def load_model(
    device: torch.device,
    model_type: Type[nn.Module],
    checkpoint_path: str = None,
    **kwargs,
) -> nn.Module:
    """
    Loads a PyTorch model from a checkpoint file and moves it to the specified device.

    Args:
        device (torch.device): The device (CPU or GPU) where the model will be loaded.
        model_type (Type[nn.Module]): The class of the model to be instantiated.
        checkpoint_path (str, optional): Path to the model checkpoint file. Defaults to
            "../models/checkpoints/pretrained_weights.pt" if not provided.
        kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        nn.Module: The loaded model instance.

    Raises:
        FileNotFoundError: If the specified checkpoint file does not exist.
    """
    checkpoint_path = (
        "../models/checkpoints/pretrained_weights.pt"
        if checkpoint_path is None
        else checkpoint_path
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = model_type(filters_size=[128, 64, 32], **kwargs).to(device)
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    return model

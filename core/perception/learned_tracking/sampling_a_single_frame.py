"""Sample a one-frame feature volume with its equivalent two-dimensional operation.

On PyTorch 2.11 MPS, a finite one-frame 3D grid sample returned nonfinite
query features during actual online tracking. Border padding makes the
singleton time axis constant, so sampling the image directly gives the
same interpolation without that failing kernel.
"""
from __future__ import annotations

from typing import Any


def map_coordinates_3d(features: Any, coordinates: Any) -> Any:
    """Use the upstream time/y/x convention, including the time half-pixel offset."""
    import torch
    import torch.nn.functional as functional

    if features.ndim != 5 or coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError('feature sampling needs a volume and time/y/x point coordinates')
    if features.shape[1] == 1:
        image = features[:, 0].permute(0, 3, 1, 2)
        scale = torch.tensor(features.shape[2:4], device=coordinates.device)
        grid = (2. * coordinates[..., 1:].float() / scale - 1.).flip(-1)[:, :, None]
        return functional.grid_sample(image, grid, mode='bilinear', padding_mode='border',
                                      align_corners=False).squeeze(-1).permute(0, 2, 1)
    image = features.permute(0, 4, 1, 2, 3)
    grid = coordinates.float().clone()
    grid[..., 0] += .5
    scale = torch.tensor(image.shape[2:], device=grid.device)
    grid = (2. * grid / scale - 1.).flip(-1)[:, :, None, None]
    return functional.grid_sample(image, grid, mode='bilinear', padding_mode='border',
                                  align_corners=False).squeeze((3, 4)).permute(0, 2, 1)


def use_single_frame_sampling() -> None:
    """Replace only the external point sampler in this explicit tracking worker."""
    from tapnet.torch import utils

    utils.map_coordinates_3d = map_coordinates_3d

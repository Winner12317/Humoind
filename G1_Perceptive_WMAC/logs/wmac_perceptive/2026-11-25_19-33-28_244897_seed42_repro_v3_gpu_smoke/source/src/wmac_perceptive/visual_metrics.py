"""Actor-view image preprocessing, training targets and depth diagnostics.

Depth is normalized by ``spec.max_depth`` and the adjacent validity channel
marks a physical ray hit. Diagnostics use *target* validity, so predicting
no hits cannot hide depth errors. An empty region has RMSE zero by convention
and reports zero valid pixels; callers must retain the counts when deciding
whether a measurement can be used for validation.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .perception import PerceptionSpec


def preprocess_depth_channels(values: torch.Tensor, spec: PerceptionSpec,
                              *, channels: int, include_valid: bool = True) -> torch.Tensor:
    """Crop and blur depth images, optionally paired with diagnostic validity.

    ``channels`` is an architecture constant. Passing it explicitly also keeps
    the grouped convolution kernel shape static during dynamic-batch ONNX
    export. Validity channels are cropped without blurring, as in the Actor.
    """
    if not isinstance(include_valid, bool):
        raise ValueError("include_valid must be boolean")
    if channels < 1 or (include_valid and (channels < 2 or channels % 2)):
        raise ValueError("Expected interleaved depth/valid channel pairs")
    if values.ndim < 3 or tuple(values.shape[-3:]) != (channels, spec.height, spec.width):
        raise ValueError("Depth image dimensions differ from the perception spec")
    height = spec.height-spec.crop_top-spec.crop_bottom
    width = spec.width-spec.crop_left-spec.crop_right
    values = values[..., spec.crop_top:spec.height-spec.crop_bottom,
                    spec.crop_left:spec.width-spec.crop_right]
    if spec.blur_kernel == 1:
        return values
    prefix = values.shape[:-3]
    flat = values.reshape(-1, channels, height, width)
    depth = flat[:, 0::2] if include_valid else flat
    axis = torch.arange(-1, 2, device=values.device, dtype=values.dtype)
    gaussian = torch.exp(-axis.square()/(2*spec.blur_sigma**2))
    gaussian = gaussian/gaussian.sum()
    depth_channels = channels//2 if include_valid else channels
    kernel = (gaussian[:, None]*gaussian[None, :])[None, None].expand(depth_channels, 1, 3, 3)
    depth = F.conv2d(F.pad(depth, (1, 1, 1, 1), mode="reflect"), kernel, groups=depth_channels)
    output = torch.stack((depth, flat[:, 1::2]), dim=2).flatten(1, 2) if include_valid else depth
    return output.reshape(*prefix, channels, height, width)


def policy_view(frame: torch.Tensor, spec: PerceptionSpec) -> torch.Tensor:
    """The cropped, blurred two-channel frame consumed by the policy encoder."""
    return preprocess_depth_channels(frame, spec, channels=2)


def _validate_pair(predicted: torch.Tensor, target: torch.Tensor,
                   spec: PerceptionSpec) -> None:
    if predicted.shape != target.shape or predicted.ndim < 3 or tuple(predicted.shape[-3:]) != spec.frame_shape:
        raise ValueError("Prediction and target must match (...,2,H,W) perception frames")


def _depth_rmse(predicted: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor, max_depth: float) -> tuple[torch.Tensor, torch.Tensor]:
    count = valid.sum((-2, -1))
    squared = ((predicted-target).square()*valid).sum((-2, -1))
    return (squared/count.clamp_min(1)).sqrt()*max_depth, count


def depth_metrics(predicted: torch.Tensor, target: torch.Tensor,
                  spec: PerceptionSpec) -> dict[str, torch.Tensor]:
    """Per-sample meter RMSE and validity error for full, Actor and outer views.

    Full and outside metrics compare raw images. The Actor metric compares the
    same crop and Gaussian-blurred depths that enter its encoder. A target hit
    count of zero is explicit even though its depth RMSE convention is zero.
    No finite-value sanitization is performed: NaNs remain visible to callers.
    """
    _validate_pair(predicted, target, spec)
    # Replay frames can be FP16/FP32 while numerical fixtures and a restored
    # model may use FP64. The same preprocessing precision describes the view
    # being compared and also satisfies BCE's exact dtype requirement.
    target = target.to(predicted)
    target_valid = target[..., 1, :, :].clamp(0, 1)
    full_rmse, full_count = _depth_rmse(predicted[..., 0, :, :], target[..., 0, :, :],
                                        target_valid, spec.max_depth)
    predicted_view, target_view = policy_view(predicted, spec), policy_view(target, spec)
    actor_valid = target_view[..., 1, :, :].clamp(0, 1)
    actor_rmse, actor_count = _depth_rmse(predicted_view[..., 0, :, :], target_view[..., 0, :, :],
                                          actor_valid, spec.max_depth)
    outside = torch.ones_like(target_valid)
    outside[..., spec.crop_top:spec.height-spec.crop_bottom,
            spec.crop_left:spec.width-spec.crop_right] = 0
    outside_valid = target_valid*outside
    outside_rmse, outside_count = _depth_rmse(predicted[..., 0, :, :], target[..., 0, :, :],
                                              outside_valid, spec.max_depth)
    mask_squared = (predicted[..., 1, :, :]-target_valid).square()
    outside_pixel_count = outside.sum((-2, -1))
    return {
        "full_rmse": full_rmse,
        "actor_rmse": actor_rmse,
        "outside_rmse": outside_rmse,
        "full_mask_error": mask_squared.mean((-2, -1)),
        "actor_mask_error": (predicted_view[..., 1, :, :]-actor_valid).square().mean((-2, -1)),
        "outside_mask_error": (mask_squared*outside).sum((-2, -1))/outside_pixel_count.clamp_min(1),
        "full_valid_pixels": full_count,
        "actor_valid_pixels": actor_count,
        "outside_valid_pixels": outside_count,
        "outside_pixel_count": outside_pixel_count,
    }


def policy_view_loss_terms(predicted: torch.Tensor, target: torch.Tensor,
                           spec: PerceptionSpec) -> dict[str, torch.Tensor]:
    """Per-sample normalized SmoothL1 depth and BCE validity in the Actor view.

    The two coefficients can be controlled independently by the caller. Target
    validity alone weights depth loss; invalid predicted masks cannot suppress
    regression. Crop and blur retain gradients into predicted image pixels.
    """
    _validate_pair(predicted, target, spec)
    target = target.to(predicted)
    predicted_view, target_view = policy_view(predicted, spec), policy_view(target, spec)
    depth, valid = predicted_view[..., 0, :, :], predicted_view[..., 1, :, :]
    target_depth = target_view[..., 0, :, :]
    target_valid = target_view[..., 1, :, :].clamp(0, 1)
    depth_loss = (F.smooth_l1_loss(depth, target_depth, reduction="none")*target_valid).sum((-2, -1))
    depth_loss = depth_loss/target_valid.sum((-2, -1)).clamp_min(1)
    valid_loss = F.binary_cross_entropy(valid.clamp(1e-6, 1-1e-6), target_valid,
                                         reduction="none").mean((-2, -1))
    return {"depth": depth_loss, "valid": valid_loss}

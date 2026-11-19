"""Transform hidden affine outputs and compensate at their next affine consumer."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import torch
from torch import nn


@dataclass(frozen=True)
class Permute:
    """Put original unit order[i] in transformed position i."""

    layer: int
    order: Sequence[int]


@dataclass(frozen=True)
class ReluScale:
    """Multiply ReLU hidden outputs by finite, strictly positive scales."""

    layer: int
    scales: Sequence[float]


@dataclass(frozen=True)
class TanhSignFlip:
    """Multiply tanh hidden outputs by signs in {-1, +1}."""

    layer: int
    signs: Sequence[int]


Transform = Permute | ReluScale | TanhSignFlip
Affine = nn.Linear | nn.Conv2d
_AFFINE = (nn.Linear, nn.Conv2d)
_POINTWISE = (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.ELU, nn.GELU,
              nn.SELU, nn.SiLU, nn.Softplus, nn.ReLU6, nn.Hardtanh)
_AVG_POOL = (nn.AvgPool2d, nn.AdaptiveAvgPool2d)
_MAX_POOL = (nn.MaxPool2d, nn.AdaptiveMaxPool2d)


def _validate_model(model: nn.Sequential) -> None:
    if type(model) is not nn.Sequential:
        raise TypeError("expected a flat torch.nn.Sequential")
    modules = list(model._modules.values())
    if any(module is None or isinstance(module, nn.Sequential) for module in modules):
        raise ValueError("expected a flat Sequential with concrete modules")
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError("shared modules are unsupported")
    # Sharing even distinct Parameter objects' storage can modify unrelated layers.
    parameters = list(model.named_parameters(remove_duplicate=False))
    storages = {
        (parameter.device, parameter.untyped_storage().data_ptr())
        for _, parameter in parameters
        if parameter.numel()
    }
    if len(storages) != sum(parameter.numel() > 0 for _, parameter in parameters):
        raise ValueError("shared parameter storage is unsupported")
    if any(module._forward_hooks or module._forward_pre_hooks for module in (model, *modules)):
        raise ValueError("forward hooks are unsupported")


def _affine_at(model: nn.Sequential, layer: int) -> Affine:
    if isinstance(layer, bool) or not isinstance(layer, Integral):
        raise TypeError("layer index must be an integer")
    if layer < 0 or layer >= len(model):
        raise ValueError("layer index is outside the model")
    affine = model[layer]
    if type(affine) not in _AFFINE:
        raise ValueError("layer must be a Linear or Conv2d")
    if isinstance(affine, nn.Conv2d) and affine.groups != 1:
        raise ValueError("grouped convolution is unsupported")
    if not affine.weight.is_floating_point() or not affine.weight.numel():
        raise ValueError("affine weight must be a nonempty floating-point tensor")
    return affine


def _path(model: nn.Sequential, layer: int) -> tuple[Affine, list[nn.Module]]:
    between = []
    for index in range(layer + 1, len(model)):
        module = model[index]
        if isinstance(module, _AFFINE):
            return _affine_at(model, index), between
        between.append(module)
    raise ValueError("hidden layer needs a succeeding Linear or Conv2d")


def _input_block(
    incoming: Affine, outgoing: Affine, between: list[nn.Module], transform: Transform
) -> int:
    channels = incoming.weight.shape[0]
    is_conv = isinstance(incoming, nn.Conv2d)
    flattened = False
    required_activation = {ReluScale: nn.ReLU, TanhSignFlip: nn.Tanh}.get(type(transform))
    saw_activation = False
    for module in between:
        kind = type(module)
        if kind is required_activation:
            saw_activation = True
        elif kind in _POINTWISE and isinstance(transform, Permute):
            continue
        elif kind is nn.Identity or (kind in (nn.Dropout, nn.Dropout2d) and not module.training):
            continue
        elif kind is nn.Flatten and (module.start_dim, module.end_dim) == (1, -1):
            if not is_conv or flattened:
                raise ValueError("flatten requires an unflattened Conv2d output")
            flattened = True
        elif kind in _AVG_POOL or (
            kind in _MAX_POOL and not module.return_indices and not isinstance(transform, TanhSignFlip)
        ):
            if not is_conv or flattened:
                raise ValueError("pooling requires an unflattened Conv2d output")
        else:
            raise ValueError(f"{type(transform).__name__} does not support {kind.__name__} on its path")

    if required_activation is not None and not saw_activation:
        raise ValueError(f"{type(transform).__name__} requires {required_activation.__name__} on its path")
    if isinstance(outgoing, nn.Conv2d):
        if not is_conv or flattened or outgoing.in_channels != channels:
            raise ValueError("expected unflattened Conv2d output with matching channels")
        return 1
    if is_conv != flattened:
        raise ValueError("Conv2d to Linear requires explicit NCHW flatten from dimension 1")
    block, extra = divmod(outgoing.in_features, channels)
    if extra or block == 0 or (not is_conv and block != 1):
        raise ValueError("successor input width does not match hidden layer")
    return block


def _integers(values: Sequence[int], width: int, field: str) -> tuple[int, ...]:
    if len(values) != width or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise ValueError(f"{field} must contain {width} integers")
    return tuple(int(value) for value in values)


def _positive_scales(scales: Sequence[float], width: int) -> tuple[float, ...]:
    message = f"scales must contain {width} finite positive numbers"
    if len(scales) != width or any(
        isinstance(scale, bool) or not isinstance(scale, Real) for scale in scales
    ):
        raise ValueError(message)
    try:
        values = tuple(float(scale) for scale in scales)
    except OverflowError as error:
        raise ValueError(message) from error
    if any(not isfinite(value) or value <= 0 for value in values):
        raise ValueError(message)
    return values


def _checked_scale(parameter: torch.Tensor, factors: torch.Tensor, axis: int) -> None:
    shape = [1] * parameter.ndim
    shape[axis] = factors.numel()
    result = parameter * factors.reshape(shape)
    if not result.isfinite().all() or ((parameter != 0) & (result == 0)).any():
        raise ValueError("scaling produced nonfinite weights or underflowed to zero")
    parameter.copy_(result)


def _apply_one(model: nn.Sequential, transform: Transform) -> None:
    if type(transform) not in (Permute, ReluScale, TanhSignFlip):
        raise TypeError("unsupported transform")
    incoming = _affine_at(model, transform.layer)
    outgoing, between = _path(model, transform.layer)
    block = _input_block(incoming, outgoing, between, transform)
    width = incoming.weight.shape[0]

    if isinstance(transform, Permute):
        order = _integers(transform.order, width, "order")
        if set(order) != set(range(width)):
            raise ValueError("order must be a permutation of the hidden unit indices")
        indices = incoming.weight.new_tensor(order, dtype=torch.long)
        incoming.weight.copy_(incoming.weight.index_select(0, indices))
        if incoming.bias is not None:
            incoming.bias.copy_(incoming.bias.index_select(0, indices))
        channels = indices.to(outgoing.weight.device)
        offsets = torch.arange(block, device=outgoing.weight.device)
        columns = (channels[:, None] * block + offsets).flatten()
        outgoing.weight.copy_(outgoing.weight.index_select(1, columns))
        return

    if isinstance(transform, TanhSignFlip):
        values = _integers(transform.signs, width, "signs")
        if any(value not in (-1, 1) for value in values):
            raise ValueError("each sign must be -1 or 1")
    else:
        values = _positive_scales(transform.scales, width)
    factors = incoming.weight.new_tensor(values)
    inverse = outgoing.weight.new_tensor([1 / value for value in values])
    if any(not factor.isfinite().all() or (factor == 0).any() for factor in (factors, inverse)):
        raise ValueError("factors and reciprocals must be finite and nonzero in the layer dtypes")
    _checked_scale(incoming.weight, factors, axis=0)
    if incoming.bias is not None:
        _checked_scale(incoming.bias, factors, axis=0)
    _checked_scale(outgoing.weight, inverse.repeat_interleave(block), axis=1)


@torch.no_grad()
def apply_transforms(model: nn.Sequential, *transforms: Transform) -> nn.Sequential:
    """Return an independent model with the transforms applied in argument order."""
    _validate_model(model)
    result = deepcopy(model)
    for transform in transforms:
        _apply_one(result, transform)
    return result


@torch.no_grad()
def shuffle_weights(model: nn.Sequential, layer: int, *, seed: int) -> nn.Sequential:
    """Permute all scalar weights in one layer, leaving biases and other layers fixed.

    This is a function-changing control for suitable nondegenerate models. Check
    the output difference on the target model rather than assuming it changes.
    """
    _validate_model(model)
    result = deepcopy(model)
    weight = _affine_at(result, layer).weight
    flat = weight.flatten()
    if not flat.isfinite().all():
        raise ValueError("shuffle control requires finite weights")
    generator = torch.Generator().manual_seed(seed)
    shuffled = flat.index_select(0, torch.randperm(flat.numel(), generator=generator).to(weight.device))
    if torch.equal(shuffled, flat):
        differing = (flat != flat[0]).nonzero()
        if not differing.numel():
            raise ValueError("shuffle control requires at least two distinct weight values")
        index = differing[0, 0].item()
        shuffled[0], shuffled[index] = flat[index], flat[0]
    weight.copy_(shuffled.reshape_as(weight))
    return result

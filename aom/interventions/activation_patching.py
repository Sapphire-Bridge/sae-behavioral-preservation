from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel


@dataclass(frozen=True)
class PatchSite:
    layer: int
    token_index: int


@dataclass(frozen=True)
class PatchSpanSite:
    layer: int
    token_indices: Tuple[int, ...]


def _model_type_of(obj: object) -> str:
    if obj is None:
        return ""
    model_type = getattr(obj, "model_type", None)
    if isinstance(model_type, str):
        return model_type.lower()
    cfg = getattr(obj, "config", None)
    model_type = getattr(cfg, "model_type", None)
    if isinstance(model_type, str):
        return model_type.lower()
    return ""


def _class_name_of(obj: object) -> str:
    return type(obj).__name__.lower()


def _is_gemma3_name_fallback(obj: object) -> bool:
    name = _class_name_of(obj)
    return "gemma3" in name and "gemma3n" not in name


def _is_gemma3_model(model: PreTrainedModel) -> bool:
    model_type = _model_type_of(model)
    if model_type in {"gemma3", "gemma3_text"}:
        return True
    if model_type:
        return False
    return _is_gemma3_name_fallback(model)


def _unwrap_gemma3_text_backbone(model: PreTrainedModel):
    if not _is_gemma3_model(model):
        raise TypeError(f"Unsupported Gemma3 object: {type(model).__name__}")

    candidates = []
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        candidates.append(language_model)
    base_model = getattr(model, "model", None)
    if base_model is not None:
        candidates.append(base_model)
    candidates.append(model)

    for candidate in candidates:
        if not (hasattr(candidate, "layers") and hasattr(candidate, "norm")):
            continue
        candidate_type = _model_type_of(candidate)
        if candidate_type == "gemma3_text" or (not candidate_type and _is_gemma3_name_fallback(candidate)):
            return candidate
    raise TypeError(f"Could not unwrap Gemma3 text backbone from {type(model).__name__}")


def _unwrap_gemma3_text_config(model: PreTrainedModel):
    backbone = _unwrap_gemma3_text_backbone(model)
    cfg = getattr(backbone, "config", None)
    if cfg is not None:
        return cfg
    model_cfg = getattr(model, "config", None)
    text_cfg = getattr(model_cfg, "text_config", None)
    if text_cfg is not None:
        return text_cfg
    if model_cfg is not None:
        return model_cfg
    raise TypeError(f"Could not resolve Gemma3 text config from {type(model).__name__}")


_ARCHITECTURE_REGISTRY: dict[
    str,
    tuple[
        Callable[[PreTrainedModel], bool],
        Callable[[PreTrainedModel], nn.ModuleList],
        Callable[[PreTrainedModel], int],
    ],
] = {
    "qwen3": (
        lambda m: hasattr(m, "model")
        and hasattr(m.model, "layers")
        and (
            str(getattr(getattr(m, "config", None), "model_type", "")).lower() == "qwen3"
            or "qwen3" in type(m).__name__.lower()
        ),
        lambda m: m.model.layers,  # type: ignore[attr-defined]
        lambda m: int(m.config.num_hidden_layers),
    ),
    "gpt2": (
        lambda m: hasattr(m, "transformer") and hasattr(m.transformer, "h"),
        lambda m: m.transformer.h,  # type: ignore[attr-defined]
        lambda m: int(m.config.n_layer),
    ),
    "qwen2": (
        lambda m: hasattr(m, "model") and hasattr(m.model, "layers") and "qwen" in type(m).__name__.lower(),
        lambda m: m.model.layers,  # type: ignore[attr-defined]
        lambda m: int(m.config.num_hidden_layers),
    ),
    "gemma3": (
        lambda m: _is_gemma3_model(m),
        lambda m: _unwrap_gemma3_text_backbone(m).layers,  # type: ignore[attr-defined]
        lambda m: int(_unwrap_gemma3_text_config(m).num_hidden_layers),
    ),
    "gemma": (
        lambda m: hasattr(m, "model")
        and hasattr(m.model, "layers")
        and (
            _model_type_of(m) in {"gemma", "gemma2"}
            or ("gemma" in _class_name_of(m) and not any(tag in _class_name_of(m) for tag in ("gemma3", "gemma3n")))
        ),
        lambda m: m.model.layers,  # type: ignore[attr-defined]
        lambda m: int(m.config.num_hidden_layers),
    ),
    "llama": (
        lambda m: hasattr(m, "model") and hasattr(m.model, "layers") and "llama" in type(m).__name__.lower(),
        lambda m: m.model.layers,  # type: ignore[attr-defined]
        lambda m: int(m.config.num_hidden_layers),
    ),
    "mistral": (
        lambda m: hasattr(m, "model") and hasattr(m.model, "layers") and "mistral" in type(m).__name__.lower(),
        lambda m: m.model.layers,  # type: ignore[attr-defined]
        lambda m: int(m.config.num_hidden_layers),
    ),
    "gpt_neox": (
        lambda m: hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"),
        lambda m: m.gpt_neox.layers,  # type: ignore[attr-defined]
        lambda m: int(m.config.num_hidden_layers),
    ),
}


def detect_architecture(model: PreTrainedModel) -> str:
    """Detect model architecture family."""
    for arch_name, (check_fn, _blocks, _n_layers) in _ARCHITECTURE_REGISTRY.items():
        if check_fn(model):
            return arch_name
    raise ValueError(
        f"Unsupported architecture: {type(model).__name__}. Supported: {list(_ARCHITECTURE_REGISTRY.keys())}"
    )


def get_decoder_blocks(model: PreTrainedModel) -> nn.ModuleList:
    """Return the transformer block ModuleList for supported architectures."""
    arch = detect_architecture(model)
    return _ARCHITECTURE_REGISTRY[arch][1](model)


def get_num_layers(model: PreTrainedModel) -> int:
    """Return number of transformer layers for supported architectures."""
    arch = detect_architecture(model)
    return _ARCHITECTURE_REGISTRY[arch][2](model)


@torch.no_grad()
def get_hidden_states(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    _ = detect_architecture(model)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    if hs is None:
        raise RuntimeError("Model did not return hidden_states; ensure output_hidden_states=True is supported.")
    return hs


@torch.no_grad()
def get_block_outputs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    layers: Sequence[int],
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """
    Capture the exact decoder-block forward outputs (resid_post semantics) for specific layers.

    This is intentionally hook-based (not output_hidden_states-based) so the captured tensors
    match the ones targeted by AoM patching hooks.
    """
    blocks = get_decoder_blocks(model)
    n_layers = len(blocks)
    layer_list = [int(l) for l in layers]
    if any(l < 0 or l >= n_layers for l in layer_list):
        raise ValueError(f"layers out of range [0, {n_layers}): {layer_list}")

    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor) or h.ndim != 3:
                raise ValueError("expected block output tensor of shape (batch, seq, hidden)")
            captured[int(layer_idx)] = h.detach()

        return _hook

    for l in layer_list:
        handles.append(blocks[int(l)].register_forward_hook(_make_hook(int(l))))
    try:
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    missing = [l for l in layer_list if int(l) not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture block outputs for layers: {missing}")
    return captured


@torch.no_grad()
def forward_with_patched_block_output(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    site: PatchSite,
    replacement: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Patch the hidden state at a given block output (resid_post) for a single token position.

    This currently supports GPT-2 style models (GPT2LMHeadModel) where blocks live at
    `model.transformer.h` and each block output is a tuple whose first element is
    `(batch, seq, hidden)`.
    """
    return forward_with_patched_block_output_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=site.layer, token_indices=(site.token_index,)),
        replacement=replacement,
        attention_mask=attention_mask,
    )


@torch.no_grad()
def forward_with_patched_block_output_span(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    site: PatchSpanSite,
    replacement: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Patch the hidden state at a given block output (resid_post) for multiple token positions.

    `replacement` must be either:
      - (hidden_dim,) when patching a single token index, or
      - (span_len, hidden_dim) when patching multiple indices.
    """
    blocks, replacement2d = _resolve_patch_site_and_replacement(model=model, site=site, replacement=replacement)
    handle = blocks[site.layer].register_forward_hook(_make_patched_block_output_hook(site=site, replacement2d=replacement2d))
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits  # type: ignore[return-value]
    finally:
        handle.remove()


def _resolve_patch_site_and_replacement(
    *,
    model: PreTrainedModel,
    site: PatchSpanSite,
    replacement: torch.Tensor,
) -> tuple[nn.ModuleList, torch.Tensor]:
    blocks = get_decoder_blocks(model)
    n_layers = len(blocks)
    if site.layer < 0 or site.layer >= n_layers:
        raise ValueError(f"layer {site.layer} out of range [0, {n_layers})")
    if not site.token_indices:
        raise ValueError("token_indices must be non-empty")
    if replacement.ndim == 1:
        if len(site.token_indices) != 1:
            raise ValueError("1D replacement requires exactly one token index")
        replacement2d = replacement.unsqueeze(0)
    elif replacement.ndim == 2:
        replacement2d = replacement
    else:
        raise ValueError("replacement must be 1D or 2D")
    if replacement2d.size(0) != len(site.token_indices):
        raise ValueError("replacement first dim must match number of token indices")
    return blocks, replacement2d


def _make_patched_block_output_hook(*, site: PatchSpanSite, replacement2d: torch.Tensor):
    def hook(module, inputs, output):  # noqa: ANN001
        _ = module
        _ = inputs
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        hidden = hidden.clone()
        for i, tok_idx in enumerate(site.token_indices):
            hidden[:, tok_idx, :] = replacement2d[i]

        if rest is None:
            return hidden
        return (hidden,) + rest

    return hook


@torch.no_grad()
def prefill_with_patched_block_output_span(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    site: PatchSpanSite,
    replacement: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Run a patched prompt prefill and return full model output (including cache).

    This is used to reuse a patched prefix cache across many continuation scorings.
    """
    blocks, replacement2d = _resolve_patch_site_and_replacement(model=model, site=site, replacement=replacement)
    handle = blocks[site.layer].register_forward_hook(_make_patched_block_output_hook(site=site, replacement2d=replacement2d))
    try:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    finally:
        handle.remove()

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from torch import nn
from transformers import PreTrainedModel

from aom.interventions.activation_patching import PatchSite, PatchSpanSite, get_decoder_blocks
from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig, CLTProtocol, reconstruct_with_error_preservation


class LatentPolicy(Protocol):
    """
    Policy that edits CLT latents, restricted to a site mask.

    latents: (batch, seq, d_latent)
    site_mask: (batch, seq, 1) boolean mask marking intervention region
    token_mask: optional (batch, seq) boolean valid-token mask
    """

    def apply(
        self,
        latents: torch.Tensor,
        *,
        site_mask: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class IdentityLatentPolicy:
    def apply(
        self,
        latents: torch.Tensor,
        *,
        site_mask: torch.Tensor,  # noqa: ARG002
        token_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return latents


@dataclass(frozen=True)
class ReplaceLatentsAtIndicesPolicy:
    token_indices: list[int]
    replacement_latents: torch.Tensor  # (B, span_len, d_latent) or (1, span_len, d_latent) for broadcast

    def apply(
        self,
        latents: torch.Tensor,
        *,
        site_mask: torch.Tensor,  # noqa: ARG002
        token_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("latents must have shape (batch, seq, d_latent)")
        if self.replacement_latents.ndim != 3:
            raise ValueError("replacement_latents must have shape (batch, span_len, d_latent)")
        # Allow broadcasting a single replacement across the batch (common when batching multiple
        # continuations for the same prompt).
        if int(self.replacement_latents.size(0)) not in {1, int(latents.size(0))}:
            raise ValueError("replacement batch size must be 1 or match latents batch size")
        if latents.size(-1) != self.replacement_latents.size(-1):
            raise ValueError("replacement latent dim must match latents latent dim")
        if int(self.replacement_latents.size(1)) != len(self.token_indices):
            raise ValueError("replacement span_len must match token_indices length")

        replacement = self.replacement_latents
        if replacement.device != latents.device or replacement.dtype != latents.dtype:
            replacement = replacement.to(device=latents.device, dtype=latents.dtype)
        if int(replacement.size(0)) == 1 and int(latents.size(0)) != 1:
            replacement = replacement.expand(int(latents.size(0)), -1, -1)

        patched = latents.clone()
        for i, tok in enumerate(self.token_indices):
            if int(tok) < 0 or int(tok) >= int(latents.size(1)):
                raise ValueError(f"token index {int(tok)} out of range [0, {int(latents.size(1))})")
            patched[:, int(tok), :] = replacement[:, i, :]
        return patched


@dataclass(frozen=True)
class ReplaceLatentDimsAtIndicesPolicy:
    """
    Replace a subset of latent dimensions at specific token indices.

    This enables "top-k feature" interventions where only selected latent coordinates
    are patched from the donor, while all other coordinates remain from the receiver.
    """

    token_indices: list[int]
    replacement_latents: torch.Tensor  # (B, span_len, d_latent) or (1, span_len, d_latent) for broadcast
    latent_dims: list[int]

    def apply(
        self,
        latents: torch.Tensor,
        *,
        site_mask: torch.Tensor,  # noqa: ARG002
        token_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("latents must have shape (batch, seq, d_latent)")
        if self.replacement_latents.ndim != 3:
            raise ValueError("replacement_latents must have shape (batch, span_len, d_latent)")
        # Allow broadcasting a single replacement across the batch (common when batching multiple
        # continuations for the same prompt).
        if int(self.replacement_latents.size(0)) not in {1, int(latents.size(0))}:
            raise ValueError("replacement batch size must be 1 or match latents batch size")
        if latents.size(-1) != self.replacement_latents.size(-1):
            raise ValueError("replacement latent dim must match latents latent dim")
        if int(self.replacement_latents.size(1)) != len(self.token_indices):
            raise ValueError("replacement span_len must match token_indices length")

        if not self.latent_dims:
            return latents

        d_latent = int(latents.size(-1))
        for dim in self.latent_dims:
            if int(dim) < 0 or int(dim) >= d_latent:
                raise ValueError(f"latent dim {int(dim)} out of range [0, {d_latent})")

        replacement = self.replacement_latents
        if replacement.device != latents.device or replacement.dtype != latents.dtype:
            replacement = replacement.to(device=latents.device, dtype=latents.dtype)
        if int(replacement.size(0)) == 1 and int(latents.size(0)) != 1:
            replacement = replacement.expand(int(latents.size(0)), -1, -1)

        patched = latents.clone()
        for i, tok in enumerate(self.token_indices):
            if int(tok) < 0 or int(tok) >= int(latents.size(1)):
                raise ValueError(f"token index {int(tok)} out of range [0, {int(latents.size(1))})")
            patched[:, int(tok), self.latent_dims] = replacement[:, i, self.latent_dims]
        return patched


@dataclass
class CLTHookState:
    total_active: float = 0.0
    total_preserved: float = 0.0

    def reset(self) -> None:
        self.total_active = 0.0
        self.total_preserved = 0.0

    def sparsity_gain(self) -> float:
        if self.total_active <= 0.0:
            return 0.0
        return 1.0 - (self.total_preserved / self.total_active)


def _infer_clt_device_dtype(clt: CLTProtocol) -> tuple[torch.device, torch.dtype]:
    if isinstance(clt, nn.Module):
        p = next(clt.parameters(), None)
        if p is not None:
            return p.device, p.dtype
    w_dec = getattr(clt, "W_dec", None)
    if isinstance(w_dec, torch.Tensor):
        return w_dec.device, w_dec.dtype
    return torch.device("cpu"), torch.float32


class CLTInterventionHook:
    """
    Forward hook for CLT latent interventions using error-preserving writeback.
    """

    def __init__(
        self,
        *,
        clt: CLTProtocol,
        token_indices: Optional[tuple[int, ...]],
        policy: LatentPolicy,
        transform: Optional[CLTInputTransform] = None,
        config: Optional[CLTPatchConfig] = None,
        token_mask: Optional[torch.Tensor] = None,
        state: Optional[CLTHookState] = None,
    ) -> None:
        self.clt = clt
        self.token_indices = token_indices
        self.policy = policy
        self.transform = transform or CLTInputTransform()
        self.config = config or CLTPatchConfig()
        self.token_mask = token_mask
        self.state = state

    def _site_mask(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        mask = torch.zeros((bsz, seq_len, 1), dtype=torch.bool, device=hidden.device)
        if self.token_indices is None:
            mask[:] = True
            return mask
        for tok in self.token_indices:
            if int(tok) < 0 or int(tok) >= int(seq_len):
                raise ValueError(f"token index {int(tok)} out of range [0, {int(seq_len)})")
            mask[:, int(tok), 0] = True
        return mask

    def __call__(self, _module: nn.Module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("CLTInterventionHook expected hidden tensor with shape (batch, seq, hidden_dim)")
        if int(hidden.size(-1)) != int(self.clt.d_in) or int(hidden.size(-1)) != int(self.clt.d_out):
            raise ValueError(
                f"CLT v1 same-site requires hidden dim to match clt.d_in and clt.d_out; got hidden={int(hidden.size(-1))}, "
                f"d_in={int(self.clt.d_in)}, d_out={int(self.clt.d_out)}"
            )

        model_device = hidden.device
        model_dtype = hidden.dtype
        clt_device, clt_dtype = _infer_clt_device_dtype(self.clt)

        if self.config.dtype_policy == "clt":
            work_device, work_dtype = clt_device, clt_dtype
        elif self.config.dtype_policy == "model":
            work_device, work_dtype = model_device, model_dtype
        else:
            raise ValueError(f"Unknown dtype_policy={self.config.dtype_policy!r}")

        hidden_work = hidden.to(device=work_device, dtype=work_dtype)
        site_mask_model = self._site_mask(hidden)

        token_mask = None
        if self.token_mask is not None:
            if self.token_mask.shape != hidden.shape[:2]:
                raise ValueError("token_mask must have shape (batch, seq)")
            token_mask_model = self.token_mask.to(device=model_device, dtype=torch.bool)
            site_mask_model = site_mask_model & token_mask_model.unsqueeze(-1)
            token_mask = token_mask_model.to(device=work_device)

        site_mask = site_mask_model.to(device=work_device)

        z_receiver = self.clt.encode(self.transform.forward(hidden_work))
        z_patched = self.policy.apply(z_receiver, site_mask=site_mask, token_mask=token_mask)
        if z_patched.shape != z_receiver.shape:
            raise ValueError(
                f"Latent policy must preserve latent shape {tuple(z_receiver.shape)}; got {tuple(z_patched.shape)}"
            )

        if self.config.decode_strategy == "delta_1decode":
            y_hat_t = self.clt.decode(z_receiver)
            y_patched_t = self.clt.decode(z_patched)
            y_delta_model = self.transform.inverse_delta(y_patched_t - y_hat_t)
            y_prime_work = hidden_work + y_delta_model
            y_hat_work = self.transform.inverse(y_hat_t)
        elif self.config.decode_strategy == "safe_2decode":
            y_prime_work, y_hat_work, _err = reconstruct_with_error_preservation(
                clt=self.clt,
                receiver_x=hidden_work,
                receiver_y=hidden_work,
                z_prime=z_patched,
                transform=self.transform,
            )
        else:
            raise ValueError(f"Unknown decode_strategy={self.config.decode_strategy!r}")

        if y_prime_work.shape != hidden_work.shape or y_hat_work.shape != hidden_work.shape:
            raise RuntimeError("CLT decode path produced shape mismatch against hidden states")

        if self.state is not None:
            eps = float(self.config.eps_active)
            active_mask = (z_receiver.abs() > eps) & site_mask.expand_as(z_receiver)
            preserved_mask = (z_patched.abs() > eps) & active_mask
            self.state.total_active += float(active_mask.sum().item())
            self.state.total_preserved += float(preserved_mask.sum().item())

        y_prime_model = y_prime_work.to(device=model_device, dtype=model_dtype)
        patched_hidden = torch.where(site_mask_model.expand_as(hidden), y_prime_model, hidden)

        if rest is None:
            return patched_hidden
        return (patched_hidden,) + rest


@torch.no_grad()
def forward_with_clt_latent_patching(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    site: PatchSite,
    clt: CLTProtocol,
    policy: LatentPolicy,
    transform: Optional[CLTInputTransform] = None,
    config: Optional[CLTPatchConfig] = None,
    state: Optional[CLTHookState] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return forward_with_clt_latent_patching_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=int(site.layer), token_indices=(int(site.token_index),)),
        clt=clt,
        policy=policy,
        transform=transform,
        config=config,
        state=state,
        attention_mask=attention_mask,
    )


@torch.no_grad()
def forward_with_clt_latent_patching_span(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    site: PatchSpanSite,
    clt: CLTProtocol,
    policy: LatentPolicy,
    transform: Optional[CLTInputTransform] = None,
    config: Optional[CLTPatchConfig] = None,
    state: Optional[CLTHookState] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    blocks = get_decoder_blocks(model)
    n_layers = len(blocks)
    if int(site.layer) < 0 or int(site.layer) >= n_layers:
        raise ValueError(f"layer {int(site.layer)} out of range [0, {n_layers})")
    if not site.token_indices:
        raise ValueError("token_indices must be non-empty")

    seq_len = int(input_ids.size(1))
    for token_idx in site.token_indices:
        if int(token_idx) < 0 or int(token_idx) >= seq_len:
            raise ValueError(f"token index {int(token_idx)} out of range [0, {seq_len})")

    site_token_indices = [int(x) for x in site.token_indices]
    policy_token_indices = getattr(policy, "token_indices", None)
    if policy_token_indices is not None:
        policy_token_indices_list = [int(x) for x in policy_token_indices]
        if policy_token_indices_list != site_token_indices:
            raise ValueError(
                "CLT policy token indices must match patch site token indices; "
                f"policy={policy_token_indices_list}, site={site_token_indices}"
            )

    token_mask = None
    attention_mask_model = None
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        attention_mask_model = attention_mask.to(device=input_ids.device)
        token_mask = attention_mask_model.to(dtype=torch.bool)

    hook = CLTInterventionHook(
        clt=clt,
        token_indices=tuple(site_token_indices),
        policy=policy,
        transform=transform,
        config=config,
        token_mask=token_mask,
        state=state,
    )
    handle = blocks[int(site.layer)].register_forward_hook(hook)
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask_model,
            use_cache=False,
            return_dict=True,
        )
        return out.logits  # type: ignore[return-value]
    finally:
        handle.remove()

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Tuple

import torch


class CLTProtocol(Protocol):
    """Protocol for CLT modules used by patching code."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        ...

    @property
    def d_in(self) -> int:
        ...

    @property
    def d_latent(self) -> int:
        ...

    @property
    def d_out(self) -> int:
        ...


@dataclass(frozen=True)
class CLTInputTransform:
    """
    Input/output scaling transform around CLT encode/decode.

    Keep this intentionally minimal and explicit to avoid hidden preprocessing drift.
    """

    scale: float = 1.0
    shift: float | None = None

    def __post_init__(self) -> None:
        if float(self.scale) == 0.0:
            raise ValueError("scale must be non-zero")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * float(self.scale)
        if self.shift is not None:
            y = y + float(self.shift)
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        x = y
        if self.shift is not None:
            x = x - float(self.shift)
        return x / float(self.scale)

    def inverse_delta(self, delta: torch.Tensor) -> torch.Tensor:
        # Additive shift cancels in deltas.
        return delta / float(self.scale)


@dataclass(frozen=True)
class CLTSiteSpec:
    """
    Explicit site contract for CLT interventions.

    v1 enforces same-site patching:
      encode_site == decode_site == writeback_site
    """

    encode_site: str = "resid_post"
    decode_site: str = "resid_post"
    writeback_site: str = "resid_post"
    site_mode: Literal["same_site_v1"] = "same_site_v1"

    def __post_init__(self) -> None:
        if self.site_mode != "same_site_v1":
            raise ValueError(f"Unsupported site_mode={self.site_mode!r}; only 'same_site_v1' is allowed in v1")
        if self.encode_site != self.decode_site or self.decode_site != self.writeback_site:
            raise ValueError(
                "CLT v1 requires encode_site == decode_site == writeback_site "
                f"(got encode={self.encode_site!r}, decode={self.decode_site!r}, writeback={self.writeback_site!r})"
            )


@dataclass(frozen=True)
class CLTPatchConfig:
    """Execution-time CLT patch config."""

    decode_strategy: Literal["safe_2decode", "delta_1decode"] = "delta_1decode"
    dtype_policy: Literal["clt", "model"] = "clt"
    eps_active: float = 1e-6

    def __post_init__(self) -> None:
        if float(self.eps_active) < 0.0:
            raise ValueError("eps_active must be >= 0")


def reconstruct_with_error_preservation(
    *,
    clt: CLTProtocol,
    receiver_x: torch.Tensor,
    receiver_y: torch.Tensor,
    z_prime: torch.Tensor,
    transform: CLTInputTransform,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Error-preserving writeback for CLT patching.

    Let:
      z = E(transform.forward(receiver_x))
      y_hat = transform.inverse(D(z))
      e = receiver_y - y_hat
      y_prime = transform.inverse(D(z_prime)) + e

    Returns:
      (y_prime, y_hat, e)
    """

    if receiver_x.ndim != 3 or receiver_y.ndim != 3 or z_prime.ndim != 3:
        raise ValueError("receiver_x, receiver_y, z_prime must all have shape (batch, seq, dim)")
    if receiver_x.shape[:2] != receiver_y.shape[:2]:
        raise ValueError("receiver_x and receiver_y must match on (batch, seq)")
    if z_prime.shape[:2] != receiver_x.shape[:2]:
        raise ValueError("z_prime must match receiver tensors on (batch, seq)")
    if int(receiver_x.size(-1)) != int(clt.d_in):
        raise ValueError(f"receiver_x hidden dim must match clt.d_in={int(clt.d_in)}")
    if int(receiver_y.size(-1)) != int(clt.d_out):
        raise ValueError(f"receiver_y hidden dim must match clt.d_out={int(clt.d_out)}")
    if int(z_prime.size(-1)) != int(clt.d_latent):
        raise ValueError(f"z_prime hidden dim must match clt.d_latent={int(clt.d_latent)}")

    receiver_x_t = transform.forward(receiver_x)
    z_receiver = clt.encode(receiver_x_t)
    y_hat_t = clt.decode(z_receiver)
    y_hat = transform.inverse(y_hat_t)

    if y_hat.shape != receiver_y.shape:
        raise ValueError(
            f"Decoded reconstruction shape {tuple(y_hat.shape)} must equal receiver_y shape {tuple(receiver_y.shape)}"
        )

    e = receiver_y - y_hat
    y_hat_prime_t = clt.decode(z_prime)
    y_hat_prime = transform.inverse(y_hat_prime_t)
    if y_hat_prime.shape != receiver_y.shape:
        raise ValueError(
            f"Patched decoded shape {tuple(y_hat_prime.shape)} must equal receiver_y shape {tuple(receiver_y.shape)}"
        )
    y_prime = y_hat_prime + e
    return y_prime, y_hat, e


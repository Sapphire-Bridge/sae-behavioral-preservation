from __future__ import annotations

import pytest
import torch
from torch import nn

from aom.interventions.clt_adapter import (
    CLTInputTransform,
    CLTSiteSpec,
    reconstruct_with_error_preservation,
)


class _IdentityCLT(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self._d_in = int(d)
        self._d_latent = int(d)
        self._d_out = int(d)

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_latent(self) -> int:
        return self._d_latent

    @property
    def d_out(self) -> int:
        return self._d_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents


def test_clt_site_spec_requires_same_site_in_v1():
    _ = CLTSiteSpec(encode_site="resid_post", decode_site="resid_post", writeback_site="resid_post")
    with pytest.raises(ValueError, match="requires encode_site == decode_site == writeback_site"):
        _ = CLTSiteSpec(encode_site="resid_pre", decode_site="mlp_out", writeback_site="mlp_out")


def test_reconstruct_with_error_preservation_identity_patch_is_exact():
    clt = _IdentityCLT(d=8)
    transform = CLTInputTransform(scale=1.0)

    receiver_x = torch.randn(2, 3, 8)
    receiver_y = torch.randn(2, 3, 8)
    z_receiver = clt.encode(transform.forward(receiver_x))

    y_prime, y_hat, err = reconstruct_with_error_preservation(
        clt=clt,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        z_prime=z_receiver,
        transform=transform,
    )

    # Identity CLT means y_hat == receiver_x.
    assert torch.allclose(y_hat, receiver_x, atol=1e-6)
    # Error-preserving writeback with z'==z_receiver should reproduce receiver_y exactly.
    assert torch.allclose(y_prime, receiver_y, atol=1e-6)
    assert torch.allclose(err, receiver_y - receiver_x, atol=1e-6)


def test_reconstruct_with_error_preservation_shape_validation():
    clt = _IdentityCLT(d=4)
    transform = CLTInputTransform(scale=1.0)
    receiver_x = torch.randn(1, 2, 4)
    receiver_y = torch.randn(1, 2, 4)
    bad_z = torch.randn(1, 2, 5)

    with pytest.raises(ValueError, match="z_prime hidden dim must match"):
        _ = reconstruct_with_error_preservation(
            clt=clt,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            z_prime=bad_z,
            transform=transform,
        )


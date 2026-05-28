from __future__ import annotations

import json

import numpy as np
import torch

from aom.interventions.clt_loader import LinearCLT, _infer_cfg_from_weights, load_clt


def test_relu_default_does_not_subtract_pre_encoder_bias():
    d_in = 4
    d_latent = 4
    W_enc = torch.eye(d_in, d_latent)
    W_dec = torch.eye(d_latent, d_in)
    b_enc = torch.zeros(d_latent)
    b_dec = torch.full((d_in,), 5.0)

    clt = LinearCLT(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        cfg={"d_in": d_in, "d_latent": d_latent, "d_out": d_in, "activation": "relu"},
    )
    x = torch.full((1, 1, d_in), 2.0)
    z = clt.encode(x)
    assert torch.allclose(z, torch.full_like(z, 2.0), atol=1e-6)


def test_jumprelu_pre_encoder_bias_opt_in():
    d_in = 4
    d_latent = 4
    W_enc = torch.eye(d_in, d_latent)
    W_dec = torch.eye(d_latent, d_in)
    b_enc = torch.zeros(d_latent)
    b_dec = torch.full((d_in,), 2.0)
    threshold = torch.zeros(d_latent)
    x = torch.full((1, 1, d_in), 3.0)

    clt_no_bias = LinearCLT(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        cfg={"d_in": d_in, "d_latent": d_latent, "d_out": d_in, "activation": "jumprelu"},
        threshold=threshold,
    )
    clt_with_bias = LinearCLT(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        cfg={
            "d_in": d_in,
            "d_latent": d_latent,
            "d_out": d_in,
            "activation": "jumprelu",
            "pre_encoder_bias": True,
        },
        threshold=threshold,
    )

    z_no_bias = clt_no_bias.encode(x)
    z_with_bias = clt_with_bias.encode(x)
    assert torch.allclose(z_no_bias, torch.full_like(z_no_bias, 3.0), atol=1e-6)
    assert torch.allclose(z_with_bias, torch.full_like(z_with_bias, 1.0), atol=1e-6)


def test_infer_cfg_sets_jumprelu_when_threshold_present():
    arrays = {
        "W_enc": np.random.randn(8, 16).astype(np.float32),
        "W_dec": np.random.randn(16, 8).astype(np.float32),
        "b_enc": np.zeros(16, dtype=np.float32),
        "b_dec": np.zeros(8, dtype=np.float32),
        "threshold": np.ones(16, dtype=np.float32),
    }
    cfg = _infer_cfg_from_weights(arrays)
    assert cfg["activation"] == "jumprelu"
    assert cfg["pre_encoder_bias"] is False


def test_load_clt_normalizes_jump_relu_alias_and_threshold(tmp_path):
    run_dir = tmp_path / "layer_0" / "width_16k" / "average_l0_71"
    run_dir.mkdir(parents=True, exist_ok=True)
    d_in = 8
    d_latent = 16
    np.savez(
        run_dir / "params.npz",
        W_enc=np.random.randn(d_in, d_latent).astype(np.float32),
        W_dec=np.random.randn(d_latent, d_in).astype(np.float32),
        b_enc=np.zeros(d_latent, dtype=np.float32),
        b_dec=np.zeros(d_in, dtype=np.float32),
        threshold=np.ones(d_latent, dtype=np.float32) * 0.5,
    )
    (run_dir / "cfg.json").write_text(
        json.dumps(
            {
                "d_in": d_in,
                "d_latent": d_latent,
                "d_out": d_in,
                "activation": "jump_relu",
                "encode_site": "resid_post",
                "decode_site": "resid_post",
                "writeback_site": "resid_post",
                "site_mode": "same_site_v1",
            }
        ),
        encoding="utf-8",
    )

    clt, _meta = load_clt(str(tmp_path), layer=0, width="16k", run_name="average_l0_71", device="cpu")
    assert clt._activation == "jumprelu"
    assert clt._threshold is not None

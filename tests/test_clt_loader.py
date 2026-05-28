from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aom.interventions.clt_loader import list_clt_runs, load_clt


def _write_clt_fixture(
    tmp_path,
    *,
    layer: int,
    width: str,
    run_name: str,
    d_in: int,
    d_latent: int,
    d_out: int,
    include_cfg: bool = True,
) -> str:
    run_dir = tmp_path / f"layer_{layer}" / f"width_{width}" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if include_cfg:
        cfg = {
            "d_in": int(d_in),
            "d_latent": int(d_latent),
            "d_out": int(d_out),
            "encode_site": "resid_post",
            "decode_site": "resid_post",
            "writeback_site": "resid_post",
            "site_mode": "same_site_v1",
            "activation": "relu",
        }
        (run_dir / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")

    # Intentionally flipped orientations to exercise canonicalization:
    # W_enc: (d_latent, d_in), W_dec: (d_out, d_latent)
    W_enc = np.random.randn(d_latent, d_in).astype(np.float32)
    W_dec = np.random.randn(d_out, d_latent).astype(np.float32)
    b_enc = np.random.randn(d_latent).astype(np.float32)
    b_dec = np.random.randn(d_out).astype(np.float32)
    np.savez(run_dir / "params.npz", W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec)
    return str(tmp_path)


def test_load_clt_local_fixture_transposes_weights(tmp_path):
    root = _write_clt_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_71", d_in=8, d_latent=16, d_out=8)
    clt, meta = load_clt(root, layer=0, width="16k", device="cpu", dtype="float32")

    assert meta.d_in == 8
    assert meta.d_latent == 16
    assert meta.d_out == 8
    assert clt.W_enc.shape == (8, 16)
    assert clt.W_dec.shape == (16, 8)
    assert meta.site_mode == "same_site_v1"

    x = torch.randn(2, 5, 8)
    z = clt.encode(x)
    y = clt.decode(z)
    assert z.shape == (2, 5, 16)
    assert y.shape == (2, 5, 8)


def test_load_clt_requires_run_name_when_ambiguous(tmp_path):
    root = _write_clt_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_71", d_in=8, d_latent=16, d_out=8)
    _ = _write_clt_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_105", d_in=8, d_latent=16, d_out=8)

    with pytest.raises(ValueError, match="Multiple CLT runs"):
        _ = load_clt(root, layer=0, width="16k", device="cpu", dtype="float32")

    clt, meta = load_clt(root, layer=0, width="16k", run_name="average_l0_71", device="cpu", dtype="float32")
    assert meta.run_name == "average_l0_71"
    assert hasattr(clt, "encode") and hasattr(clt, "decode")


def test_list_clt_runs_local_fixture(tmp_path):
    root = _write_clt_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_71", d_in=8, d_latent=16, d_out=8)
    _ = _write_clt_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_105", d_in=8, d_latent=16, d_out=8)
    runs = list_clt_runs(root, layer=0, width="16k")
    assert runs == sorted(runs)
    assert set(runs) == {"average_l0_71", "average_l0_105"}


def test_load_clt_infers_cfg_when_cfg_json_missing(tmp_path):
    root = _write_clt_fixture(
        tmp_path,
        layer=0,
        width="16k",
        run_name="average_l0_71",
        d_in=8,
        d_latent=16,
        d_out=8,
        include_cfg=False,
    )
    clt, meta = load_clt(root, layer=0, width="16k", run_name="average_l0_71", device="cpu", dtype="float32")

    assert meta.d_in == 8
    assert meta.d_latent == 16
    assert meta.d_out == 8
    assert meta.inferred_from_weights is True
    assert meta.cfg_path is None
    assert clt.cfg.get("inferred_from_weights") is True
    assert clt.cfg.get("dtype") == "float32"


def test_load_clt_hf_run_name_skips_directory_discovery(monkeypatch, tmp_path):
    try:
        import huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        pytest.skip("huggingface_hub not installed")

    params_path = tmp_path / "params.npz"
    d_in, d_latent, d_out = 8, 16, 8
    W_enc = np.random.randn(d_latent, d_in).astype(np.float32)
    W_dec = np.random.randn(d_out, d_latent).astype(np.float32)
    b_enc = np.random.randn(d_latent).astype(np.float32)
    b_dec = np.random.randn(d_out).astype(np.float32)
    np.savez(params_path, W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec)

    class _NoListApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("HfApi() should not be constructed when run_name is provided")

    def _fake_hf_hub_download(repo_id_or_path: str, *, filename: str, **kwargs) -> str:
        assert repo_id_or_path == "dummy/repo"
        assert filename in {
            "layer_0/width_16k/average_l0_71/params.npz",
            "layer_0/width_16k/average_l0_71/cfg.json",
        }
        if filename.endswith("params.npz"):
            return str(params_path)
        raise EntryNotFoundError("cfg.json not found")

    monkeypatch.setattr(huggingface_hub, "HfApi", _NoListApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)

    clt, meta = load_clt("dummy/repo", layer=0, width="16k", run_name="average_l0_71", device="cpu", dtype="float32")
    assert meta.run_name == "average_l0_71"
    assert meta.inferred_from_weights is True
    assert meta.cfg_path is None
    assert clt.W_enc.shape == (d_in, d_latent)
    assert clt.W_dec.shape == (d_latent, d_out)


def test_load_clt_rejects_cross_site_cfg(tmp_path):
    run_dir = tmp_path / "layer_0" / "width_16k" / "average_l0_71"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "d_in": 8,
        "d_latent": 16,
        "d_out": 8,
        "encode_site": "resid_pre",
        "decode_site": "mlp_out",
        "writeback_site": "mlp_out",
        "site_mode": "same_site_v1",
    }
    (run_dir / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    np.savez(
        run_dir / "params.npz",
        W_enc=np.random.randn(16, 8).astype(np.float32),
        W_dec=np.random.randn(8, 16).astype(np.float32),
        b_enc=np.random.randn(16).astype(np.float32),
        b_dec=np.random.randn(8).astype(np.float32),
    )
    with pytest.raises(ValueError, match="requires encode_site == decode_site == writeback_site"):
        _ = load_clt(str(tmp_path), layer=0, width="16k", run_name="average_l0_71", device="cpu", dtype="float32")


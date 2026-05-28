from __future__ import annotations

from types import SimpleNamespace

import pytest

from aom.provenance.protocol import ProtocolArgBinding, coerce_bool, enforce_protocol_bindings


def test_enforce_protocol_bindings_accepts_matching_values():
    args = SimpleNamespace(bootstrap_n=1000, bootstrap_seed=42, ci=0.95, require_git=True)
    cfg = {
        "bootstrap": {"n": 1000, "seed": 42, "ci": 0.95},
        "repro": {"require_git": True},
    }
    enforce_protocol_bindings(
        args=args,
        protocol_config=cfg,
        bindings=[
            ProtocolArgBinding("bootstrap_n", ("bootstrap", "n"), int),
            ProtocolArgBinding("bootstrap_seed", ("bootstrap", "seed"), int),
            ProtocolArgBinding("ci", ("bootstrap", "ci"), float),
            ProtocolArgBinding("require_git", ("repro", "require_git"), coerce_bool),
        ],
        context="unit",
    )


def test_enforce_protocol_bindings_rejects_mismatch():
    args = SimpleNamespace(bootstrap_n=2000, bootstrap_seed=42, ci=0.95, require_git=True)
    cfg = {"bootstrap": {"n": 1000, "seed": 42, "ci": 0.95}}
    with pytest.raises(ValueError, match="protocol_path"):
        enforce_protocol_bindings(
            args=args,
            protocol_config=cfg,
            bindings=[ProtocolArgBinding("bootstrap_n", ("bootstrap", "n"), int)],
            context="unit",
            protocol_path="/tmp/protocol.yaml",
            protocol_sha256="a" * 64,
        )


def test_enforce_protocol_bindings_missing_key_fails_when_required():
    args = SimpleNamespace(seed=42)
    cfg = {"bootstrap": {"n": 1000}}
    with pytest.raises(ValueError, match="requires protocol key"):
        enforce_protocol_bindings(
            args=args,
            protocol_config=cfg,
            bindings=[ProtocolArgBinding("seed", ("splits", "random_seed"), int, required=True)],
            context="unit",
        )


def test_enforce_protocol_bindings_missing_key_ignored_when_optional():
    args = SimpleNamespace(seed=42)
    cfg = {"bootstrap": {"n": 1000}}
    enforce_protocol_bindings(
        args=args,
        protocol_config=cfg,
        bindings=[ProtocolArgBinding("seed", ("splits", "random_seed"), int, required=False)],
        context="unit",
    )


def test_enforce_protocol_bindings_coerces_both_sides():
    args = SimpleNamespace(bootstrap_n="1000")
    cfg = {"bootstrap": {"n": 1000}}
    enforce_protocol_bindings(
        args=args,
        protocol_config=cfg,
        bindings=[ProtocolArgBinding("bootstrap_n", ("bootstrap", "n"), int)],
        context="unit",
    )


def test_enforce_protocol_bindings_bool_coerce_accepts_string_runtime():
    args = SimpleNamespace(require_git="true")
    cfg = {"repro": {"require_git": True}}
    enforce_protocol_bindings(
        args=args,
        protocol_config=cfg,
        bindings=[ProtocolArgBinding("require_git", ("repro", "require_git"), coerce_bool)],
        context="unit",
    )


def test_enforce_protocol_bindings_sequence_comparison_is_type_aware():
    args = SimpleNamespace(k_grid=(1, 2, 4, 8))
    cfg = {"multiplicity": {"k_grid": [1, 2, 4, 8]}}
    enforce_protocol_bindings(
        args=args,
        protocol_config=cfg,
        bindings=[ProtocolArgBinding("k_grid", ("multiplicity", "k_grid"))],
        context="unit",
    )


def test_enforce_protocol_bindings_string_comparison_strips_whitespace():
    args = SimpleNamespace(operator="mean_replacement ")
    cfg = {"head_ablation": {"operator": "mean_replacement"}}
    enforce_protocol_bindings(
        args=args,
        protocol_config=cfg,
        bindings=[ProtocolArgBinding("operator", ("head_ablation", "operator"))],
        context="unit",
    )

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import aom.models.loader as loader
from aom.models.loader import load_causal_lm


class _DummyModel:
    def __init__(self):
        self.config = object()

    def to(self, _device):
        return self

    def eval(self):
        return self


class _DummyTokenizer:
    def __init__(self, *, init_kwargs: dict):
        self.init_kwargs = dict(init_kwargs)
        self.pad_token_id = None
        self.eos_token = "[EOS]"
        self.pad_token = None


def test_load_causal_lm_threads_revisions(monkeypatch):
    captured: dict[str, dict] = {}

    def fake_model_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        captured["model"] = {"model_name_or_path": model_name_or_path, **kwargs}
        return _DummyModel()

    def fake_tokenizer_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        captured["tokenizer"] = {"model_name_or_path": model_name_or_path, **kwargs}
        return _DummyTokenizer(init_kwargs=kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(fake_model_from_pretrained))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(fake_tokenizer_from_pretrained))

    import aom.interventions.activation_patching as ap

    monkeypatch.setattr(ap, "detect_architecture", lambda _m: "dummy_arch")

    loaded = load_causal_lm(
        "dummy-model",
        device=torch.device("cpu"),
        revision="modelrev",
        tokenizer_revision="tokrev",
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map=None,
    )
    assert captured["model"]["revision"] == "modelrev"
    assert captured["tokenizer"]["revision"] == "tokrev"
    assert captured["model"]["local_files_only"] is True
    assert captured["tokenizer"]["local_files_only"] is True
    assert captured["model"]["trust_remote_code"] is True
    assert captured["tokenizer"]["trust_remote_code"] is True
    assert loaded.model_commit_hash is None
    assert loaded.tokenizer_revision_effective == "tokrev"


def test_load_causal_lm_tokenizer_revision_defaults_to_revision(monkeypatch):
    captured: dict[str, dict] = {}

    def fake_model_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        captured["model"] = {"model_name_or_path": model_name_or_path, **kwargs}
        return _DummyModel()

    def fake_tokenizer_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        captured["tokenizer"] = {"model_name_or_path": model_name_or_path, **kwargs}
        return _DummyTokenizer(init_kwargs=kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(fake_model_from_pretrained))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(fake_tokenizer_from_pretrained))

    import aom.interventions.activation_patching as ap

    monkeypatch.setattr(ap, "detect_architecture", lambda _m: "dummy_arch")

    load_causal_lm(
        "dummy-model",
        device=torch.device("cpu"),
        revision="modelrev",
        tokenizer_revision=None,
        local_files_only=False,
        trust_remote_code=False,
        attn_implementation="eager",
        device_map=None,
    )
    assert captured["tokenizer"]["revision"] == "modelrev"


def test_load_causal_lm_metadata_does_not_crash_without_config(monkeypatch):
    class _NoConfigModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def fake_model_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        return _NoConfigModel()

    def fake_tokenizer_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        return _DummyTokenizer(init_kwargs={})

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(fake_model_from_pretrained))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(fake_tokenizer_from_pretrained))

    import aom.interventions.activation_patching as ap

    monkeypatch.setattr(ap, "detect_architecture", lambda _m: "dummy_arch")

    loaded = load_causal_lm("dummy-model", device=torch.device("cpu"))
    assert loaded.model_commit_hash is None


def test_load_causal_lm_extracts_commit_hash_when_present(monkeypatch):
    class _ModelWithCommit:
        def __init__(self):
            self.config = type("Cfg", (), {"_commit_hash": "abc123"})()

        def to(self, _device):
            return self

        def eval(self):
            return self

    def fake_model_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        return _ModelWithCommit()

    def fake_tokenizer_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        return _DummyTokenizer(init_kwargs={})

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(fake_model_from_pretrained))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(fake_tokenizer_from_pretrained))

    import aom.interventions.activation_patching as ap

    monkeypatch.setattr(ap, "detect_architecture", lambda _m: "dummy_arch")

    loaded = load_causal_lm("dummy-model", device=torch.device("cpu"))
    assert loaded.model_commit_hash == "abc123"


def test_load_causal_lm_does_not_pass_revision_when_unset(monkeypatch):
    captured: dict[str, dict] = {}

    def fake_model_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        captured["model"] = {"model_name_or_path": model_name_or_path, **kwargs}
        return _DummyModel()

    def fake_tokenizer_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        captured["tokenizer"] = {"model_name_or_path": model_name_or_path, **kwargs}
        return _DummyTokenizer(init_kwargs=kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(fake_model_from_pretrained))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(fake_tokenizer_from_pretrained))

    import aom.interventions.activation_patching as ap

    monkeypatch.setattr(ap, "detect_architecture", lambda _m: "dummy_arch")

    load_causal_lm("dummy-model", device=torch.device("cpu"), revision=None, tokenizer_revision=None)
    assert "revision" not in captured["model"]
    assert "revision" not in captured["tokenizer"]


def test_load_causal_lm_resolves_local_snapshot_path_when_offline(monkeypatch):
    captured: dict[str, dict] = {}

    def fake_model_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        captured["model"] = {"model_name_or_path": model_name_or_path, **kwargs}
        return _DummyModel()

    def fake_tokenizer_from_pretrained(cls, model_name_or_path, **kwargs):  # noqa: ANN001
        captured["tokenizer"] = {"model_name_or_path": model_name_or_path, **kwargs}
        return _DummyTokenizer(init_kwargs=kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(fake_model_from_pretrained))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(fake_tokenizer_from_pretrained))
    monkeypatch.setattr(loader, "_resolve_local_snapshot_path", lambda *_args, **_kwargs: "/tmp/local-model")

    import aom.interventions.activation_patching as ap

    monkeypatch.setattr(ap, "detect_architecture", lambda _m: "dummy_arch")

    load_causal_lm(
        "google/gemma-2-2b",
        device=torch.device("cpu"),
        revision="rev123",
        tokenizer_revision="tok123",
        local_files_only=True,
    )

    assert captured["model"]["model_name_or_path"] == "/tmp/local-model"
    assert captured["tokenizer"]["model_name_or_path"] == "/tmp/local-model"
    assert "revision" not in captured["model"]
    assert "revision" not in captured["tokenizer"]

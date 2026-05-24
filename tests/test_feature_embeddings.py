"""Unit tests for ``legoloaderx.FeatureEmbeddings``.

Tests avoid touching the network — they build a product via ``from_matrix``
with a randomly-initialised weight tensor rather than running a real encoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from legoloaderx import FeatureEmbeddings, FeatureEmbeddingsConfig


# --------------------------------------------------------------- fixtures

@pytest.fixture
def groups():
    return {
        "treatments": ["pm25", "no2"],
        "confounders": ["population", "median_age", "pop_white"],
        "outcomes": ["anemia", "asthma"],
    }


@pytest.fixture
def vocab(groups):
    names = [n for stream in ("treatments", "confounders", "outcomes") for n in groups[stream]]
    return {n: i for i, n in enumerate(names)}


@pytest.fixture
def weight(vocab):
    torch.manual_seed(0)
    return torch.randn(len(vocab), 8)


@pytest.fixture
def emb(weight, vocab, groups):
    return FeatureEmbeddings.from_matrix(
        weight=weight,
        vocab=vocab,
        groups=groups,
        source_model="fake/encoder-v1",
        source_library="sentence_transformers",
        normalize=True,
    )


# --------------------------------------------------------------- basics

def test_shape_and_len(emb, vocab):
    assert len(emb) == emb.config.vocab_size == len(vocab) == 7
    assert emb.config.embed_dim == 8
    assert emb.embedding.weight.shape == (7, 8)


def test_weight_matches_source(emb, weight):
    assert torch.equal(emb.embedding.weight.detach(), weight.float())


def test_config_fields(emb):
    assert emb.config.source_model == "fake/encoder-v1"
    assert emb.config.source_library == "sentence_transformers"
    assert emb.config.normalize is True
    assert emb.config.model_type == "feature_embeddings"


def test_lookup(emb):
    assert emb.get_idx("pm25") == 0
    assert emb.get_idx("asthma") == 6
    ids = emb.get_ids(["pm25", "asthma", "population"])
    assert ids.tolist() == [0, 6, 2]
    assert ids.dtype == torch.long


def test_lookup_missing_raises(emb):
    with pytest.raises(KeyError):
        emb.get_idx("does_not_exist")


def test_group_ids(emb):
    assert emb.group_ids("treatments").tolist() == [0, 1]
    assert emb.group_ids("confounders").tolist() == [2, 3, 4]
    assert emb.group_ids("outcomes").tolist() == [5, 6]


def test_group_names_missing_raises(emb):
    with pytest.raises(KeyError):
        emb.group_names("exposures")


def test_forward(emb):
    ids = emb.get_ids(["pm25", "asthma"])
    out = emb(ids)
    assert out.shape == (2, emb.config.embed_dim)
    # Row 0 should equal the first weight row.
    assert torch.equal(out[0], emb.embedding.weight[0])


# --------------------------------------------------------------- validation

def test_non_dense_vocab_rejected(weight, groups):
    bad_vocab = {"a": 0, "b": 2}  # missing id 1
    bad_weight = torch.randn(2, 4)
    with pytest.raises(ValueError):
        FeatureEmbeddings.from_matrix(
            weight=bad_weight, vocab=bad_vocab, groups=groups,
            source_model="x", source_library="sentence_transformers",
        )


def test_vocab_weight_mismatch_rejected(groups):
    with pytest.raises(ValueError):
        FeatureEmbeddings.from_matrix(
            weight=torch.randn(3, 4),
            vocab={"a": 0, "b": 1},           # 2 entries but 3 rows
            groups=groups,
            source_model="x",
            source_library="sentence_transformers",
        )


def test_rejects_1d_weight(groups):
    with pytest.raises(ValueError):
        FeatureEmbeddings.from_matrix(
            weight=torch.randn(4),
            vocab={"a": 0, "b": 1, "c": 2, "d": 3},
            groups=groups,
            source_model="x",
            source_library="sentence_transformers",
        )


# --------------------------------------------------------------- serialisation

def test_save_pretrained_produces_two_files(emb, tmp_path):
    emb.save_pretrained(tmp_path)
    files = {p.name for p in tmp_path.iterdir()}
    assert "config.json" in files
    # HF saves either model.safetensors (preferred) or pytorch_model.bin. We
    # accept either to stay robust to the transformers default.
    assert ("model.safetensors" in files) or ("pytorch_model.bin" in files)


def test_roundtrip_preserves_everything(emb, tmp_path):
    emb.save_pretrained(tmp_path)
    loaded = FeatureEmbeddings.from_pretrained(tmp_path)

    assert loaded.config.vocab == emb.config.vocab
    assert loaded.config.groups == emb.config.groups
    assert loaded.config.source_model == emb.config.source_model
    assert loaded.config.source_library == emb.config.source_library
    assert loaded.config.normalize == emb.config.normalize
    assert loaded.config.embed_dim == emb.config.embed_dim
    assert loaded.config.vocab_size == emb.config.vocab_size

    assert torch.equal(
        loaded.embedding.weight.detach(),
        emb.embedding.weight.detach(),
    )


def test_roundtrip_lookups_still_work(emb, tmp_path):
    emb.save_pretrained(tmp_path)
    loaded = FeatureEmbeddings.from_pretrained(tmp_path)
    assert loaded.group_ids("confounders").tolist() == emb.group_ids("confounders").tolist()
    assert loaded.get_idx("pm25") == 0
    ids = loaded.get_ids(["anemia", "asthma"])
    out_loaded = loaded(ids)
    out_orig = emb(ids)
    assert torch.equal(out_loaded, out_orig)


def test_config_json_contains_vocab_and_groups(emb, tmp_path):
    emb.save_pretrained(tmp_path)
    with open(tmp_path / "config.json") as f:
        blob = json.load(f)
    assert blob["model_type"] == "feature_embeddings"
    assert blob["vocab"] == emb.config.vocab
    assert blob["groups"] == emb.config.groups
    assert blob["embed_dim"] == emb.config.embed_dim
    assert blob["vocab_size"] == emb.config.vocab_size


def test_loaded_config_round_trips_via_json(emb, tmp_path):
    emb.save_pretrained(tmp_path)
    cfg = FeatureEmbeddingsConfig.from_pretrained(tmp_path)
    assert cfg.vocab == emb.config.vocab
    assert cfg.groups == emb.config.groups


def test_to_device_and_back(emb, tmp_path):
    emb.save_pretrained(tmp_path)
    loaded = FeatureEmbeddings.from_pretrained(tmp_path)
    loaded_cpu = loaded.to("cpu")
    assert loaded_cpu.embedding.weight.device.type == "cpu"

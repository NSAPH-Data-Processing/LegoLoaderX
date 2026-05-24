"""Shared feature-embedding product, HuggingFace-native.

A ``FeatureEmbeddings`` instance is a tiny ``PreTrainedModel`` that bundles:

  - a *vocab*             ``{name: row_idx}`` (persisted inside ``config.json``)
  - an *embedding matrix* ``(vocab_size, embed_dim)`` float32 tensor (persisted
                          as ``model.safetensors``)
  - a *config*            ``FeatureEmbeddingsConfig`` with everything else
                          (groups, source_model, normalize flag, ...)

Because the class inherits from ``PreTrainedModel`` + ``PretrainedConfig``,
``from_pretrained`` / ``save_pretrained`` / safetensors I/O / hub push are
free — we only add the small name-lookup sugar and a producer-side builder
that runs a text encoder.

Producer (legoloaderx preprocessing)::

    emb = FeatureEmbeddings.from_descriptions(
        descriptions={"pm25": "...", "population": "...", ...},
        groups={"treatments": [...], "confounders": [...], "outcomes": [...]},
        source_model="BAAI/bge-small-en-v1.5",
        source_library="sentence_transformers",
        normalize=True,
    )
    emb.save_pretrained("<root_dir>/embeddings/BAAI/bge-small-en-v1.5")

Consumer (climhealth)::

    emb = FeatureEmbeddings.from_pretrained(
        "<root_dir>/embeddings/BAAI/bge-small-en-v1.5"
    )
    table      = emb.embedding                       # nn.Embedding(V, d)
    conf_ids   = emb.group_ids("confounders")        # Long tensor
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class FeatureEmbeddingsConfig(PretrainedConfig):
    """Config for :class:`FeatureEmbeddings`.

    ``vocab`` and ``groups`` are serialised inside ``config.json`` via the
    ``PretrainedConfig.to_json_file`` machinery — no custom handling needed.
    """

    model_type = "feature_embeddings"

    def __init__(
        self,
        vocab: Optional[Mapping[str, int]] = None,
        groups: Optional[Mapping[str, Sequence[str]]] = None,
        embed_dim: int = 0,
        source_model: str = "",
        source_library: str = "",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab: Dict[str, int] = dict(vocab) if vocab else {}
        self.groups: Dict[str, List[str]] = (
            {k: list(v) for k, v in groups.items()} if groups else {}
        )
        self.embed_dim = int(embed_dim)
        self.vocab_size = len(self.vocab)
        self.source_model = source_model
        self.source_library = source_library
        self.normalize = bool(normalize)


class FeatureEmbeddings(PreTrainedModel):
    """A shared feature-embedding table.

    The vocab is expected to be globally unique across streams — one name
    maps to one row in the matrix no matter which stream it belongs to.
    """

    config_class = FeatureEmbeddingsConfig
    base_model_prefix = "feature_embeddings"

    def __init__(self, config: FeatureEmbeddingsConfig):
        super().__init__(config)
        if config.vocab_size <= 0 or config.embed_dim <= 0:
            raise ValueError(
                f"vocab_size and embed_dim must be positive, got "
                f"vocab_size={config.vocab_size}, embed_dim={config.embed_dim}"
            )
        ids = sorted(config.vocab.values())
        if ids != list(range(config.vocab_size)):
            raise ValueError("config.vocab indices must be a dense 0..N-1 range")

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Standard HF initialisation hook; users can override.
        self.post_init()

    # Required by PreTrainedModel; used for _init_weights and hub templating.
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ forward

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)

    def __len__(self) -> int:
        return self.config.vocab_size

    # ------------------------------------------------------------------ lookup

    def get_idx(self, name: str) -> int:
        try:
            return self.config.vocab[name]
        except KeyError as e:
            raise KeyError(
                f"{name!r} not in vocab (size={self.config.vocab_size})"
            ) from e

    def get_ids(
        self,
        names: Iterable[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        ids = torch.tensor([self.get_idx(n) for n in names], dtype=torch.long)
        return ids.to(device) if device is not None else ids

    def group_names(self, stream: str) -> List[str]:
        if stream not in self.config.groups:
            raise KeyError(
                f"{stream!r} not in groups; available: {list(self.config.groups)}"
            )
        return list(self.config.groups[stream])

    def group_ids(
        self,
        stream: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.get_ids(self.group_names(stream), device=device)

    # ------------------------------------------------------------------ builders

    @classmethod
    def from_matrix(
        cls,
        weight: torch.Tensor,
        vocab: Mapping[str, int],
        groups: Mapping[str, Sequence[str]],
        source_model: str = "",
        source_library: str = "",
        normalize: bool = True,
    ) -> "FeatureEmbeddings":
        """Build directly from an existing weight tensor — useful for tests
        and for loading pre-computed vectors without going through an encoder.
        """
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
        if weight.shape[0] != len(vocab):
            raise ValueError(
                f"vocab has {len(vocab)} entries but weight has {weight.shape[0]} rows"
            )
        config = FeatureEmbeddingsConfig(
            vocab=vocab,
            groups=groups,
            embed_dim=int(weight.shape[1]),
            source_model=source_model,
            source_library=source_library,
            normalize=normalize,
        )
        model = cls(config)
        with torch.no_grad():
            model.embedding.weight.copy_(weight.detach().to(model.embedding.weight.dtype))
        return model

    @classmethod
    def from_descriptions(
        cls,
        descriptions: Mapping[str, str],
        groups: Mapping[str, Sequence[str]],
        source_model: str,
        source_library: str,
        normalize: bool = True,
    ) -> "FeatureEmbeddings":
        """Run the configured text encoder over ``descriptions`` and return
        a ready-to-save product. Row order follows ``descriptions.keys()``.
        """
        names = list(descriptions.keys())
        if len(names) != len(set(names)):
            raise ValueError("descriptions keys must be unique")
        texts = [descriptions[n] for n in names]

        weight = _encode(
            texts,
            source_model=source_model,
            source_library=source_library,
            normalize=normalize,
        )
        vocab = {name: i for i, name in enumerate(names)}
        return cls.from_matrix(
            weight=weight,
            vocab=vocab,
            groups=groups,
            source_model=source_model,
            source_library=source_library,
            normalize=normalize,
        )


# ---------------------------------------------------------------- encoders

def _encode(
    texts: Sequence[str],
    source_model: str,
    source_library: str,
    normalize: bool,
) -> torch.Tensor:
    if source_library == "sentence_transformers":
        return _encode_sentence_transformers(texts, source_model, normalize)
    if source_library == "transformers":
        return _encode_transformers(texts, source_model, normalize)
    raise ValueError(
        f"Unknown source_library {source_library!r}. "
        "Expected 'sentence_transformers' or 'transformers'."
    )


def _encode_sentence_transformers(
    texts: Sequence[str], model_name: str, normalize: bool
) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        list(texts),
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    return vectors.detach().cpu().float()


def _encode_transformers(
    texts: Sequence[str], model_name: str, normalize: bool
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    rows = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            pooled = model(**inputs).pooler_output.squeeze(0)
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
            rows.append(pooled.detach().cpu())
    return torch.stack(rows, dim=0).float()

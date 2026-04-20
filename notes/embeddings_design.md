# Feature-embedding preprocessing — design doc

Status: DRAFT (pending approval)
Issue: [#38 fetch and place embeddings in root_dir](https://github.com/NSAPH-Data-Processing/LegoLoaderX/issues/38)
Branch: `mauriciogtec/issue-38-preprocessing-embeddings`

---

## 1. Problem

Climhealth needs **per-variable text embeddings** — a fixed-size vector per
feature name, computed once from a natural-language description of the
variable (e.g. `"pm25" -> "Particulate Matter (PM2.5) concentration ..." -> R^384`).
These are used to:

- initialise a shared embedding table (`nn.Embedding`) inside the model;
- look up per-stream id tensors (`confounder_ids`, `treatment_ids`, ...) in
  `prepare_batch`;
- (eventually) replace ad-hoc per-branch embedding heads.

Today this is scattered:

- **Script**: `climhealth/data/processing/embed_icd10.py` — ~700 lines, half
  commented, mixes ICD-10 grouping logic with embedding generation.
- **Descriptions**: duplicated in `climhealth/conf/var_dict/lego.yaml::data_dict`,
  incomplete (no confounder or climate-type coverage).
- **Artifacts**: checked-in `.pth` files in `climhealth/ccw/`, `ccw_pqe/`,
  plus `data/embeddings/<model>/data_description_embeddings.json`.
- **Schema**: ad-hoc JSON keyed by name. No shared vocab, no matrix, no
  config.

## 2. Scope of this branch

Only the **producer**: owned by `legoloaderx`.

- Ship a single preprocessing step that writes a self-contained embedding
  artifact per source model to a shared path.
- Define the class that serialises/deserialises the artifact so both
  legoloaderx and climhealth consume it the same way.

Consumer wiring (loading into `HealthXDataset` / climhealth `ClimHealthModel`)
is a **follow-up branch** on climhealth, cherry-picking PR #104 where possible.

## 3. Three concepts

The product has three logical pieces, as flagged by the user:

| Concept          | What                                                  | Analogous HF piece                |
|------------------|-------------------------------------------------------|-----------------------------------|
| **Shared vocab** | `name -> int` mapping, stable across source models.   | `PreTrainedTokenizer` vocab       |
| **Embeddings**   | `float32` matrix of shape `(vocab_size, embed_dim)`.  | `nn.Embedding.weight`             |
| **Config / class** | Metadata + Python wrapper with `from_pretrained`.   | `PretrainedConfig` / `PreTrainedModel` |

The **vocab is a function of `conf/var_dict.yaml`, not of the source model** —
every model produces rows in the same order for the same names. This matters
because downstream code does `conf_ids = emb.get_ids(confounder_names)` and
gets a tensor that's valid no matter which source model is loaded.

## 4. Reinvent-the-wheel-the-least plan

HuggingFace already solves this problem class. Reuse:

- **`transformers.PretrainedConfig`** — base class. Gives us `from_json_file` /
  `to_json_file`, `from_pretrained` / `save_pretrained`, hub-push machinery.
- **`transformers.PreTrainedModel`** — base class. Handles save/load of
  `model.safetensors` + `config.json` atomically, including sharding if ever
  needed.
- **`safetensors`** — canonical tensor serialization (already a dep of
  `transformers`). Safer and faster than `torch.save`.
- **Optional future**: `huggingface_hub` push/pull — comes free once we
  subclass `PreTrainedModel`.

Net effect: we write ~80 lines of thin wrapper class + ~60 lines of Hydra
entrypoint. Serialization, deserialization, hub compatibility, safe tensor
format — all inherited.

## 5. On-disk layout (proposed)

Per source model, under `<output_dir>/<source_model>/`:

```
<output_dir>/<source_model>/
├── config.json        # PretrainedConfig JSON + {vocab, groups, source_model, ...}
└── model.safetensors  # single tensor {"embedding.weight": (V, d) fp32}
```

Two files, same convention as every HF checkpoint on the Hub.

`config.json` example (truncated):

```json
{
  "model_type": "feature_embeddings",
  "source_model": "BAAI/bge-small-en-v1.5",
  "source_library": "sentence_transformers",
  "normalize": true,
  "embed_dim": 384,
  "vocab_size": 92,
  "vocab": {
    "pm25": 0, "no2": 1, "o3": 2,
    "... all 92 names ...": 91
  },
  "groups": {
    "treatments":   ["pm25", "no2", ..., "PM25"],
    "confounders":  ["housing_occupied", ..., "EF"],
    "outcomes":     ["anemia", ..., "hypoth"]
  },
  "transformers_version": "..."
}
```

The vocab lives **inside** `config.json`, not as a separate `vocab.json`,
because (a) it is small (92 entries today, O(100s) long-term), (b) it binds
tightly to `vocab_size` and `groups`, and (c) it keeps the artifact a
2-file bundle instead of 3.

## 6. Class

```python
# legoloaderx/feature_embeddings.py
from transformers import PretrainedConfig, PreTrainedModel
import torch
import torch.nn as nn


class FeatureEmbeddingsConfig(PretrainedConfig):
    model_type = "feature_embeddings"

    def __init__(
        self,
        vocab: dict | None = None,
        groups: dict | None = None,
        embed_dim: int = 0,
        source_model: str = "",
        source_library: str = "",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab = vocab or {}
        self.groups = groups or {}
        self.embed_dim = embed_dim
        self.vocab_size = len(self.vocab)
        self.source_model = source_model
        self.source_library = source_library
        self.normalize = normalize


class FeatureEmbeddings(PreTrainedModel):
    config_class = FeatureEmbeddingsConfig

    def __init__(self, config: FeatureEmbeddingsConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)

    # --- HF surface already handles: from_pretrained, save_pretrained,
    #     push_to_hub, state_dict, safetensors I/O. We only add sugar. ---

    def get_idx(self, name: str) -> int: ...
    def get_ids(self, names) -> torch.Tensor: ...
    def group_ids(self, stream: str) -> torch.Tensor: ...
    @classmethod
    def from_descriptions(cls, descriptions, groups, source_model,
                          source_library, normalize=True) -> "FeatureEmbeddings": ...
```

Everything load/save related is inherited. `from_descriptions` is the
producer-side builder that runs the encoder and initialises weights; after
that the object is a normal `PreTrainedModel`.

## 7. Producer flow (snakemake)

```
conf/var_dict.yaml   ─┐
                      ├──► src/preprocessing_embeddings.py (Hydra)
conf/embeddings/      │         │
  snakemake.yaml   ──►│         │ uses FeatureEmbeddings.from_descriptions(...)
  config.yaml      ──►│         │ then .save_pretrained(<out_dir>/<model>)
                      │         ▼
snakefile_embeddings.smk   ─► <output_dir>/<source_model>/{config.json, model.safetensors}
```

Snakemake `rule embed_descriptions` fans out over the models declared in
`conf/embeddings/snakemake.yaml` (currently 3: BAAI/bge-small-en-v1.5,
emilyalsentzer/Bio_ClinicalBERT, FremyCompany/BioLORD-STAMB2-v1). Each rule
emits one artifact directory, atomic per-model.

## 8. Consumer flow (climhealth, follow-up branch)

```python
from legoloaderx import FeatureEmbeddings

emb   = FeatureEmbeddings.from_pretrained(f"{root_dir}/embeddings/{source_model}")
table = emb.embedding                                    # nn.Embedding(V, d)
conf_ids  = emb.group_ids("confounders").to(device)
treat_ids = emb.group_ids("treatments").to(device)
out_ids   = emb.group_ids("outcomes").to(device)
```

This is exactly the contract PR #104 set up on the climhealth side
(`confounder_ids`, `treatment_ids`, `feature_vocab_size` in the batch dict),
except now they come from a lego-shared artifact rather than being rebuilt
per-run from `variable_dict`.

## 9. Serialization semantics

- **Writer**: `FeatureEmbeddings.save_pretrained(out_dir)` (inherited from
  `PreTrainedModel`). Writes `config.json` + `model.safetensors` atomically.
- **Reader**: `FeatureEmbeddings.from_pretrained(out_dir)`. Loads both;
  reconstructs the `nn.Embedding` submodule with the right shape.
- **Hub-ready**: since the class inherits `PreTrainedModel`, doing
  `emb.push_to_hub("org/feature-embeddings-bge")` works with no extra code —
  useful if we ever want to distribute a canonical cohort-wide artifact.
- **Safetensors**: format is memory-mapped, safe to load (no pickle code
  execution), and what HF has standardised on.

## 10. Where outputs live

- **Development** (snakemake default): `<repo>/data/embeddings/<model>/`
  (gitignored via existing `data/*`).
- **Shared lab path** (production): override
  `output_dir=/n/dominici_lab/lab/lego_loader_x/embeddings` (cannon) or the
  equivalent fasse path. Same convention as `covars` / `health` already in
  `conf/datapaths/*.yaml`.
- Directory choice is a single Hydra override — no code change needed.

## 11. Open questions / things I want your call on

1. **Embed outcomes in the same vocab?** Currently I include outcomes in the
   shared table so `emb.get_ids(["anemia", ...])` works. PR #104 only shared
   vocab across confounders + treatments. Extra outcome rows cost nothing and
   keep `outcome_query_embeddings` sourceable from the same artifact.
   **Proposal: keep outcomes in.**
2. **One vocab across all source models, or per-model vocab?** Proposing
   identical vocab (derived deterministically from var_dict) — only the
   matrix changes across source models. Simpler consumer code.
3. **Shared root_dir today or later?** The snakefile defaults to local
   `data/embeddings`. Flipping to `/n/dominici_lab/lab/lego_loader_x/embeddings`
   is a single config override. Proposing: **leave default local; operators
   run snakemake with `output_dir=...` at deploy time.**
4. **Keep `FeatureEmbeddings` in legoloaderx or in a neutral package?**
   Proposing legoloaderx — same package that owns `var_dict.yaml` (the
   source of truth for the vocab) and `HealthXDataset` (the consumer-adjacent
   loader that could eventually hand the loaded embeddings to climhealth via
   the batch dict).

---

**Awaiting approval on §4, §5, §6, §11 before I finalise the class and re-run snakemake.**

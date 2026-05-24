"""Build a shared text-embedding table by walking ``conf/var_group/*.yaml``,
then save it as a HuggingFace-style artifact under ``{output_dir}/{model}/``.

Each var_group yaml contributes its ``vars:`` list plus a ``descriptions:`` dict
(``name -> sentence``). Which groups feed which stream is read from
``streams:`` in the Hydra config (``conf/embeddings/config.yaml``).

Encoding, serialization, and the on-disk schema live in
``legoloaderx.FeatureEmbeddings``. This script is just the Snakemake entrypoint.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from legoloaderx.feature_embeddings import FeatureEmbeddings


log = logging.getLogger(__name__)


STREAM_ORDER: Tuple[str, ...] = ("treatments", "confounders", "outcomes")


def _load_group(var_group_dir: str, name: str) -> dict:
    path = Path(var_group_dir) / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"var_group {name!r} has no yaml at {path}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def _collect(
    streams_cfg: Dict[str, List[str]],
    var_group_dir: str,
) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """Walk the requested groups per stream and build:
      - flat list of names in deterministic order (stream-order, list-order),
      - descriptions merged across groups,
      - groups mapping {stream -> [names]} preserving the same order.
    """
    flat: List[str] = []
    descriptions: Dict[str, str] = {}
    groups: Dict[str, List[str]] = {s: [] for s in STREAM_ORDER}

    for stream in STREAM_ORDER:
        group_names = streams_cfg.get(stream, []) or []
        for g in group_names:
            vg = _load_group(var_group_dir, g)
            vg_vars = vg.get("vars") or []
            vg_desc = vg.get("descriptions") or {}

            missing = [v for v in vg_vars if v not in vg_desc]
            if missing:
                raise ValueError(
                    f"var_group {g!r} is missing descriptions for: {missing}"
                )

            for name in vg_vars:
                if name in flat:
                    raise ValueError(
                        f"Duplicate variable name {name!r} across var_groups — "
                        "the shared vocab assumes globally-unique names."
                    )
                flat.append(name)
                descriptions[name] = vg_desc[name]
                groups[stream].append(name)

    return flat, descriptions, groups


@hydra.main(config_path="../conf/embeddings", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    streams_cfg = OmegaConf.to_container(cfg.streams, resolve=True)
    names, descriptions, groups = _collect(streams_cfg, cfg.var_group_dir)

    log.info(
        "Embedding %d descriptions (%d groups, streams=%s) with %s via %s",
        len(names),
        sum(len(v) for v in streams_cfg.values()),
        list(streams_cfg.keys()),
        cfg.model,
        cfg.library,
    )

    emb = FeatureEmbeddings.from_descriptions(
        descriptions={n: descriptions[n] for n in names},
        groups=groups,
        source_model=cfg.model,
        source_library=cfg.library,
        normalize=bool(cfg.normalize),
    )

    out_dir = os.path.join(cfg.output_dir, cfg.model)
    emb.save_pretrained(out_dir)
    log.info(
        "Wrote %s (vocab_size=%d, embed_dim=%d)",
        out_dir, len(emb), emb.config.embed_dim,
    )


if __name__ == "__main__":
    main()

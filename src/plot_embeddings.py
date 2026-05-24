"""Diagnostic plots for a ``FeatureEmbeddings`` artifact.

Two subcommands, both Snakemake-friendly:

    python src/plot_embeddings.py per-model \
        --embeddings-dir data/embeddings/<model> \
        --out-dir data/embeddings/<model>/plots

    python src/plot_embeddings.py compare \
        --out data/embeddings/plots/model_comparison.png \
        --embeddings-dirs data/embeddings/<model1> data/embeddings/<model2> ...

Per-model outputs:
    latent_pca.png           2D PCA of the rows, coloured by stream
    latent_tsne.png          2D t-SNE of the rows, coloured by stream
    similarity_heatmap.png   cosine-similarity matrix, hierarchically clustered
                             with annotation bars + dendrogram
    top_pairs.txt            top-K most similar cross-stream pairs (K=20)

Compare output:
    A 2 x N grid (top row PCA, bottom row t-SNE) with a single shared legend.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from legoloaderx.feature_embeddings import FeatureEmbeddings


# ---------------------------------------------------------------- aesthetics

# Okabe-Ito colour-blind-safe palette. We reuse blue/orange/vermilion for the
# three streams (high contrast, prints well in greyscale).
STREAM_COLOURS: Dict[str, str] = {
    "treatments": "#0072B2",   # blue
    "confounders": "#E69F00",  # orange
    "outcomes": "#D55E00",     # vermilion
    "other": "#999999",        # neutral grey fallback
}

STREAM_ORDER = ("treatments", "confounders", "outcomes", "other")


def _apply_style() -> None:
    """Apply a consistent matplotlib style across every figure."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _scatter_style(ax: plt.Axes) -> None:
    """Scatter-plot axes: no gridlines, no ticks, keep a tight frame."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#888888")


# ---------------------------------------------------------------- data access

def _stream_of(name: str, groups: Dict[str, List[str]]) -> str:
    for stream, names in groups.items():
        if name in names:
            return stream
    return "other"


def _load(path: str | Path) -> FeatureEmbeddings:
    return FeatureEmbeddings.from_pretrained(str(path))


def _weight_array(emb: FeatureEmbeddings) -> np.ndarray:
    return emb.embedding.weight.detach().cpu().numpy()


def _names_and_streams(emb: FeatureEmbeddings) -> Tuple[List[str], List[str]]:
    names = sorted(emb.config.vocab, key=emb.config.vocab.get)
    streams = [_stream_of(n, emb.config.groups) for n in names]
    return names, streams


def _pretty_model_name(name: str) -> str:
    """Strip org prefix for compact display: "BAAI/bge-small-en-v1.5" -> "bge-small-en-v1.5"."""
    if "/" in name:
        return name.split("/", 1)[1]
    return name


# -------------------------------------------------------- projections / math

def _pca_2d(x: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=0).fit_transform(x)


def _tsne_2d(x: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE
    perplexity = max(2.0, min(30.0, (x.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=0
    ).fit_transform(x)


def _cosine_similarity(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    xn = x / norms
    return xn @ xn.T


def _hierarchical_order(sim: np.ndarray):
    """Return (leaf_order, linkage_matrix) for agglomerative clustering on
    ``1 - sim`` using average linkage. Uses scipy's ``optimal_ordering`` to
    make neighbour rows visually similar.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    # symmetrise, clamp, zero the diagonal; convert to condensed distance.
    d = 1.0 - np.clip((sim + sim.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(d, 0.0)
    # Distances must be non-negative; clip any tiny numerical negatives.
    d = np.clip(d, 0.0, None)
    condensed = squareform(d, checks=False)
    z = linkage(condensed, method="average", optimal_ordering=True)
    order = leaves_list(z)
    return np.asarray(order), z


# ---------------------------------------------------------------- drawing

def _scatter(
    ax: plt.Axes,
    points: np.ndarray,
    streams: List[str],
    title: str,
    show_legend: bool = True,
) -> None:
    for stream in STREAM_ORDER:
        mask = np.array([s == stream for s in streams])
        if not mask.any():
            continue
        ax.scatter(
            points[mask, 0], points[mask, 1],
            c=STREAM_COLOURS[stream], label=stream, s=26, alpha=0.85,
            edgecolors="white", linewidths=0.4,
        )
    ax.set_title(title)
    _scatter_style(ax)
    if show_legend:
        ax.legend(loc="best")


def _corner_annotation(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.99, 0.01, text,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#555555",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
    )


# ---------------------------------------------------------------- per-model

def _save_pca(out_path: str, w: np.ndarray, streams: List[str], model_name: str) -> None:
    pts = _pca_2d(w)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _scatter(ax, pts, streams, f"PCA  —  {_pretty_model_name(model_name)}")
    _corner_annotation(ax, f"n={w.shape[0]}, d={w.shape[1]}")
    fig.savefig(out_path)
    plt.close(fig)


def _save_tsne(out_path: str, w: np.ndarray, streams: List[str], model_name: str) -> None:
    pts = _tsne_2d(w)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _scatter(ax, pts, streams, f"t-SNE  —  {_pretty_model_name(model_name)}")
    _corner_annotation(ax, f"n={w.shape[0]}, d={w.shape[1]}")
    fig.savefig(out_path)
    plt.close(fig)


def _save_heatmap(
    out_path: str,
    w: np.ndarray,
    names: List[str],
    streams: List[str],
    model_name: str,
) -> None:
    """Hierarchically-clustered cosine-similarity heatmap with dendrogram and
    stream annotation bars.
    """
    from scipy.cluster.hierarchy import dendrogram

    sim = _cosine_similarity(w)
    order, z = _hierarchical_order(sim)
    sim_ord = sim[np.ix_(order, order)]
    names_ord = [names[i] for i in order]
    streams_ord = [streams[i] for i in order]

    # Layout: top dendrogram, left dendrogram, annotation strips, heatmap, colorbar.
    #   rows: [top_dendro, top_bar,       heatmap      ]
    #   cols: [left_dendro, left_bar, heatmap, colorbar]
    fig = plt.figure(figsize=(10, 9))
    gs = GridSpec(
        3, 4,
        width_ratios=[0.08, 0.025, 1.0, 0.03],
        height_ratios=[0.08, 0.025, 1.0],
        wspace=0.02, hspace=0.02,
        left=0.08, right=0.95, top=0.93, bottom=0.22,
    )

    ax_top = fig.add_subplot(gs[0, 2])
    ax_left = fig.add_subplot(gs[2, 0])
    ax_top_bar = fig.add_subplot(gs[1, 2])
    ax_left_bar = fig.add_subplot(gs[2, 1])
    ax_heat = fig.add_subplot(gs[2, 2])
    ax_cbar = fig.add_subplot(gs[2, 3])

    # Dendrograms, drawn with muted lines.
    with plt.rc_context({"lines.linewidth": 0.6}):
        dendrogram(z, ax=ax_top, orientation="top", no_labels=True,
                   color_threshold=0, above_threshold_color="#555555")
        dendrogram(z, ax=ax_left, orientation="left", no_labels=True,
                   color_threshold=0, above_threshold_color="#555555")
    for a in (ax_top, ax_left):
        a.set_xticks([]); a.set_yticks([])
        for spine in a.spines.values():
            spine.set_visible(False)
        a.grid(False)

    # Annotation bars (one coloured cell per row/column, by stream).
    stream_rgb = np.array([mpl.colors.to_rgb(STREAM_COLOURS[s]) for s in streams_ord])
    # top bar: one row of colours, one per column
    ax_top_bar.imshow(stream_rgb[None, :, :], aspect="auto", interpolation="nearest")
    ax_top_bar.set_xticks([]); ax_top_bar.set_yticks([])
    for spine in ax_top_bar.spines.values():
        spine.set_visible(False)
    ax_top_bar.grid(False)
    # left bar: one column of colours, one per row
    ax_left_bar.imshow(stream_rgb[:, None, :], aspect="auto", interpolation="nearest")
    ax_left_bar.set_xticks([]); ax_left_bar.set_yticks([])
    for spine in ax_left_bar.spines.values():
        spine.set_visible(False)
    ax_left_bar.grid(False)

    # Heatmap
    im = ax_heat.imshow(sim_ord, cmap="viridis", vmin=-1, vmax=1, aspect="auto",
                        interpolation="nearest")
    ax_heat.grid(False)

    # Tick labels — skip every other if it would help readability.
    n = len(names_ord)
    label_stride = 2 if n > 60 else 1
    tick_idx = np.arange(0, n, label_stride)
    tick_labels = [names_ord[i] for i in tick_idx]
    label_fs = 7 if n > 60 else 8

    ax_heat.set_xticks(tick_idx)
    ax_heat.set_xticklabels(tick_labels, rotation=45, ha="right",
                            fontsize=label_fs, rotation_mode="anchor")
    ax_heat.set_yticks(tick_idx)
    ax_heat.set_yticklabels(tick_labels, fontsize=label_fs)
    ax_heat.tick_params(axis="both", which="both", length=0, pad=1)

    # Stream boundary lines, computed from the clustered ordering.
    boundaries: List[float] = []
    prev = None
    for i, s in enumerate(streams_ord):
        if s != prev and i > 0:
            boundaries.append(i - 0.5)
        prev = s
    for b in boundaries:
        ax_heat.axhline(b, color="white", lw=0.7, alpha=0.9)
        ax_heat.axvline(b, color="white", lw=0.7, alpha=0.9)

    # Colorbar
    fig.colorbar(im, cax=ax_cbar)
    ax_cbar.set_ylabel("cosine similarity", fontsize=9)

    # Legend for annotation bars (streams present in this dataset).
    present_streams = [s for s in STREAM_ORDER if s in set(streams_ord)]
    handles = [Patch(facecolor=STREAM_COLOURS[s], edgecolor="none", label=s)
               for s in present_streams]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.52, 0.02), frameon=False)

    fig.suptitle(
        f"Cosine similarity  —  {_pretty_model_name(model_name)}",
        fontsize=13, y=0.985,
    )
    # Small caption with n/d under the title.
    fig.text(0.52, 0.955, f"n={w.shape[0]}, d={w.shape[1]}  |  rows ordered by hierarchical clustering (avg linkage, 1 - cos)",
             ha="center", va="top", fontsize=9, color="#555555")

    fig.savefig(out_path)
    plt.close(fig)


def _save_top_pairs(
    out_path: str,
    w: np.ndarray,
    names: List[str],
    streams: List[str],
    k: int = 20,
) -> List[Tuple[str, str, str, str, float]]:
    """Rank the top-K most similar *cross-stream* pairs and write a small
    plain-text table. Returns the ranked list in case the caller wants to
    do more with it.
    """
    sim = _cosine_similarity(w)
    n = sim.shape[0]
    streams_arr = np.asarray(streams)

    pairs: List[Tuple[str, str, str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if streams_arr[i] == streams_arr[j]:
                continue
            pairs.append(
                (names[i], streams_arr[i], names[j], streams_arr[j], float(sim[i, j]))
            )

    pairs.sort(key=lambda t: t[4], reverse=True)
    top = pairs[:k]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Top-{k} most similar cross-stream pairs (cosine similarity)\n")
        f.write(f"# total candidate pairs: {len(pairs)}\n")
        f.write("rank\tcosine\tname_a\tstream_a\tname_b\tstream_b\n")
        for rank, (na, sa, nb, sb, c) in enumerate(top, start=1):
            f.write(f"{rank}\t{c:+.4f}\t{na}\t{sa}\t{nb}\t{sb}\n")
    return top


def per_model(embeddings_dir: str, out_dir: str) -> None:
    _apply_style()
    emb = _load(embeddings_dir)
    w = _weight_array(emb)
    names, streams = _names_and_streams(emb)

    os.makedirs(out_dir, exist_ok=True)

    _save_pca(os.path.join(out_dir, "latent_pca.png"), w, streams, emb.config.source_model)
    _save_tsne(os.path.join(out_dir, "latent_tsne.png"), w, streams, emb.config.source_model)
    _save_heatmap(
        os.path.join(out_dir, "similarity_heatmap.png"),
        w, names, streams, emb.config.source_model,
    )
    _save_top_pairs(
        os.path.join(out_dir, "top_pairs.txt"),
        w, names, streams, k=20,
    )


# ---------------------------------------------------------------- compare

def _scatter_into(
    ax: plt.Axes,
    points: np.ndarray,
    streams: List[str],
    title: str,
) -> None:
    """Scatter without a per-axis legend (legend is shared at figure level)."""
    for stream in STREAM_ORDER:
        mask = np.array([s == stream for s in streams])
        if not mask.any():
            continue
        ax.scatter(
            points[mask, 0], points[mask, 1],
            c=STREAM_COLOURS[stream], label=stream, s=22, alpha=0.85,
            edgecolors="white", linewidths=0.3,
        )
    ax.set_title(title)
    _scatter_style(ax)


def compare(out: str, embeddings_dirs: Sequence[str]) -> None:
    _apply_style()
    embs = [_load(d) for d in embeddings_dirs]
    n = len(embs)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Precompute projections so we don't re-load weights per row.
    cached = []
    for emb in embs:
        w = _weight_array(emb)
        names, streams = _names_and_streams(emb)
        cached.append({
            "model": emb.config.source_model,
            "w": w,
            "streams": streams,
            "pca": _pca_2d(w),
            "tsne": _tsne_2d(w),
        })

    fig, axes = plt.subplots(
        2, n,
        figsize=(5.2 * n, 9),
        squeeze=False,
    )

    for j, info in enumerate(cached):
        label = _pretty_model_name(info["model"])
        ax_top = axes[0, j]
        ax_bot = axes[1, j]
        _scatter_into(ax_top, info["pca"], info["streams"], label)
        _scatter_into(ax_bot, info["tsne"], info["streams"], "")
        _corner_annotation(ax_top, f"n={info['w'].shape[0]}, d={info['w'].shape[1]}")

    # Row labels on the far-left axes.
    axes[0, 0].set_ylabel("PCA", fontsize=12, fontweight="semibold", labelpad=8)
    axes[1, 0].set_ylabel("t-SNE", fontsize=12, fontweight="semibold", labelpad=8)

    # Shared legend across the figure.
    present = sorted(
        {s for info in cached for s in info["streams"]},
        key=lambda s: STREAM_ORDER.index(s) if s in STREAM_ORDER else len(STREAM_ORDER),
    )
    handles = [Patch(facecolor=STREAM_COLOURS[s], edgecolor="none", label=s)
               for s in present]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        "Feature-embedding projections — source-model comparison",
        fontsize=14, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------- CLI

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("per-model")
    pm.add_argument("--embeddings-dir", required=True)
    pm.add_argument("--out-dir", required=True)

    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--out", required=True)
    cmp_.add_argument("--embeddings-dirs", nargs="+", required=True)

    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cmd == "per-model":
        per_model(args.embeddings_dir, args.out_dir)
    elif args.cmd == "compare":
        compare(args.out, args.embeddings_dirs)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

BarDict = Dict[Union[int, float, str], Union[int, float]]

_LABELS = {
    "ru": {
        "share": "share p(k)",
        "theory_title": "theory",
        "theory_x": "bin size k",
        "sim_title": "simulation",
        "sim_x": "cluster size k",
        "theory_only": "target distribution",
    },
    "en": {
        "share": "share p(k)",
        "theory_title": "theory",
        "theory_x": "bin size k",
        "sim_title": "simulation",
        "sim_x": "cluster size k",
        "theory_only": "target distribution",
    },
}


def normalize_bar(bar: BarDict) -> Tuple[List[float], List[float]]:
    """Return sorted keys and normalized values"""
    keys = sorted(float(k) for k in bar.keys())
    total = sum(float(v) for v in bar.values()) or 1.0
    vals: List[float] = []
    for k in keys:
        v = 0.0
        for bk, bv in bar.items():
            if abs(float(bk) - k) < 1e-9:
                v = float(bv)
                break
        vals.append(v / total)
    return keys, vals


def _labels(lang: str) -> dict:
    return _LABELS.get(lang, _LABELS["ru"])


def populate_theory_axis(ax, theory: BarDict, *, lang: str = "ru", title_suffix: str = "") -> None:
    lb = _labels(lang)
    k, v = normalize_bar(theory)
    ax.plot(k, v, "o-", color="#4c78a8", markersize=4, linewidth=1.2)
    ax.set_ylabel(lb["share"])
    title = lb["theory_title"]
    if title_suffix:
        title = f"{title_suffix} — {title}"
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(lb["theory_x"])


def populate_sim_axis(ax, experimental: BarDict, *, lang: str = "ru", title_suffix: str = "") -> None:
    lb = _labels(lang)
    k, v = normalize_bar(experimental)
    ax.bar(k, v, width=0.8, color="#54a24b", edgecolor="#2d5a27")
    ax.set_ylabel(lb["share"])
    title = lb["sim_title"]
    if title_suffix:
        title = f"{title_suffix} — {title}"
    ax.set_title(title)
    ax.set_xlabel(lb["sim_x"])
    ax.grid(True, axis="y", alpha=0.3)


def save_theory_only_figure(
    theory: BarDict,
    *,
    title: str,
    out_path: Path,
    dpi: int = 150,
    lang: str = "ru",
    figsize: Tuple[float, float] = (9.0, 4.0),
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    populate_theory_axis(ax, theory, lang=lang)
    if title:
        ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_split_figure(
    theory: BarDict,
    experimental: Optional[BarDict],
    *,
    title: str = "",
    out_path: Path,
    dpi: int = 150,
    lang: str = "ru",
    figsize: Tuple[float, float] = (9.0, 6.0),
    loss: Optional[float] = None,
    method_label: str = "",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subtitle = method_label
    if loss is not None:
        subtitle = f"{method_label}, L={loss:.3f}".strip(", ")

    if experimental and sum(float(v) for v in experimental.values()) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1.2, 1]})
        populate_theory_axis(ax1, theory, lang=lang)
        populate_sim_axis(ax2, experimental, lang=lang)
        if title:
            fig.suptitle(title, fontsize=11, y=1.02)
        elif subtitle:
            fig.suptitle(subtitle, fontsize=11, y=1.02)
    else:
        fig, ax1 = plt.subplots(figsize=(figsize[0], figsize[1] * 0.55))
        populate_theory_axis(ax1, theory, lang=lang, title_suffix=title or subtitle)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
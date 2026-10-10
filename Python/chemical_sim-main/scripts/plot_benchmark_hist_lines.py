"""
Linear comparison of p(k): like the GUI window "Comparison: p(k) lines" (tabs All methods / Top-5).

From benchmark_inverse-*.json, reconstructs histograms from best_params and plots:
  — theory: black line, marker ^
  — methods: colored lines o-, in the legend "N. Name  L=..."

"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classes.simulation_objective import STEPS_PER_TRIAL, SimulationObjective  
from resources import benchmark_protocol as proto  
from resources.gui_benchmark_export import BENCHMARK_JSON_KEY_TO_ALGO_LABEL  

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("matplotlib required:", e, file=sys.stderr)
    raise SystemExit(1)

CMAP = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class _Sleeper:
    is_exit = False
    scale_sleep = SimpleNamespace(get=lambda: 0)


def _p_at_key(bar: dict, kf: float, s: float) -> float:
    for key in bar:
        if abs(float(key) - kf) < 1e-9:
            return float(bar[key]) / s
    return 0.0


def _key_label(kf: float) -> str:
    if abs(kf - int(kf)) < 1e-9:
        return str(int(round(kf)))
    return f"{kf:g}"


def _display_name(mkey: str, mname: str) -> str:
    if mkey in BENCHMARK_JSON_KEY_TO_ALGO_LABEL:
        return BENCHMARK_JSON_KEY_TO_ALGO_LABEL[mkey]
    low = mname.strip().lower().replace(" ", "_")
    for jk, label in BENCHMARK_JSON_KEY_TO_ALGO_LABEL.items():
        if low == jk or low in jk or jk in low:
            return label
    return mname or mkey


def _target_id(raw: str) -> str:
    if raw.startswith("preset:"):
        return raw
    return f"preset:{raw}"


def _hist_preset_name(target_id: str) -> str:
    m = re.search(r"hist(\d+)", target_id, re.I)
    if m:
        return f"hist{m.group(1)}"
    return re.sub(r"^preset:", "", target_id)


def _find_target_run(payload: dict, target: str, run_index: int) -> Tuple[dict, dict]:
    tid = _target_id(target)
    for tgt in payload.get("targets", []):
        label = str(tgt.get("id", ""))
        if label != tid and tid not in label and target not in label:
            continue
        for run in tgt.get("runs", []):
            if int(run.get("run_index", -1)) == run_index:
                return tgt, run
    raise KeyError(f"target={target!r} run_index={run_index} not in JSON")


def _replay_best_hist(
    *,
    rows: int,
    theory: dict,
    params: dict,
    n_repeats: int,
    run_index: int,
    steps: int,
) -> Tuple[float, dict]:
    sim = SimulationObjective(rows=rows, theory=theory, sleeper=_Sleeper())
    losses: List[float] = []
    hists: List[dict] = []
    for rep in range(n_repeats):
        loss, hist = sim.run_one_repeat(
            float(params["add"]),
            float(params["transit"]),
            float(params["merge"]),
            eval_index=run_index,
            rep=rep,
            steps=steps,
        )
        losses.append(loss)
        hists.append(hist)
    best_rep = int(np.argmin(losses))
    return float(losses[best_rep]), hists[best_rep]


def _collect_plottable(run: dict) -> List[Tuple[str, str, float, dict]]:
    """(mkey, display_name, best_loss, params), sorted by loss."""
    rows_out: List[Tuple[str, str, float, dict]] = []
    for mkey, block in run.get("methods", {}).items():
        if not isinstance(block, dict):
            continue
        bl = block.get("best_loss")
        params = block.get("best_params")
        if bl is None or not isinstance(params, dict):
            continue
        mname = str(block.get("method") or mkey)
        rows_out.append((mkey, _display_name(mkey, mname), float(bl), params))
    rows_out.sort(key=lambda x: x[2])
    return rows_out


def _build_axis(
    theory: dict,
    bars: List[dict],
) -> Tuple[List[float], List[float]]:
    s_t = float(sum(float(v) for v in theory.values()))
    if s_t <= 0:
        raise ValueError("theory sum is zero")
    all_k_float = {float(x) for x in theory.keys()}
    for bar in bars:
        all_k_float |= {float(x) for x in bar.keys()}
    all_ks = sorted(all_k_float)
    p_theory = [_p_at_key(theory, kf, s_t) for kf in all_ks]
    return all_ks, p_theory


def populate_mode_compare_ax(
    ax,
    *,
    all_ks: List[float],
    p_theory: List[float],
    series: List[Tuple[str, float, dict]],
    title: str,
    thick_lines: bool,
    lang: str,
) -> None:
    """Like ui_tkinter._populate_mode_compare_ax."""
    x_idx = list(range(len(all_ks)))
    ax.set_xlabel(
        "Cluster size index (see tick labels)"
        if lang == "en"
        else "Индекс размера кластера (см. подписи по Ox)",
        fontsize=9,
    )
    ax.set_ylabel("share p(k)" if lang == "en" else "Доля p(k)", fontsize=9)
    ax.set_title(title, fontsize=10)

    ms_theory = 7 if thick_lines else 5
    lw_theory = 2.4 if thick_lines else 1.5
    ax.plot(
        x_idx,
        p_theory,
        linestyle="-",
        marker="^",
        color="#111",
        label="Theory" if lang == "en" else "Теория",
        markersize=ms_theory,
        linewidth=lw_theory,
        zorder=1000,
    )

    n_pt = len(series)
    ms = 5 if thick_lines else 3.5
    lw = 1.6 if thick_lines else 1.0
    base_alpha = 0.88 if thick_lines else 0.82
    ymax = max(p_theory) if p_theory else 0.12

    for idx, (name, bl, bar) in enumerate(series):
        s_e = float(sum(float(x) for x in bar.values()))
        if s_e <= 0:
            continue
        c = CMAP[idx % len(CMAP)]
        p_ex = [_p_at_key(bar, kf, s_e) for kf in all_ks]
        ymax = max(ymax, max(p_ex) if p_ex else 0.0)
        ls = f"  L={bl:.2f}" if math.isfinite(bl) else f"  L={bl}"
        nm = name if len(name) < 36 else name[:33] + "…"
        short = f"{idx + 1}. {nm}{ls}"
        ax.plot(
            x_idx,
            p_ex,
            "o-",
            color=c,
            label=short,
            markersize=ms,
            linewidth=lw,
            alpha=base_alpha,
            zorder=10 + (n_pt - idx),
        )

    ax.set_xticks(x_idx)
    ax.set_xticklabels([_key_label(k) for k in all_ks], rotation=50, ha="right", fontsize=7)
    ax.set_ylim(0, min(1.0, max(ymax * 1.12, 0.12)))
    n_leg = 1 + n_pt
    ax.legend(
        loc="center left" if n_leg > 6 else "upper left",
        bbox_to_anchor=(1.02, 0.5) if n_leg > 6 else (0.98, 0.98),
        fontsize=7 if thick_lines else 6,
        borderaxespad=0.8,
        framealpha=0.95,
    )
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)


def _render_and_save(
    *,
    all_ks: List[float],
    p_theory: List[float],
    series: List[Tuple[str, float, dict]],
    title: str,
    thick_lines: bool,
    lang: str,
    fig_width: float,
    fig_height: float,
    dpi: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    populate_mode_compare_ax(
        ax,
        all_ks=all_ks,
        p_theory=p_theory,
        series=series,
        title=title,
        thick_lines=thick_lines,
        lang=lang,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout(rect=[0, 0.03, 0.78 if len(series) > 5 else 0.82, 0.96])
    except Exception:
        fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Comparison of p(k) lines (GUI style) from benchmark_inverse JSON."
    )
    p.add_argument("json_path", type=str)
    p.add_argument("--target", type=str, default="hist1", help="hist1 / hist2 / preset:hist1")
    p.add_argument("--run-index", type=int, default=0)
    p.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of best methods in a separate figure (0 — do not save).",
    )
    p.add_argument(
        "--all-methods",
        action="store_true",
        help="Additionally save a figure with all methods (like the 'All methods' tab).",
    )
    p.add_argument("--methods", nargs="*", default=None, help="Subset of JSON keys: gp pso de etc.")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--lang", choices=("ru", "en"), default="ru")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--fig-width", type=float, default=14.0)
    p.add_argument("--fig-height", type=float, default=8.0)
    args = p.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    _tgt, run = _find_target_run(data, args.target, args.run_index)

    hist_name = _hist_preset_name(_target_id(args.target))
    theory = proto.load_theory_preset(hist_name)
    rows = int(meta.get("rows", proto.DEFAULT_ROWS))
    n_repeats = int(meta.get("n_repeats", proto.DEFAULT_N_REPEATS))
    steps = int(meta.get("steps_per_trial", STEPS_PER_TRIAL))

    blocks = _collect_plottable(run)
    if args.methods:
        want = {m.strip().lower() for m in args.methods}
        blocks = [b for b in blocks if b[0].lower() in want]

    if not blocks:
        print("No methods to plot.", file=sys.stderr)
        return 1

    replayed: List[Tuple[str, float, dict]] = []
    for _mkey, dname, json_loss, params in blocks:
        _sim_loss, hist = _replay_best_hist(
            rows=rows,
            theory=theory,
            params=params,
            n_repeats=n_repeats,
            run_index=args.run_index,
            steps=steps,
        )
        if hist and float(sum(float(x) for x in hist.values())) > 0:
            replayed.append((dname, json_loss, hist))
            print(f"  {dname}: L={json_loss:.2f}")

    if not replayed:
        print("No replayed histograms.", file=sys.stderr)
        return 1

    all_bars = [h for _n, _l, h in replayed]
    all_ks, p_theory = _build_axis(theory, all_bars)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.json_path).parent / "figures"
    safe_tgt = re.sub(r"[^\w\-]+", "_", _target_id(args.target))[:40]
    run_tag = f"run{args.run_index}"

    tgt_label = _target_id(args.target)
    if args.lang == "en":
        title_all = "Theory and best p(k) per algorithm; legend ranked by L"
        title_top = "Top five methods by L" if args.top_n == 5 else f"Top {args.top_n} methods by L"
    else:
        title_all = "Теория и лучшие p(k) по алгоритмам; в легенде — ранг по L"
        title_top = "Пять лучших методов по L" if args.top_n == 5 else f"Топ-{args.top_n} методов по L"

    save_all = args.all_methods or args.top_n <= 0
    if save_all:
        _render_and_save(
            all_ks=all_ks,
            p_theory=p_theory,
            series=replayed,
            title=title_all,
            thick_lines=False,
            lang=args.lang,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
            out_path=out_dir / f"pk_lines_all_{safe_tgt}_{run_tag}.png",
        )

    if args.top_n > 0:
        top = replayed[: args.top_n]
        _render_and_save(
            all_ks=all_ks,
            p_theory=p_theory,
            series=top,
            title=title_top,
            thick_lines=True,
            lang=args.lang,
            fig_width=args.fig_width,
            fig_height=args.fig_height * 0.92,
            dpi=args.dpi,
            out_path=out_dir / f"pk_lines_top{args.top_n}_{safe_tgt}_{run_tag}.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
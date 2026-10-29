#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_logs.py

Парсит текстовые логи SIMPLE-солвера (scalar / vectorized / GPU) в табличные
данные (CSV) и строит графики сравнения.

Примеры:
    python analyze_logs.py --logs ./logs --out ./parsed --plots ./plots
    python analyze_logs.py --logs ./logs --case k --no-plot
    python analyze_logs.py --logs ./logs --excel
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Регулярные выражения (терпимые к небольшим различиям между версиями логов)
# ---------------------------------------------------------------------------

RE_HEADER = re.compile(
    r"^#\s*version=(?P<version>\S+)\s+solver=(?P<solver>\S+)\s+case=(?P<case>\S+)"
    r"\s+NX=(?P<nx>\d+)\s+NY=(?P<ny>\d+)\s+attempt=(?P<attempt>\d+)"
)
RE_STARTED  = re.compile(r"^#\s*Started:\s*(?P<ts>\S+)")
RE_FINISHED = re.compile(r"^#\s*Finished:\s*(?P<ts>\S+)")
RE_ELAPSED  = re.compile(r"\(elapsed\s+(?P<elapsed>[\d.]+)\s*s\)")

RE_GRID = re.compile(
    r"hx=(?P<hx>[-\d.eE+]+)\s+hy=(?P<hy>[-\d.eE+]+)\s+"
    r"DT=(?P<dt>[-\d.eE+]+)\s+LAMBDA=(?P<lam>[-\d.eE+]+)"
)
RE_PSOMEGA = re.compile(r"PS_OMEGA=(?P<omega>[-\d.eE+]+)")
RE_BATCH   = re.compile(r"\bbatch=(?P<batch>\d+)")

# строка истории: [  100] |duv|=.. |div|=.. t=..s ps_avg=.. [decay=.. ETA=..]
RE_ITER = re.compile(
    r"^\[\s*(?P<it>\d+)\]\s+"
    r"\|duv\|=(?P<duv>\S+)\s+"
    r"\|div\|=(?P<div>\S+)"
    r"(?:\s+t=\s*(?P<t>[\d.]+)s)?"
    r"(?:\s+ps_avg=(?P<ps>[\d.]+))?"
)
RE_DECAY = re.compile(r"decay=(?P<decay>[\d.]+)x/100it")
RE_ETA   = re.compile(r"ETA=(?P<eta>\d+)it")

RE_CONVERGED = re.compile(
    r"^Converged at iter\s+(?P<it>\d+),\s+\|duv_rel\|\s*=\s*(?P<duv>\S+)"
)
RE_SOR_WARN  = re.compile(r"SOR hit PS_MAX_ITER=(?P<maxit>\d+) in (?P<hits>\d+)/(?P<total>\d+)")
RE_TIME      = re.compile(r"^Time:\s+(?P<t>[\d.]+)s")
RE_ERR_U     = re.compile(r"^Error U:\s+L2=(?P<l2>\S+)\s+Linf=(?P<inf>\S+)")
RE_ERR_V     = re.compile(r"^Error V:\s+L2=(?P<l2>\S+)\s+Linf=(?P<inf>\S+)")
RE_MASS      = re.compile(r"^Mass divergence:\s+(?P<mass>\S+)")

RE_PROF = re.compile(
    r"^(?P<key>momentum|sor|correction|bcs|residual|copy_old|download)"
    r"\s*:\s*(?P<val>[\d.]+)$"
)
RE_SOR_SYNCS = re.compile(r"^SOR syncs\s*:\s*(?P<val>\d+)$")

RE_SUMMARY = re.compile(
    r"^\s{2,}(?P<key>iterations|time|err_u_l2|err_u_inf|err_v_l2|err_v_inf|"
    r"mass|ps_cap_hits|converged|outcome):\s*(?P<val>\S+)\s*$"
)


# ---------------------------------------------------------------------------
# Разбор одного лога
# ---------------------------------------------------------------------------

def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return None


def parse_log(path: Path):
    meta = {
        "run_id": path.stem,
        "file": path.name,
        "path": str(path),
        "version": None, "solver": None, "case": None,
        "nx": None, "ny": None, "attempt": None,
        "started": None, "finished": None, "elapsed_reported": None,
        "hx": None, "hy": None, "dt": None, "lam": None,
        "ps_omega": None, "batch": None,
        "time_header": None,
        "iterations": None, "time": None,
        "err_u_l2": None, "err_u_inf": None,
        "err_v_l2": None, "err_v_inf": None,
        "mass": None, "ps_cap_hits": None,
        "converged": None, "outcome": None,
        "conv_iter": None, "conv_duv": None,
        "sor_warn_hits": None, "sor_warn_total": None, "sor_syncs": None,
        "prof_momentum": None, "prof_sor": None, "prof_correction": None,
        "prof_bcs": None, "prof_residual": None, "prof_copy_old": None,
        "prof_download": None,
        "n_history": 0,
    }
    history = []

    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.rstrip()

        # --- заголовок с метаданными ---
        m = RE_HEADER.search(line)
        if m:
            meta.update({
                "version": m.group("version"),
                "solver": m.group("solver"),
                "case": m.group("case"),
                "nx": int(m.group("nx")),
                "ny": int(m.group("ny")),
                "attempt": int(m.group("attempt")),
            })
            continue

        # --- комментарии (# Started / # Finished / # Log) ---
        if line.startswith("#"):
            m = RE_STARTED.search(line)
            if m:
                meta["started"] = m.group("ts")
                continue
            m = RE_FINISHED.search(line)
            if m:
                meta["finished"] = m.group("ts")
                e = RE_ELAPSED.search(line)
                if e:
                    meta["elapsed_reported"] = _to_float(e.group("elapsed"))
            continue

        # --- строка истории итераций ---
        m = RE_ITER.match(line)
        if m:
            dm = RE_DECAY.search(line)
            em = RE_ETA.search(line)
            history.append({
                "run_id": path.stem,
                "iter": int(m.group("it")),
                "duv": _to_float(m.group("duv")),
                "div": _to_float(m.group("div")),
                "t": _to_float(m.group("t")),
                "ps_avg": _to_float(m.group("ps")),
                "decay": _to_float(dm.group("decay")) if dm else None,
                "eta_it": int(em.group("eta")) if em else None,
            })
            continue

        # --- шаги сетки ---
        m = RE_GRID.search(line)
        if m:
            meta["hx"] = _to_float(m.group("hx"))
            meta["hy"] = _to_float(m.group("hy"))
            meta["dt"] = _to_float(m.group("dt"))
            meta["lam"] = _to_float(m.group("lam"))
            continue

        # --- параметры SOR ---
        if meta["ps_omega"] is None:
            m = RE_PSOMEGA.search(line)
            if m:
                meta["ps_omega"] = _to_float(m.group("omega"))
        if meta["batch"] is None:
            m = RE_BATCH.search(line)
            if m:
                meta["batch"] = int(m.group("batch"))

        # --- сходимость ---
        m = RE_CONVERGED.match(line)
        if m:
            meta["conv_iter"] = int(m.group("it"))
            meta["conv_duv"] = _to_float(m.group("duv"))
            continue

        # --- предупреждение SOR ---
        m = RE_SOR_WARN.search(line)
        if m:
            meta["sor_warn_hits"] = int(m.group("hits"))
            meta["sor_warn_total"] = int(m.group("total"))
            continue

        # --- Time: ... s ---
        m = RE_TIME.match(line)
        if m:
            meta["time_header"] = _to_float(m.group("t"))
            continue

        # --- ошибки ---
        m = RE_ERR_U.match(line)
        if m:
            meta["err_u_l2"] = _to_float(m.group("l2"))
            meta["err_u_inf"] = _to_float(m.group("inf"))
            continue

        m = RE_ERR_V.match(line)
        if m:
            meta["err_v_l2"] = _to_float(m.group("l2"))
            meta["err_v_inf"] = _to_float(m.group("inf"))
            continue

        m = RE_MASS.match(line)
        if m:
            meta["mass"] = _to_float(m.group("mass"))
            continue

        # --- профилирование (только GPU) ---
        stripped = line.strip()
        m = RE_PROF.match(stripped)
        if m:
            meta["prof_" + m.group("key")] = _to_float(m.group("val"))
            continue
        m = RE_SOR_SYNCS.match(stripped)
        if m:
            meta["sor_syncs"] = int(m.group("val"))
            continue

        # --- итоговый блок "iterations: ..." ---
        m = RE_SUMMARY.match(line)
        if m:
            key, val = m.group("key"), m.group("val")
            if key == "iterations":
                meta["iterations"] = int(val)
            elif key == "time":
                meta["time"] = _to_float(val)
            elif key == "ps_cap_hits":
                meta["ps_cap_hits"] = int(val)
            elif key == "converged":
                meta["converged"] = _to_bool(val)
            elif key == "outcome":
                meta["outcome"] = val
            else:
                meta[key] = _to_float(val)
            continue

    if meta["time"] is None:
        meta["time"] = meta["time_header"]
    meta["n_history"] = len(history)
    return meta, history


# ---------------------------------------------------------------------------
# Сводные таблицы
# ---------------------------------------------------------------------------

def select_representatives(runs: pd.DataFrame) -> pd.DataFrame:
    """Если на одну (case, solver, nx, ny) несколько попыток — оставить одну:
    сначала сходившуюся, из них самую быструю."""
    df = runs.copy()
    if df.empty:
        df["selected"] = []
        return df

    conv = df["converged"].apply(lambda v: bool(v) if pd.notna(v) else False)
    t = pd.to_numeric(df["time"], errors="coerce").fillna(np.inf)

    df = df.join(pd.DataFrame({"_conv": (~conv).astype(int), "_time": t}, index=df.index))
    keys = ["case", "solver", "nx", "ny"]
    idx = (df.sort_values(["_conv", "_time"], kind="mergesort")
             .drop_duplicates(subset=keys, keep="first").index)
    df["selected"] = df.index.isin(idx)
    return df.drop(columns=["_conv", "_time"])


def pivot(runs: pd.DataFrame, value: str) -> pd.DataFrame:
    if value not in runs.columns:
        return pd.DataFrame()
    sel = runs[runs["selected"]]
    if sel.empty:
        return pd.DataFrame()
    return (sel.pivot_table(index="nx", columns="solver", values=value, aggfunc="min")
               .sort_index())


def speedup_table(runs: pd.DataFrame) -> pd.DataFrame:
    pt = pivot(runs, "time")
    if pt.empty or "s" not in pt.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=pt.index)
    for solver in pt.columns:
        if solver == "s":
            continue
        out[solver] = pt["s"] / pt[solver]
    return out


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

SOLVER_LABEL = {"s": "scalar (CPU)", "v": "vectorized (NumPy)", "g": "GPU (OpenCL)"}
SOLVER_COLOR = {"s": "#1f77b4", "v": "#ff7f0e", "g": "#2ca02c"}
_SOLVER_PREF = ["s", "v", "g"]


def _order_solvers(solvers):
    return sorted(solvers, key=lambda s: (_SOLVER_PREF.index(s) if s in _SOLVER_PREF else 99, str(s)))


def _label(solver):
    return SOLVER_LABEL.get(solver, f"solver={solver}")


def _color(solver):
    return SOLVER_COLOR.get(solver, None)


def _plot_convergence(plt, runs, hist, value, ylabel, fname, outdir,
                      x="iter", xlabel="номер итерации", logy=True, hline=None, title=""):
    sel = runs[runs["selected"]]
    nxs = sorted(sel["nx"].dropna().unique())
    if sel.empty or hist.empty or not nxs:
        return

    ncol = min(3, len(nxs))
    nrow = int(np.ceil(len(nxs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.8 * nrow), squeeze=False)

    handles, labels = [], []
    for ax, nx in zip(axes.flat, nxs):
        sub = sel[sel["nx"] == nx]
        for solver in _order_solvers(sub["solver"].dropna().unique()):
            run = sub[sub["solver"] == solver]
            if run.empty:
                continue
            h = hist[hist["run_id"] == run.iloc[0]["run_id"]].dropna(subset=[value, x])
            h = h.sort_values(x)
            if h.empty:
                continue
            line, = ax.plot(h[x], h[value], lw=1.6, color=_color(solver), label=_label(solver))
            if _label(solver) not in labels:
                handles.append(line)
                labels.append(_label(solver))
        ax.set_title(f"NX = NY = {int(nx)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        if logy:
            ax.set_yscale("log")
        if hline is not None:
            ax.axhline(hline, color="k", ls="--", lw=0.8)

    for ax in axes.flat[len(nxs):]:
        ax.set_visible(False)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96 if title else 1.0))
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_vs_nx(plt, runs, value, ylabel, fname, outdir, loglog=True, title=""):
    if value not in runs.columns:
        return
    sel = runs[runs["selected"]].dropna(subset=[value])
    if sel.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for solver in _order_solvers(sel["solver"].dropna().unique()):
        sub = sel[sel["solver"] == solver].sort_values("nx")
        if sub.empty:
            continue
        ax.plot(sub["nx"], sub[value], marker="o", ms=6, lw=1.8,
                color=_color(solver), label=_label(solver))
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("NX = NY")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)


def _plot_speedup(plt, runs, fname, outdir):
    sp = speedup_table(runs)
    if sp.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for solver in sp.columns:
        ax.plot(sp.index, sp[solver], marker="s", ms=6, lw=1.8,
                color=_color(solver), label=f"{_label(solver)} / {_label('s')}")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("NX = NY")
    ax.set_ylabel("ускорение  T_scalar / T_solver")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)


def _plot_time_bars(plt, runs, fname, outdir):
    sel = runs[runs["selected"]].dropna(subset=["time"])
    if sel.empty:
        return
    nxs = sorted(sel["nx"].dropna().unique())
    solvers = _order_solvers(sel["solver"].dropna().unique())
    width = 0.8 / max(len(solvers), 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xpos = np.arange(len(nxs))
    for i, solver in enumerate(solvers):
        vals = []
        for nx in nxs:
            s = sel[(sel["nx"] == nx) & (sel["solver"] == solver)]["time"]
            vals.append(float(s.min()) if not s.empty else np.nan)
        ax.bar(xpos + i * width, vals, width, color=_color(solver), label=_label(solver))
    ax.set_xticks(xpos + width * (len(solvers) - 1) / 2.0)
    ax.set_xticklabels([str(int(nx)) for nx in nxs])
    ax.set_yscale("log")
    ax.set_xlabel("NX = NY")
    ax.set_ylabel("полное время, с (лог. шкала)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)


def _plot_profile(plt, runs, fname, outdir):
    prof_cols = ["prof_momentum", "prof_sor", "prof_correction", "prof_bcs",
                 "prof_residual", "prof_copy_old", "prof_download"]
    sel = runs[runs["selected"]]
    if sel.empty:
        return
    sub = sel[sel[prof_cols].notna().any(axis=1)]
    if sub.empty:
        return

    labels = [f"{_label(s.solver)} NX={int(s.nx)}" for s in sub.itertuples()]
    xpos = np.arange(len(sub))
    bottoms = np.zeros(len(sub))

    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(sub) + 3), 4.8))
    for col in prof_cols:
        vals = sub[col].fillna(0.0).to_numpy(dtype=float)
        if not np.any(vals):
            continue
        ax.bar(xpos, vals, bottom=bottoms, label=col.replace("prof_", ""))
        bottoms += vals
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("время, с")
    ax.set_title("Профиль GPU-солвера по стадиям")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)


def make_plots(runs: pd.DataFrame, hist: pd.DataFrame, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
        "figure.dpi": 110, "savefig.dpi": 160,
    })

    _plot_convergence(plt, runs, hist, "duv", r"$|duv|$", "conv_duv.png", outdir,
                      logy=True, hline=1e-3, title="Сходимость по |duv|")
    _plot_convergence(plt, runs, hist, "div", r"$|div|$", "conv_div.png", outdir,
                      logy=True, title="Дивергенция по итерациям")
    _plot_convergence(plt, runs, hist, "duv", r"$|duv|$", "conv_duv_time.png", outdir,
                      x="t", xlabel="время, с", logy=True, hline=1e-3,
                      title="Сходимость по |duv| от времени")

    _plot_metric_vs_nx(plt, runs, "time", "полное время, с", "time_vs_nx.png", outdir,
                       title="Время решения vs размер сетки")
    _plot_metric_vs_nx(plt, runs, "iterations", "число итераций", "iters_vs_nx.png", outdir,
                       title="Число итераций vs размер сетки")
    _plot_metric_vs_nx(plt, runs, "time_per_iter", "время на итерацию, с",
                       "time_per_iter_vs_nx.png", outdir,
                       title="Стоимость одной итерации")
    _plot_metric_vs_nx(plt, runs, "err_u_l2", r"$L_2$ ошибка U", "err_u_l2_vs_nx.png", outdir,
                       title="Точность vs размер сетки")
    _plot_speedup(plt, runs, "speedup.png", outdir)
    _plot_time_bars(plt, runs, "time_bars.png", outdir)
    _plot_profile(plt, runs, "profile.png", outdir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=Path("logs"),
                    help="каталог с логами (или один файл)")
    ap.add_argument("--pattern", default="*.txt",
                    help="маска файлов при рекурсивном поиске")
    ap.add_argument("--out", type=Path, default=Path("parsed"),
                    help="куда сохранять CSV")
    ap.add_argument("--plots", type=Path, default=Path("plots"),
                    help="куда сохранять графики")
    ap.add_argument("--case", default=None, help="оставить только этот case (k, c, ...)")
    ap.add_argument("--no-plot", action="store_true", help="не строить графики")
    ap.add_argument("--excel", action="store_true",
                    help="дополнительно сохранить всё в один .xlsx")
    args = ap.parse_args(argv)

    if args.logs.is_file():
        files = [args.logs]
    elif args.logs.is_dir():
        files = sorted(args.logs.rglob(args.pattern))
    else:
        sys.exit(f"Не найден путь с логами: {args.logs}")
    if not files:
        sys.exit(f"В {args.logs} не найдено файлов по маске {args.pattern!r}")

    metas, hists = [], []
    for p in files:
        try:
            meta, hist = parse_log(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] не удалось разобрать {p.name}: {exc}", file=sys.stderr)
            continue
        metas.append(meta)
        hists.extend(hist)
    if not metas:
        sys.exit("Ни один лог не разобран.")

    run_columns = list(metas[0].keys())
    runs = pd.DataFrame(metas)[run_columns]
    runs["time_per_iter"] = runs["time"] / runs["iterations"]
    runs = runs.sort_values(["case", "solver", "nx", "attempt"],
                            na_position="last").reset_index(drop=True)

    hist = pd.DataFrame(hists)
    if hist.empty:
        hist = pd.DataFrame(columns=["run_id", "iter", "duv", "div", "t",
                                     "ps_avg", "decay", "eta_it"])
    else:
        hist = hist.sort_values(["run_id", "iter"]).reset_index(drop=True)

    if args.case:
        runs = runs[runs["case"] == args.case].copy()
        if runs.empty:
            sys.exit(f"Нет запусков для case={args.case!r}")
        hist = hist[hist["run_id"].isin(runs["run_id"])].copy()

    runs = select_representatives(runs)

    # ---------------- сохранение данных ----------------
    args.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out / "runs.csv", index=False, float_format="%.6g")

    hist_full = hist.merge(
        runs[["run_id", "solver", "case", "nx", "ny", "attempt", "selected"]],
        on="run_id", how="left")
    hist_full.to_csv(args.out / "history.csv", index=False, float_format="%.6g")

    pivots = {}
    for name in ("time", "iterations", "err_u_l2", "err_v_l2", "mass", "time_per_iter"):
        pt = pivot(runs, name)
        if not pt.empty:
            pivots[name] = pt
            pt.to_csv(args.out / f"pivot_{name}.csv", float_format="%.6g")

    sp = speedup_table(runs)
    if not sp.empty:
        sp.to_csv(args.out / "pivot_speedup.csv", float_format="%.4f")

    if args.excel:
        try:
            with pd.ExcelWriter(args.out / "summary.xlsx", engine="openpyxl") as xw:
                runs.to_excel(xw, sheet_name="runs", index=False)
                hist_full.to_excel(xw, sheet_name="history", index=False)
                for name, pt in pivots.items():
                    pt.to_excel(xw, sheet_name=f"pivot_{name}"[:31])
                if not sp.empty:
                    sp.to_excel(xw, sheet_name="speedup")
        except ImportError:
            print("[WARN] openpyxl не установлен — xlsx не сохранён", file=sys.stderr)

    # ---------------- консольная сводка ----------------
    show = [c for c in ["run_id", "solver", "nx", "iterations", "time",
                        "time_per_iter", "err_u_l2", "converged", "selected"]
            if c in runs.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(runs[show].to_string(index=False))

    # ---------------- графики ----------------
    if not args.no_plot:
        args.plots.mkdir(parents=True, exist_ok=True)
        make_plots(runs, hist_full, args.plots)
        print(f"\nГрафики: {args.plots.resolve()}")
    print(f"Данные:  {args.out.resolve()}")


if __name__ == "__main__":
    main()
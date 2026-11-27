"""
Build the report tables and figure from finished transfer runs.

  python experiments/make_report.py --tags final_ungated strict ld_control

Reads results/transfer/<tag>/{summary.json, per_question.jsonl, applicability.jsonl}
and writes results/transfer/report/report.md and transfer_gain.png.

Confidence intervals are paired bootstrap intervals (5,000 resamples of the
questions) on accuracy(condition) - accuracy(no_skills). The figure is skipped,
with a note, if matplotlib is not installed; the tables are always written.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results" / "transfer"

BLUE, INK, INK2, GRID = "#257de0", "#191919", "#666666", "#e4e4e4"


def load_run(tag):
    run_dir = RESULTS / tag
    if not (run_dir / "summary.json").exists():
        return None
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    per_q = {}
    for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        per_q.setdefault(row["condition"], {})[row["id"]] = row
    applic = []
    path = run_dir / "applicability.jsonl"
    if path.exists():
        applic = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"tag": tag, "summary": summary, "per_q": per_q, "applic": applic}


def paired_bootstrap(base, other, ids, n_boot=5000, seed=0):
    """95% CI for the accuracy difference, resampling questions."""
    d = np.array([int(other[i]["correct"]) - int(base[i]["correct"]) for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(run):
    base = run["per_q"]["no_skills"]
    ids = sorted(base)
    rows = {}
    for cond, r in run["summary"]["conditions"].items():
        if cond == "no_skills":
            continue
        lo, hi = paired_bootstrap(base, run["per_q"][cond], ids)
        rows[cond] = {**r, "ci": (lo, hi)}
    return rows


def applicability_notes(run):
    rows, base, app = run["applic"], run["per_q"]["no_skills"], run["per_q"].get("applicability")
    if not rows or app is None:
        return []
    n = len(rows)
    kept = {r["id"] for r in rows if r["kept"]}
    pairs = sum(len(r["judged"]) for r in rows)
    yes = sum(v["applicable"] for r in rows for v in r["judged"].values())
    fails = sum(v["parse_failed"] for r in rows for v in r["judged"].values())
    rep = sum(1 for i in kept if not base[i]["correct"] and app[i]["correct"])
    brk = sum(1 for i in kept if base[i]["correct"] and not app[i]["correct"])
    per_skill = {}
    for r in rows:
        for sid, v in r["judged"].items():
            per_skill[sid] = per_skill.get(sid, 0) + int(v["applicable"])
    return [
        f"- Judge said YES to {yes} of {pairs} (question, skill) pairs ({yes / pairs:.1%}); unparseable replies: {fails}.",
        f"- {len(kept)} of {n} questions ({len(kept) / n:.0%}) received at least one skill; "
        f"on those, accuracy changed by {rep} repaired and {brk} broken (the other questions have the no-skills prompt).",
        "- YES count per skill: " + ", ".join(f"{k} {v}" for k, v in sorted(per_skill.items())) + ".",
    ]


def write_markdown(runs, analyses, path):
    out = ["# Transfer results", "",
           "Accuracy change is relative to the no-skills run on the same questions. CI = 95% paired bootstrap "
           "interval; p = exact sign test on repaired vs broken. A CI that includes 0 means the change cannot be "
           "told apart from no change.", ""]
    for run, rows in zip(runs, analyses):
        s = run["summary"]
        base_acc = run["summary"]["conditions"]["no_skills"]["accuracy"]
        out += [f"## {run['tag']}: {s['target']}, {s['n_skills']} skills, k={s['k']}", "",
                f"Library: `{s['library']}`   Model: {s['model']}   no_skills accuracy: {base_acc:.3f}", "",
                "| Condition | Accuracy | Change (pts) | 95% CI (pts) | Repaired / broken | p | Skills shown | Model calls | Prompts reused |",
                "|---|---|---|---|---|---|---|---|---|"]
        for cond, r in rows.items():
            c = r.get("cost", {})
            out.append(f"| {cond} | {r['accuracy']:.3f} | {r['transfer_gain'] * 100:+.1f} | "
                       f"[{r['ci'][0] * 100:+.1f}, {r['ci'][1] * 100:+.1f}] | {r['repaired']} / {r['broken']} | "
                       f"{r['p_value']:.2f} | {r['mean_skills_shown']:.2f} | {c.get('calls', '')}{(' (+' + str(c['judge_calls']) + ' judge)') if c.get('judge_calls') else ''} | {c.get('prompts_reused', '')} |")
        notes = applicability_notes(run)
        if notes:
            out += ["", "Applicability judge:", *notes]
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def draw_figure(runs, analyses, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the figure (tables were written).")
        return False

    n = len(runs)
    lo_all = min(r["ci"][0] for rows in analyses for r in rows.values()) * 100
    hi_all = max(r["ci"][1] for rows in analyses for r in rows.values()) * 100
    xlim = (min(-2, lo_all - 3), max(2, hi_all) + 9)      # right margin holds the value column
    label_x = xlim[1] - 0.5

    height = 1.0 + 0.55 * max(len(rows) for rows in analyses) + 1.0
    ncols = min(4, n)                       # wrap into rows so the figure fits a page
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, height * nrows), sharex=True, squeeze=False)
    axes = list(axes.flatten())
    for extra in axes[n:]:
        extra.set_visible(False)

    for ax, run, rows in zip(axes, runs, analyses):
        conds = list(rows)
        ys = list(range(len(conds)))[::-1]
        for y, cond in zip(ys, conds):
            r = rows[cond]
            gain, lo, hi = r["transfer_gain"] * 100, r["ci"][0] * 100, r["ci"][1] * 100
            ax.plot([lo, hi], [y, y], color=BLUE, lw=1.5, solid_capstyle="round", zorder=2)
            excludes_zero = lo > 0 or hi < 0
            ax.plot(gain, y, "o", ms=8, color=BLUE, mfc=BLUE if excludes_zero else "white", mew=1.5, zorder=3)
            ax.text(label_x, y, f"{gain:+.1f}", va="center", ha="right", fontsize=9.5, color=INK)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([c.replace("_", " ") for c in conds], color=INK, fontsize=10)
        ax.set_ylim(-0.6, len(conds) - 0.4)
        ax.set_xlim(*xlim)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK2, labelsize=9)
        ax.tick_params(axis="y", length=0)
        s = run["summary"]
        target = "TSO" if "tracking" in s["target"] else "LD (control)"
        judge = ", 7B judge" if "7B" in str(s.get("judge_model", "")) and "same as solver" not in str(s.get("judge_model", "")) else ""
        ax.set_title(f"{run['tag'].replace('_', ' ')}\n{target}, {s['n_skills']} skills{judge}",
                     loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("Accuracy change vs no skills (points)", color=INK2, fontsize=9)

    fig.text(0.01, 0.01, "Dot = change in accuracy; line = 95% paired bootstrap CI. Filled dot: CI excludes 0. "
             "Hollow dot: CI includes 0 (cannot be told apart from no change).", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05 / nrows * 1.5, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=["final_ungated", "strict", "ld_control"])
    parser.add_argument("--out", default=str(RESULTS / "report"))
    args = parser.parse_args()

    runs = []
    for tag in args.tags:
        run = load_run(tag)
        if run is None:
            print(f"skipping '{tag}': no summary.json yet")
        else:
            runs.append(run)
    if not runs:
        sys.exit("no finished runs found")

    analyses = [analyze(r) for r in runs]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(runs, analyses, out_dir / "report.md")
    drew = draw_figure(runs, analyses, out_dir / "transfer_gain.png")
    print(f"wrote {out_dir / 'report.md'}" + (f" and {out_dir / 'transfer_gain.png'}" if drew else ""))


if __name__ == "__main__":
    main()

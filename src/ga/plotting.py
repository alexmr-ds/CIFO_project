"""
Reusable matplotlib figures for GA experiment analysis.

These functions capture the visualization patterns that repeat across
multiple notebook sections, so each cell only needs to pass data and labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .parallel import TrialSummary


def _twin(ax: plt.Axes, x, y, **style) -> None:
    """Add a faint RMSE reference curve on a secondary y-axis."""
    ax2 = ax.twinx()
    ax2.plot(x, y, **style)
    ax2.set_ylabel("RMSE", fontsize=7, color="grey")
    ax2.tick_params(axis="y", labelsize=7, colors="grey")


# ---------------------------------------------------------------------------
# Grid-search results panel
# ---------------------------------------------------------------------------

def plot_grid_search_results(
    results: dict[Any, Any],
    param_values: list[Any],
    param_name: str,
    x_tick_labels: list[str],
    label_fmt: str,
    baseline_fitness: float,
    suptitle: str,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Two-panel grid-search figure: convergence curves (left) + bar chart (right).

    Args:
        results:          Dict mapping each param value to a TrialSummary.
        param_values:     Ordered list of parameter values (controls plot order).
        param_name:       Human-readable name for axis labels, e.g. "Population size".
        x_tick_labels:    X-axis tick strings for the bar chart.
        label_fmt:        Format string for convergence legend, e.g. "pop={v}".
        baseline_fitness: Reference RMSE drawn as a dashed line on both panels.
        suptitle:         Main figure title.

    Returns:
        (fig, axes)
    """
    colors_map = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(suptitle, fontweight="bold")

    for i, v in enumerate(param_values):
        s     = results[v]
        h     = np.array(s.mean_history)
        std   = np.array(s.std_history)
        label = f"{label_fmt.format(v=v)}  ({s.mean_fitness:.4f} ± {s.std_fitness:.4f})"
        axes[0].plot(h, color=colors_map[i], linewidth=1.5, label=label)
        axes[0].fill_between(range(len(h)), h - std, h + std,
                             alpha=0.15, color=colors_map[i])

    axes[0].axhline(baseline_fitness, color="black", linestyle="--", linewidth=1.2,
                    label=f"Baseline mean RMSE={baseline_fitness:.4f}")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best fitness (RMSE)")
    axes[0].set_title("Mean Convergence Curves (shaded = ± 1 std)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    means = [results[v].mean_fitness for v in param_values]
    stds  = [results[v].std_fitness  for v in param_values]

    bars = axes[1].bar(
        x_tick_labels,
        means,
        yerr=stds,
        capsize=5,
        color=[colors_map[i] for i in range(len(param_values))],
        edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "black"},
    )
    axes[1].axhline(baseline_fitness, color="black", linestyle="--", linewidth=1.2,
                    label=f"Baseline mean RMSE={baseline_fitness:.4f}")
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel("Mean final RMSE")
    axes[1].set_title(f"Mean Final RMSE ± std by {param_name}")
    axes[1].legend(fontsize=8)
    for bar, mean, std in zip(bars, means, stds):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.001,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Diversity comparison panel
# ---------------------------------------------------------------------------

def plot_diversity_comparison(
    logs_dict: dict[str, dict],
    suptitle: str,
) -> plt.Figure:
    """
    3×2 diversity-comparison panel for multiple GA configurations.

    Row 0 (full width) — RMSE reference curves.
    Row 1–2            — genotypic variance, genotypic entropy,
                         phenotypic variance, phenotypic entropy,
                         each with a faint RMSE twin axis.

    The entry keyed "Baseline" is plotted in black dashed; all others in tab10.

    Args:
        logs_dict: Ordered dict mapping label → log dict from run_diversity_trial.
        suptitle:  Main figure title.

    Returns:
        Figure — call plt.show() after this function.
    """
    colors_map = plt.cm.tab10.colors
    RMSE_KW    = dict(color="black", lw=0.9, alpha=0.35, linestyle=":")

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(suptitle, fontweight="bold", fontsize=12)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38,
                           height_ratios=[1, 1.2, 1.2])

    def _style(label: str, ci: int) -> tuple[str, str]:
        if label == "Baseline":
            return "black", "--"
        return colors_map[ci - 1], "-"

    ax0 = fig.add_subplot(gs[0, :])
    for ci, (label, log) in enumerate(logs_dict.items()):
        col, ls = _style(label, ci)
        ax0.plot(log["gen"], log["best_fitness"], color=col, lw=1.8, ls=ls, label=label)
    ax0.set_title("RMSE — reference", fontweight="bold")
    ax0.set_xlabel("Generation")
    ax0.set_ylabel("Best RMSE")
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    panels = [
        (gs[1, 0], "geno_var",      "gen",       "Genotypic Variance",        "Mean norm. variance"),
        (gs[1, 1], "geno_entropy",  "gen",       "Genotypic Entropy (bits)",  "Mean entropy (bits)"),
        (gs[2, 0], "pheno_var",     "pheno_gen", "Phenotypic Variance",       "Mean pixel variance"),
        (gs[2, 1], "pheno_entropy", "pheno_gen", "Phenotypic Entropy (bits)", "Mean entropy (bits)"),
    ]

    _first_log = next(iter(logs_dict.values()))
    rmse_gen   = np.array(_first_log["gen"])
    rmse_vals  = np.array(_first_log["best_fitness"])

    for spec, ykey, xkey, title, ylabel in panels:
        ax = fig.add_subplot(spec)
        for ci, (label, log) in enumerate(logs_dict.items()):
            col, ls = _style(label, ci)
            ax.plot(log[xkey], log[ykey], color=col, lw=1.8, ls=ls, label=label)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if xkey == "pheno_gen":
            pheno_x   = np.array(_first_log["pheno_gen"])
            twin_rmse = np.interp(pheno_x, rmse_gen, rmse_vals)
            _twin(ax, pheno_x, twin_rmse, **RMSE_KW)
        else:
            _twin(ax, rmse_gen, rmse_vals, **RMSE_KW)

    return fig


# ---------------------------------------------------------------------------
# Method comparison panel (baseline + N variants)
# ---------------------------------------------------------------------------

def plot_method_comparison(
    baseline: TrialSummary,
    variants: dict[str, TrialSummary],
    suptitle: str,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Two-panel comparison: baseline (black dashed) + N variants (tab10).

    Left panel  — convergence curves (mean ± 1 std band).
    Right panel — bar chart of mean final RMSE ± std.

    Args:
        baseline: Reference TrialSummary — drawn in black dashed.
        variants: Ordered dict mapping label → TrialSummary.
        suptitle: Main figure title.

    Returns:
        (fig, axes)
    """
    colors_map = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(suptitle, fontweight="bold")

    bl_h   = np.array(baseline.mean_history)
    bl_std = np.array(baseline.std_history)
    axes[0].plot(bl_h, color="black", lw=1.8, linestyle="--",
                 label=f"Baseline  ({baseline.mean_fitness:.4f}±{baseline.std_fitness:.4f})")
    axes[0].fill_between(range(len(bl_h)), bl_h - bl_std, bl_h + bl_std,
                         alpha=0.12, color="black")

    for ci, (label, summary) in enumerate(variants.items()):
        h   = np.array(summary.mean_history)
        std = np.array(summary.std_history)
        axes[0].plot(h, color=colors_map[ci], lw=1.8,
                     label=f"{label}  ({summary.mean_fitness:.4f}±{summary.std_fitness:.4f})")
        axes[0].fill_between(range(len(h)), h - std, h + std,
                             alpha=0.12, color=colors_map[ci])

    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best fitness (RMSE)")
    axes[0].set_title("Mean Convergence Curves (shaded = ± 1 std)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    all_labels = ["Baseline"] + list(variants.keys())
    all_means  = [baseline.mean_fitness] + [v.mean_fitness for v in variants.values()]
    all_stds   = [baseline.std_fitness]  + [v.std_fitness  for v in variants.values()]
    bar_colors = ["black"] + [colors_map[i] for i in range(len(variants))]

    bars = axes[1].bar(
        all_labels,
        all_means,
        yerr=all_stds,
        capsize=5,
        color=bar_colors,
        alpha=0.80,
        error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
    )
    for bar, mean, std in zip(bars, all_means, all_stds):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.001,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    axes[1].set_ylabel("Mean final RMSE")
    axes[1].set_title("Mean Final RMSE ± std")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Cumulative convergence across all stages
# ---------------------------------------------------------------------------

def plot_cumulative_convergence(
    stages: list[tuple[str, Any]],
    suptitle: str = "Cumulative Method Comparison",
) -> plt.Figure:
    """
    Single-panel figure overlaying the mean convergence curve for each stage.

    Args:
        stages:   List of (label, TrialSummary) in display order.
        suptitle: Main figure title.

    Returns:
        Figure — call plt.show() after this function.
    """
    colors_map = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(suptitle, fontweight="bold")

    for i, (label, summary) in enumerate(stages):
        h     = np.array(summary.mean_history)
        color = colors_map[i % len(colors_map)]
        lw    = 2.2 if i == 0 else 1.6
        ls    = "--" if i == 0 else "-"
        ax.plot(h, color=color, lw=lw, ls=ls,
                label=f"{label}  (final={summary.mean_fitness:.4f})")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Mean Convergence Curve per Cumulative Stage  (dashed = baseline)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Printed summary tables
# ---------------------------------------------------------------------------

def print_grid_search_summary(
    results: dict,
    param_values: list,
    param_name: str,
    baseline_fitness: float,
) -> Any:
    """Print a summary table for a single-parameter grid search. Returns best value."""
    width = max(max(len(str(v)) for v in param_values), len(param_name))
    sep = "=" * 74
    print(sep)
    print(f"  {param_name} Grid Search — Summary")
    print(sep)
    print(f"  {param_name:<{width}}  {'Mean RMSE':>10}  {'± Std':>8}  {'vs Baseline':>12}  {'Min':>10}  {'Max':>10}")
    print(f"  {'-'*width}  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}")
    best = min(param_values, key=lambda v: results[v].mean_fitness)
    for v in param_values:
        s      = results[v]
        delta  = s.mean_fitness - baseline_fitness
        sign   = "+" if delta > 0 else ""
        marker = "  <-- best" if v == best else ""
        print(f"  {str(v):<{width}}  {s.mean_fitness:>10.6f}  {s.std_fitness:>8.6f}"
              f"  {sign}{delta:>11.6f}  {s.min_fitness:>10.6f}  {s.max_fitness:>10.6f}{marker}")
    print(sep)
    print(f"\n  Best {param_name.lower()}: {best}"
          f"  (mean RMSE = {results[best].mean_fitness:.6f} ± {results[best].std_fitness:.6f})")
    return best


def print_method_comparison_summary(
    baseline: Any,
    variants: dict,
    baseline_label: str = "Baseline",
) -> Any:
    """Print a summary table comparing a baseline against variants. Returns best label."""
    all_items  = {baseline_label: baseline, **variants}
    best_label = min(all_items, key=lambda k: all_items[k].mean_fitness)
    bl_mean    = baseline.mean_fitness
    sep = "=" * 80
    print(sep)
    print("  Method Comparison — Summary")
    print(sep)
    print(f"  {'Config':<25}  {'Mean RMSE':>10}  {'± Std':>8}  {'vs Baseline':>12}  {'Best trial':>10}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*10}")
    for label, s in all_items.items():
        if label == baseline_label:
            diff = "—"
        else:
            delta = s.mean_fitness - bl_mean
            diff  = f"{'+' if delta > 0 else ''}{delta:.6f}"
        marker = "  <-- best" if label == best_label else ""
        print(f"  {label:<25}  {s.mean_fitness:>10.6f}  {s.std_fitness:>8.6f}"
              f"  {diff:>12}  {s.min_fitness:>10.6f}{marker}")
    print(sep)
    return best_label


def plot_single_diversity(
    log: dict,
    pop: int,
    gens: int,
    pheno_interval: int,
    pheno_scale: float,
) -> plt.Figure:
    """
    3×2 diversity figure for a single GA run.

    Row 0 (full width) — RMSE reference curve with fill.
    Row 1–2            — genotypic variance, genotypic entropy,
                         phenotypic variance, phenotypic entropy,
                         each with a faint RMSE twin axis.

    Args:
        log:            Dict returned by run_diversity_trial.
        pop:            Population size (used in suptitle).
        gens:           Number of generations (used in suptitle).
        pheno_interval: Phenotypic sampling interval (used in suptitle).
        pheno_scale:    Phenotypic render scale (used in suptitle).

    Returns:
        Figure — call plt.show() after this function.
    """
    gen_arr    = np.array(log["gen"])
    rmse       = np.array(log["best_fitness"])
    geno_var   = np.array(log["geno_var"])
    geno_H     = np.array(log["geno_entropy"])
    pheno_gens = np.array(log["pheno_gen"])
    pheno_var  = np.array(log["pheno_var"])
    pheno_H    = np.array(log["pheno_entropy"])
    rmse_pheno = np.interp(pheno_gens, gen_arr, rmse)

    RMSE_KW = dict(color="black", lw=1.0, alpha=0.4, linestyle="--")

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"Population Diversity Over Generations\n"
        f"(pop={pop}, gen={gens}, pheno sampled every {pheno_interval} gens @ {pheno_scale:.0%} scale)",
        fontweight="bold", fontsize=12,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38,
                           height_ratios=[1, 1.2, 1.2])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(gen_arr, rmse, color="black", lw=2)
    ax0.fill_between(gen_arr, rmse, rmse.max(), alpha=0.07, color="black")
    ax0.set_title("Fitness (RMSE) — reference curve", fontweight="bold")
    ax0.set_xlabel("Generation")
    ax0.set_ylabel("Best RMSE")
    ax0.grid(True, alpha=0.3)

    panels = [
        (gs[1, 0], gen_arr,    geno_var,  rmse,       "#1976D2", "Genotypic Variance",            "Mean normalised variance",      False),
        (gs[1, 1], gen_arr,    geno_H,    rmse,       "#388E3C", "Genotypic Entropy (bits)",       "Mean Shannon entropy (bits)",   False),
        (gs[2, 0], pheno_gens, pheno_var, rmse_pheno, "#E53935", "Phenotypic Variance (rendered)", "Mean pixel-wise variance",      True),
        (gs[2, 1], pheno_gens, pheno_H,   rmse_pheno, "#7B1FA2", "Phenotypic Entropy (rendered)",  "Mean pixel-wise entropy (bits)", True),
    ]
    for spec, x, y, rmse_ref, color, title, ylabel, markers in panels:
        ax = fig.add_subplot(spec)
        kw = {"marker": "o", "ms": 4} if markers else {}
        ax.plot(x, y, color=color, lw=1.8, **kw)
        ax.fill_between(x, 0, y, alpha=0.15, color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        _twin(ax, x, rmse_ref, **RMSE_KW)

    return fig


def print_single_diversity_summary(log: dict) -> None:
    """Print start/end/Δ summary table and auto-diagnosis for a single diversity log."""
    rmse      = np.array(log["best_fitness"])
    geno_var  = np.array(log["geno_var"])
    geno_H    = np.array(log["geno_entropy"])
    pheno_var = np.array(log["pheno_var"])
    pheno_H   = np.array(log["pheno_entropy"])

    print("\nDiversity Summary")
    print("=" * 68)
    print(f"  {'Metric':<25}  {'Start':>10}  {'End':>10}  {'Δ':>10}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label, arr, fmt in [
        ("RMSE",                     rmse,     ".5f"),
        ("Genotypic variance",       geno_var, ".5f"),
        ("Genotypic entropy (bits)", geno_H,   ".3f"),
        ("Phenotypic variance",      pheno_var, ".5f"),
        ("Phenotypic entropy (bits)", pheno_H, ".3f"),
    ]:
        print(f"  {label:<25}  {arr[0]:>10{fmt}}  {arr[-1]:>10{fmt}}  {arr[-1]-arr[0]:>+10{fmt}}")
    print("=" * 68)

    gv_drop  = (geno_var[0] - geno_var[-1]) / max(geno_var[0], 1e-9)
    pv_drop  = (pheno_var[0] - pheno_var[-1]) / max(pheno_var[0], 1e-9)
    rmse_imp = (rmse[0] - rmse[-1]) / max(rmse[0], 1e-9)
    print()
    if gv_drop > 0.5 and rmse_imp < 0.1:
        print("  ⚠  Strong genotypic convergence with little RMSE gain")
        print("     → Premature convergence. Consider: higher mutation rate, larger population,")
        print("       or tournament size reduction.")
    elif gv_drop < 0.15 and rmse_imp < 0.1:
        print("  ⚠  Diversity maintained but RMSE barely improved")
        print("     → Operators are not exploiting gene combinations effectively.")
        print("       Consider: stronger elitism, a different crossover operator, or more generations.")
    elif pv_drop > 0.5 and gv_drop < 0.3:
        print("  ⚠  Phenotypic convergence faster than genotypic")
        print("     → Many genotypes map to similar images (gene redundancy / epistasis).")
    else:
        print("  ✓  Diversity and RMSE are evolving as expected.")
        print(f"     RMSE improved {rmse_imp*100:.1f}%  |  "
              f"Geno variance dropped {gv_drop*100:.1f}%  |  "
              f"Pheno variance dropped {pv_drop*100:.1f}%")


def print_diversity_summary(logs_dict: dict, title: str = "Diversity Summary") -> None:
    """Print a percentage-drop summary table for multiple diversity logs."""
    sep = "=" * 76
    print(f"\n{title}")
    print(sep)
    print(f"  {'Config':<22}  {'Geno Var drop':>14}  {'Geno H drop':>12}"
          f"  {'Pheno Var drop':>15}  {'RMSE drop':>10}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*12}  {'-'*15}  {'-'*10}")
    for label, log in logs_dict.items():
        def _drop(key):
            a = np.array(log[key])
            return (a[0] - a[-1]) / max(a[0], 1e-9) * 100
        print(f"  {label:<22}  {_drop('geno_var'):>13.1f}%  {_drop('geno_entropy'):>11.1f}%"
              f"  {_drop('pheno_var'):>14.1f}%  {_drop('best_fitness'):>9.1f}%")
    print(sep)

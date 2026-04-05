"""
demand_side_analysis.py
=======================
Demand-Side Trend Validation Pipeline
Reads candidate_trends.json from the supply-side pipeline, fetches Google
Trends search volume for each trend keyword, and validates whether consumer
interest actually moved during Fashion Month.

Pipeline:
    1. Load candidate trends from supply_side_model.py output
    2. Map each trend label to one or more search queries
    3. Fetch weekly Google Trends data (pytrends)
    4. Compute Fashion Month momentum score
    5. Apply threshold: only confirmed trends pass
    6. Output validated_trends.json + visualisation

Install:  pip install pytrends pandas numpy matplotlib seaborn
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")        # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pytrends.request import TrendReq   # pip install pytrends
from pytrends.exceptions import TooManyRequestsError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DemandConfig:
    # Input: output of supply_side_model.py
    candidate_trends_path: str = "outputs/supply_side/candidate_trends.json"

    # Output directory
    output_dir: str = "outputs/demand_side"

    # ── Fashion Month windows ─────────────────────────────────────────────────
    # Adjust these to match the season you are analysing.
    # AW26 Fashion Month ran February–March 2026.
    fashion_month_start: str = "2026-02-01"
    fashion_month_end:   str = "2026-03-31"

    # Full analysis window (should comfortably surround Fashion Month)
    analysis_start: str = "2025-11-01"
    analysis_end:   str = "2026-04-30"

    # ── Momentum scoring ──────────────────────────────────────────────────────
    # A trend must reach this momentum score to be validated.
    # Score = mean(during_fashion_month) / mean(pre_fashion_month)
    # e.g. 1.5 = 50% uplift during Fashion Month
    momentum_threshold: float = 1.5

    # ── pytrends settings ─────────────────────────────────────────────────────
    geo: str = ""          # "" = worldwide; "US", "GB", etc. for country-level
    gprop: str = ""        # "" = web search; "news", "images", "youtube"
    hl: str = "en-US"
    tz: int = 0            # UTC

    # Seconds to sleep between API calls (avoid rate limiting)
    request_delay: float = 3.0
    max_retries: int = 3

    # pytrends compares up to 5 keywords per batch; we use 1 per call for
    # cleaner absolute-ish indices, then normalise across all trends ourselves.
    timeframe: str = ""    # auto-built from analysis_start / analysis_end


# ─────────────────────────────────────────────────────────────────────────────
# Trend keyword mapping
# ─────────────────────────────────────────────────────────────────────────────

# Maps CLIP zero-shot labels (from supply_side_model.py) to Google search
# queries that real consumers actually type.
# Extend this dict to cover your full fashion_elements vocabulary.
LABEL_TO_QUERY: dict = {
    # Silhouettes
    "oversized silhouette":          "oversized fashion",
    "slim tailored silhouette":      "slim fit suit fashion",
    "voluminous silhouette":         "voluminous dress trend",
    "asymmetric hemline":            "asymmetric skirt",
    "structured shoulders":          "structured shoulder jacket",
    "dropped waist":                 "drop waist dress",
    # Fabrics
    "sheer organza fabric":          "sheer organza outfit",
    "leather or faux leather":       "leather fashion trend",
    "velvet fabric":                 "velvet clothing trend",
    "denim":                         "denim fashion trend",
    "knitwear":                      "knitwear fashion",
    "satin fabric":                  "satin outfit trend",
    "faux fur":                      "faux fur coat trend",
    "sequins and embellishment":     "sequin outfit trend",
    # Details
    "floral print":                  "floral print fashion",
    "animal print":                  "animal print fashion",
    "abstract graphic print":        "graphic print fashion",
    "cut-out detail":                "cutout fashion trend",
    "ruching and draping":           "ruched dress trend",
    "bow detail":                    "bow detail fashion",
    "cape overlay":                  "cape fashion trend",
    "corseted waist":                "corset fashion trend",
    "utilitarian pockets":           "utility fashion trend",
    # Colours
    "monochromatic all-black look":  "all black outfit trend",
    "bright primary colours":        "colour blocking fashion",
    "earth tones and neutrals":      "neutral tones fashion",
    "pastel palette":                "pastel fashion trend",
}


def label_to_query(label: str) -> str:
    """Return the best Google search query for a CLIP label."""
    return LABEL_TO_QUERY.get(label, label)   # fall back to label itself


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrendResult:
    cluster_id: int
    label: str
    search_query: str
    momentum_score: float          # during / pre ratio
    mean_pre:    float             # mean weekly interest before Fashion Month
    mean_during: float             # mean weekly interest during Fashion Month
    mean_post:   float             # mean weekly interest after Fashion Month
    peak_week: Optional[str]       # ISO date of peak search interest
    is_validated: bool             # passed momentum threshold
    supply_house_count: int
    supply_cross_house_score: float


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load candidates
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(cfg: DemandConfig) -> list:
    path = Path(cfg.candidate_trends_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate trends file not found: {path}\n"
            "Run supply_side_model.py first."
        )
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} candidate trends from {path}.")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Google Trends fetching
# ─────────────────────────────────────────────────────────────────────────────

def build_timeframe(cfg: DemandConfig) -> str:
    return f"{cfg.analysis_start} {cfg.analysis_end}"


def fetch_trends(
    query: str,
    cfg: DemandConfig,
    pytrends: TrendReq,
) -> Optional[pd.DataFrame]:
    """
    Fetch weekly search interest for a single query.
    Returns a DataFrame with columns [date, interest] or None on failure.
    """
    timeframe = build_timeframe(cfg)
    for attempt in range(1, cfg.max_retries + 1):
        try:
            pytrends.build_payload(
                kw_list=[query],
                cat=0,
                timeframe=timeframe,
                geo=cfg.geo,
                gprop=cfg.gprop,
            )
            df = pytrends.interest_over_time()
            if df.empty:
                logger.warning(f"No data returned for query: '{query}'")
                return None
            df = df.reset_index()[["date", query]].rename(
                columns={query: "interest"}
            )
            df["date"] = pd.to_datetime(df["date"])
            return df
        except TooManyRequestsError:
            wait = cfg.request_delay * (2 ** attempt)
            logger.warning(
                f"Rate limited fetching '{query}' "
                f"(attempt {attempt}/{cfg.max_retries}). "
                f"Waiting {wait:.0f}s ..."
            )
            time.sleep(wait)
        except Exception as exc:
            logger.error(f"Error fetching '{query}': {exc}")
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Momentum scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum(
    df: pd.DataFrame,
    cfg: DemandConfig,
) -> dict:
    """
    Split the time series into pre / during / post Fashion Month windows
    and compute a momentum score = mean_during / mean_pre.
    """
    fm_start = pd.Timestamp(cfg.fashion_month_start)
    fm_end   = pd.Timestamp(cfg.fashion_month_end)

    pre    = df[df["date"] < fm_start]["interest"]
    during = df[(df["date"] >= fm_start) & (df["date"] <= fm_end)]["interest"]
    post   = df[df["date"] > fm_end]["interest"]

    mean_pre    = float(pre.mean())    if len(pre)    > 0 else 0.0
    mean_during = float(during.mean()) if len(during) > 0 else 0.0
    mean_post   = float(post.mean())   if len(post)   > 0 else 0.0

    # Avoid division by zero: if pre is 0, score is 0 (no baseline to uplift from)
    momentum = (mean_during / mean_pre) if mean_pre > 0 else 0.0

    peak_week = None
    if not df.empty:
        peak_idx  = df["interest"].idxmax()
        peak_week = df.loc[peak_idx, "date"].strftime("%Y-%m-%d")

    return {
        "mean_pre":       round(mean_pre,    2),
        "mean_during":    round(mean_during, 2),
        "mean_post":      round(mean_post,   2),
        "momentum_score": round(momentum,    4),
        "peak_week":      peak_week,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Run validation loop
# ─────────────────────────────────────────────────────────────────────────────

def validate_trends(
    candidates: list,
    cfg: DemandConfig,
) -> tuple:
    """
    For each candidate trend, fetch Google Trends and compute momentum.
    Returns (list[TrendResult], dict[label -> pd.DataFrame]).
    """
    pytrends = TrendReq(hl=cfg.hl, tz=cfg.tz, timeout=(10, 25))
    results: list = []
    series_store: dict = {}   # label -> raw DataFrame (for plotting)

    for i, candidate in enumerate(candidates):
        label = candidate["top_label"]
        query = label_to_query(label)

        logger.info(
            f"[{i+1}/{len(candidates)}] '{label}'  ->  Google query: '{query}'"
        )

        df = fetch_trends(query, cfg, pytrends)
        time.sleep(cfg.request_delay)

        if df is None:
            logger.warning(f"Skipping '{label}' — no Trends data.")
            continue

        momentum = compute_momentum(df, cfg)
        series_store[label] = df

        result = TrendResult(
            cluster_id=candidate["cluster_id"],
            label=label,
            search_query=query,
            momentum_score=momentum["momentum_score"],
            mean_pre=momentum["mean_pre"],
            mean_during=momentum["mean_during"],
            mean_post=momentum["mean_post"],
            peak_week=momentum["peak_week"],
            is_validated=momentum["momentum_score"] >= cfg.momentum_threshold,
            supply_house_count=candidate["house_count"],
            supply_cross_house_score=candidate["cross_house_score"],
        )
        results.append(result)

        status = "VALIDATED" if result.is_validated else "look only"
        logger.info(
            f"   -> momentum={result.momentum_score:.2f}  [{status}]"
        )

    return results, series_store


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_trend_series(
    series_store: dict,
    results: list,
    cfg: DemandConfig,
    output_dir: Path,
):
    """
    Plot one time-series panel per trend, highlighting the Fashion Month window.
    Saves a combined figure: trend_timelines.png
    """
    if not series_store:
        logger.warning("No series data available for plotting.")
        return

    n = len(series_store)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(6 * cols, 3.5 * rows),
        facecolor="#0f0f0f",
    )
    axes = np.array(axes).flatten()

    result_map = {r.label: r for r in results}

    fm_start = pd.Timestamp(cfg.fashion_month_start)
    fm_end   = pd.Timestamp(cfg.fashion_month_end)

    for ax, (label, df) in zip(axes, series_store.items()):
        res = result_map.get(label)
        validated = res.is_validated if res else False
        colour = "#FF6B9D" if validated else "#888888"

        ax.set_facecolor("#1a1a1a")
        ax.plot(df["date"], df["interest"], color=colour, linewidth=1.8)
        ax.fill_between(df["date"], df["interest"], alpha=0.15, color=colour)

        # Fashion Month band
        ax.axvspan(fm_start, fm_end, alpha=0.18, color="#FF6B9D", label="Fashion Month")

        title = label[:32] + "..." if len(label) > 32 else label
        suffix = " ✓" if validated else ""
        ax.set_title(f"{title}{suffix}", color="white", fontsize=8.5, pad=4)
        ax.tick_params(colors="#999999", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        if res:
            ax.set_xlabel(
                f"momentum: {res.momentum_score:.2f}",
                color="#999999",
                fontsize=7,
            )

    # Hide unused subplots
    for ax in axes[len(series_store):]:
        ax.set_visible(False)

    validated_patch = mpatches.Patch(color="#FF6B9D", label="Validated trend")
    look_patch      = mpatches.Patch(color="#888888", label="Look only")
    band_patch      = mpatches.Patch(color="#FF6B9D", alpha=0.3, label="Fashion Month window")

    fig.legend(
        handles=[validated_patch, look_patch, band_patch],
        loc="lower center",
        ncol=3,
        fontsize=8,
        facecolor="#1a1a1a",
        labelcolor="white",
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Fashion Week Search Momentum — Supply vs. Demand Validation",
        color="white",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    plot_path = output_dir / "trend_timelines.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)
    logger.info(f"Saved trend timelines -> {plot_path}")


def plot_momentum_bar(results: list, cfg: DemandConfig, output_dir: Path):
    """Horizontal bar chart of momentum scores, sorted descending."""
    if not results:
        return

    df = pd.DataFrame([asdict(r) for r in results]).sort_values(
        "momentum_score", ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.45)), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a1a")

    colours = ["#FF6B9D" if v else "#555555" for v in df["is_validated"]]
    bars = ax.barh(df["label"], df["momentum_score"], color=colours, height=0.65)

    # Threshold line
    ax.axvline(
        cfg.momentum_threshold,
        color="white",
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold ({cfg.momentum_threshold}x)",
    )

    ax.set_xlabel("Momentum Score  (during / pre Fashion Month)", color="white", fontsize=9)
    ax.set_title("Google Trends Momentum — All Candidate Trends", color="white", fontsize=11)
    ax.tick_params(colors="#bbbbbb", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")

    plt.tight_layout()
    bar_path = output_dir / "momentum_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)
    logger.info(f"Saved momentum bar chart -> {bar_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: list, cfg: DemandConfig, output_dir: Path):
    # Full JSON
    all_path = output_dir / "all_trend_results.json"
    with open(all_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"Saved all results -> {all_path}")

    # Validated only
    validated = [r for r in results if r.is_validated]
    val_path = output_dir / "validated_trends.json"
    with open(val_path, "w") as f:
        json.dump([asdict(r) for r in validated], f, indent=2)
    logger.info(f"Saved {len(validated)} validated trends -> {val_path}")

    # Human-readable report
    report_path = output_dir / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write("DEMAND-SIDE VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fashion Month window : {cfg.fashion_month_start}  to  {cfg.fashion_month_end}\n")
        f.write(f"Momentum threshold   : {cfg.momentum_threshold}x\n")
        f.write(f"Candidates assessed  : {len(results)}\n")
        f.write(f"Trends validated     : {len(validated)}\n\n")
        f.write("─" * 60 + "\n\n")

        f.write("VALIDATED TRENDS (passed both filters)\n\n")
        for r in sorted(validated, key=lambda x: x.momentum_score, reverse=True):
            f.write(f"  {r.label.upper()}\n")
            f.write(f"    Search query     : {r.search_query}\n")
            f.write(f"    Momentum score   : {r.momentum_score:.2f}x\n")
            f.write(f"    Pre-FM interest  : {r.mean_pre:.1f}\n")
            f.write(f"    During-FM interest: {r.mean_during:.1f}\n")
            f.write(f"    Peak search week : {r.peak_week}\n")
            f.write(f"    Supply houses    : {r.supply_house_count}\n\n")

        f.write("─" * 60 + "\n\n")
        f.write("LOOKS (failed demand filter)\n\n")
        looks = [r for r in results if not r.is_validated]
        for r in sorted(looks, key=lambda x: x.momentum_score, reverse=True):
            f.write(
                f"  {r.label}  —  momentum: {r.momentum_score:.2f}x\n"
            )

    logger.info(f"Saved validation report -> {report_path}")
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: DemandConfig):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load supply-side candidates
    candidates = load_candidates(cfg)

    # 2 & 3. Fetch Trends + compute momentum
    results, series_store = validate_trends(candidates, cfg)

    # 4. Save JSON + report
    validated = save_results(results, cfg, output_dir)

    # 5. Visualise
    plot_trend_series(series_store, results, cfg, output_dir)
    plot_momentum_bar(results, cfg, output_dir)

    return validated


if __name__ == "__main__":
    cfg = DemandConfig()
    validated = run(cfg)

    print("\n── VALIDATED TRENDS (runway + search signal) ───────────────")
    for r in sorted(validated, key=lambda x: x.momentum_score, reverse=True):
        print(
            f"  {r.label:<35}  momentum: {r.momentum_score:.2f}x"
            f"  |  {r.supply_house_count} houses"
        )

    if not validated:
        print("  No trends passed both filters at the current threshold.")
        print(
            f"  Consider lowering DemandConfig.momentum_threshold "
            f"(currently {DemandConfig().momentum_threshold})."
        )

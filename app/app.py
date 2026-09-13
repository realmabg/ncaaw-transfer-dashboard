from pathlib import Path
import json
import html

import asttokens  # noqa: F401 - direct import lets Shinylive install this transitive dependency.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from data_engine import (
    CLASSES,
    POS_COLOR,
    height_str,
    load_wbb_dataset,
    resolve_default_wbb_dataset_path,
)

HERE = Path(__file__).parent
DATASET_PATH = resolve_default_wbb_dataset_path(HERE)
DATA = load_wbb_dataset(DATASET_PATH)
df = DATA["df"]
league_avg = DATA["league_avg"]
similar_to_fn = DATA["similar_to"]
conferences = DATA["conferences"]
TOTAL_PLAYERS = len(df)
HISTORICAL_PATH = (
    HERE / "wbb_historical_player_index.csv"
    if (HERE / "wbb_historical_player_index.csv").exists()
    else HERE.parent / "data" / "processed" / "wbb_historical_player_index.csv"
)
HISTORICAL = pd.read_csv(HISTORICAL_PATH) if HISTORICAL_PATH.exists() else pd.DataFrame()
HISTORICAL_TABLE_LIMIT = 25
HISTORICAL_CURRENT_LIMIT = 8
HISTORICAL_TRACKER_COMP_LIMIT = 5
TRITON_DEFAULT_MIN_MPG = 10.0
TRITON_DEFAULT_MIN_GP = 5
TRITON_TABLE_LIMITS = {"25": "Top 25", "50": "Top 50", "100": "Top 100", "all": "All"}
UCSD_WBB_ROSTER_URL = "https://ucsdtritons.com/sports/womens-basketball/roster"

POSITION_GROUPS = {
    "G": {"label": "Guard", "members": {"G"}, "color": POS_COLOR["G"]},
    "F": {"label": "Forward", "members": {"G/F", "F"}, "color": POS_COLOR["F"]},
    "C": {"label": "Center", "members": {"F/C", "C"}, "color": "#7cc47a"},
}
POSITION_GROUP_ORDER = ["G", "F", "C"]
SIMILARITY_METRIC_LABELS = {
    "mahalanobis": "Mahalanobis dist. over PC1-PC6",
    "euclidean": "Euclidean dist. over PC1-PC6",
}
RADAR_STATS = [
    ("ppg", "PPG", "ppg", "PPG", "{:.1f}"),
    ("rpg", "RPG", "rpg", "RPG", "{:.1f}"),
    ("apg", "APG", "apg", "APG", "{:.1f}"),
    ("spg", "SPG", "spg", "SPG", "{:.2f}"),
    ("bpg", "BPG", "bpg", "BPG", "{:.2f}"),
    ("ts", "TS%", "ts", "TS%", "{:.1%}"),
]
DEFAULT_RADAR_STAT_KEYS = [key for key, *_ in RADAR_STATS]
RADAR_STAT_LOOKUP = {key: stat for key, *stat in RADAR_STATS}
RADAR_PALETTE = [
    "#c8a84b",
    "#4a9eed",
    "#7cc47a",
    "#e8a44a",
    "#d86f74",
    "#8d7cc4",
]
ARCHETYPE_SCORE_LABELS = {
    "score_pg_combo": "PG / Combo Guard",
    "score_wing_2_4": "2-4 Wing",
    "score_stretch_big": "Stretch Big",
}
ARCHETYPE_COLOR = {
    "PG / Combo Guard": "#4a9eed",
    "2-4 Wing": "#5ab87a",
    "Stretch Big": "#c8a84b",
}
TRITON_ZONE_METRICS = [
    {"key": "efg", "col": "efg", "label": "eFG%", "long": "Effective FG%", "scale": 100.0, "target": 50.0, "higher_is_better": True, "weight": 20.0},
    {"key": "three_pct", "col": "tp", "label": "3PT%", "long": "Three-point percentage", "scale": 100.0, "target": 36.0, "higher_is_better": True, "weight": 18.0},
    {"key": "three_rate", "col": "three_share", "label": "3PA/FGA", "long": "Three-point rate", "scale": 100.0, "target": 45.0, "higher_is_better": True, "weight": 15.0},
    {"key": "tov_pct", "col": "tov_pct", "label": "TOV%", "long": "Turnover rate", "scale": 100.0, "target": 15.0, "higher_is_better": False, "weight": 15.0},
    {"key": "two_pct", "col": "two_pct", "label": "2PT%", "long": "Two-point percentage", "scale": 100.0, "target": 55.0, "higher_is_better": True, "weight": 12.0},
    {"key": "drb_pct", "col": "drb_pct", "label": "DRB%", "long": "Defensive rebound rate", "scale": 100.0, "target": 15.0, "higher_is_better": True, "weight": 12.0},
    {"key": "orb_pct", "col": "orb_pct", "label": "ORB%", "long": "Offensive rebound rate", "scale": 100.0, "target": 6.0, "higher_is_better": True, "weight": 8.0},
]
TRITON_SPECIAL_ARCHETYPES = {
    "stretch_big": {
        "label": "Stretch Big",
        "note": "Men's shooting gates with a women's height cutoff: 6'1\"+, 34%+ from three on a 40%+ three-point rate.",
        "criteria": [
            {"key": "stretch_height", "col": "heightIn", "label": "Height", "scale": 1.0, "target": 73.0, "higher_is_better": True, "kind": "height"},
            {"key": "stretch_three_pct", "col": "tp", "label": "3PT%", "scale": 100.0, "target": 34.0, "higher_is_better": True},
            {"key": "stretch_three_rate", "col": "three_share", "label": "3PA/FGA", "scale": 100.0, "target": 40.0, "higher_is_better": True},
        ],
    },
    "shooter": {
        "label": "3PT Specialist",
        "note": "Same as men's: 65%+ of shots from three at better than 35%.",
        "criteria": [
            {"key": "shooter_three_rate", "col": "three_share", "label": "3PA/FGA", "scale": 100.0, "target": 65.0, "higher_is_better": True},
            {"key": "shooter_three_pct", "col": "tp", "label": "3PT%", "scale": 100.0, "target": 35.0, "higher_is_better": True},
        ],
    },
}
TRITON_ARCHETYPE_FILTERS = {
    "all": "All players",
    "zone": "Triton Zone only",
    "stretch_big": "Stretch Big",
    "shooter": "3PT Specialist",
}
HISTORICAL_FEATURES = [
    ("height_inches", "heightIn", 0.6), ("mins_per_game", "mpg", 0.4),
    ("pts_per_game", "ppg", 0.8), ("treb_per_game", "rpg", 0.65),
    ("ast_per_game", "apg", 0.65), ("stl_per_game", "spg", 0.35),
    ("blk_per_game", "bpg", 0.35), ("bpm", "bpm", 0.75),
    ("eFG", "efg", 0.55), ("3P_pct", "tp", 0.55),
    ("AST_TOV", "ast_tov", 0.55), ("AST_pct", "ast_pct", 0.45),
    ("DRB_pct", "drb_pct", 0.45), ("three_share", "three_share", 0.45),
    ("rim_share", "rim_share", 0.3), ("mid_share", "mid_share", 0.25),
]


def dataset_status_text() -> str:
    if DATA["source_status"] == "loaded":
        return "2025-26 women's Division I player dataset loaded"
    return "No processed women's Division I dataset found yet."


def pct_display(value):
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "N/A" if pd.isna(num) else f"{num * 100:.1f}%"


def _as_float(value, default=np.nan):
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(num) if pd.notna(num) else default


def metric_values(frame, metric):
    return pd.to_numeric(frame.get(metric["col"], pd.Series(np.nan, index=frame.index)), errors="coerce") * metric["scale"]


def build_triton_frame(frame):
    out = frame.copy()
    weighted = pd.Series(0.0, index=out.index)
    total_weight = sum(metric["weight"] for metric in TRITON_ZONE_METRICS)
    checks = pd.Series(0, index=out.index)
    zone = pd.Series(True, index=out.index)
    for metric in TRITON_ZONE_METRICS:
        values = metric_values(out, metric)
        target = metric["target"]
        ok = values >= target if metric["higher_is_better"] else values <= target
        ok = ok.fillna(False)
        clean = values.dropna()
        spread = ((clean.quantile(.84) - clean.quantile(.16)) / 2.0) if not clean.empty else 1.0
        spread = spread if np.isfinite(spread) and spread > 0 else 1.0
        z = (values - target) / spread
        if not metric["higher_is_better"]:
            z = -z
        sub = np.where(z >= 0, 70.0 + 30.0 * np.clip(z, 0, 1), 70.0 * np.clip(1 + z / 1.5, 0, 1))
        out[f"triton_val_{metric['key']}"] = values
        out[f"triton_ok_{metric['key']}"] = ok
        out[f"triton_sub_{metric['key']}"] = sub
        weighted += pd.Series(sub, index=out.index).fillna(0) * metric["weight"]
        checks += ok.astype(int)
        zone &= ok
    out["triton_checks_passed"] = checks
    out["triton_checks_total"] = len(TRITON_ZONE_METRICS)
    out["triton_zone"] = zone
    out["triton_war"] = (weighted / total_weight).clip(0, 100)
    for key, meta in TRITON_SPECIAL_ARCHETYPES.items():
        meets = pd.Series(True, index=out.index)
        for criterion in meta["criteria"]:
            values = metric_values(out, criterion)
            target = criterion["target"]
            ok = values >= target if criterion["higher_is_better"] else values <= target
            meets &= ok.fillna(False)
        out[f"triton_is_{key}"] = meets
    return out


df = build_triton_frame(df)


def historical_slider_range(column, step=1.0, fallback=(0, 1)):
    if HISTORICAL.empty or column not in HISTORICAL.columns:
        return fallback
    vals = pd.to_numeric(HISTORICAL[column], errors="coerce").dropna()
    if vals.empty:
        return fallback
    return float(np.floor(vals.min() / step) * step), float(np.ceil(vals.max() / step) * step)


def historical_current_comps(row, n=HISTORICAL_CURRENT_LIMIT):
    if row is None or df.empty:
        return []
    cols = [(h, c, w) for h, c, w in HISTORICAL_FEATURES if h in HISTORICAL.columns and c in df.columns]
    if not cols:
        return []
    current = df.copy()
    if "pos" in current.columns and "pos" in row:
        current = current[current["pos"].eq(row["pos"])]
        if current.empty:
            current = df.copy()
    score = pd.Series(0.0, index=current.index)
    weight = pd.Series(0.0, index=current.index)
    for hist_col, current_col, w in cols:
        hval = _as_float(row.get(hist_col))
        cvals = pd.to_numeric(current[current_col], errors="coerce")
        pool = pd.concat([pd.to_numeric(HISTORICAL[hist_col], errors="coerce"), pd.to_numeric(df[current_col], errors="coerce")]).dropna()
        spread = pool.std(ddof=0)
        if not np.isfinite(hval) or not np.isfinite(spread) or spread <= 0:
            continue
        dist = ((cvals - hval) / spread).abs()
        score += dist.fillna(0) * w
        weight += cvals.notna().astype(float) * w
    current = current.assign(distance=(score / weight.replace(0, np.nan))).dropna(subset=["distance"])
    current = current.sort_values(["distance", "bpm"], ascending=[True, False]).head(n)
    comps = []
    for i, (_, comp) in enumerate(current.iterrows(), start=1):
        sim = 100 * np.exp(-0.75 * float(comp["distance"]))
        comps.append({"rank": i, "similarity_score": max(0, min(100, sim)), **comp.to_dict()})
    return comps


def historical_row_by_id(row_id):
    if HISTORICAL.empty or not row_id:
        return None
    rows = HISTORICAL[HISTORICAL["season_player_id"].astype(str).eq(str(row_id))]
    return None if rows.empty else rows.iloc[0]


def historical_profile_subtitle(row):
    if row is None:
        return ""
    parts = [
        str(row.get("team", "")).strip(),
        str(int(_as_float(row.get("year"), 0))) if np.isfinite(_as_float(row.get("year"), np.nan)) else "",
        str(row.get("pos", "")).strip(),
        str(row.get("archetype", "")).strip(),
    ]
    return " · ".join([part for part in parts if part])


def historical_metric(row, col, fmt="{:.1f}", default="N/A"):
    value = _as_float(row.get(col), np.nan) if row is not None else np.nan
    return default if not np.isfinite(value) else fmt.format(value)


def tracker_default_rows(limit=3):
    if HISTORICAL.empty or "team" not in HISTORICAL.columns:
        return []
    rows = HISTORICAL[HISTORICAL["team"].astype(str).eq("UC San Diego")].copy()
    if rows.empty:
        return []
    rows["__year"] = pd.to_numeric(rows.get("year"), errors="coerce")
    rows["__bpm"] = pd.to_numeric(rows.get("bpm"), errors="coerce")
    rows["__mpg"] = pd.to_numeric(rows.get("mins_per_game"), errors="coerce")
    rows = rows[rows["__mpg"].fillna(0) >= 10]
    rows = rows.sort_values(["__year", "__bpm", "__mpg"], ascending=[False, False, False]).head(limit)
    return [row for _, row in rows.iterrows()]


def tracker_table_head():
    return ui.div(
        {"class": "similarity-beta-table-head"},
        ui.div("#"),
        ui.div("Δ"),
        ui.div("Current player"),
        ui.div("PPG"),
        ui.div("APG"),
        ui.div("RPG"),
    )


def tracker_comp_rows(row, comps, board_index=0):
    if not comps:
        return [ui.div("No current-player comps available for this historical profile.", class_="similarity-beta-empty")]
    rows = []
    for comp in comps:
        rows.append(
            ui.div(
                {"class": "similarity-beta-row similarity-beta-row--clickable", "onclick": f"Shiny.setInputValue('d1_select_similar',{json.dumps(str(comp['id']))},{{priority:'event'}})"},
                ui.div(str(comp["rank"]), class_="similarity-beta-rank"),
                ui.div("-", class_="similarity-beta-move flat"),
                ui.div(
                    {"class": "similarity-beta-player-cell"},
                    ui.div(comp["name"], class_="similarity-beta-player"),
                    ui.div(f"{comp['team']} · {comp['cls']} · {comp['primary_archetype']}", class_="similarity-beta-team"),
                ),
                ui.div(f"{_as_float(comp.get('ppg'), 0):.1f}", class_="similarity-beta-stat"),
                ui.div(f"{_as_float(comp.get('apg'), 0):.1f}", class_="similarity-beta-stat"),
                ui.div(f"{_as_float(comp.get('rpg'), 0):.1f}", class_="similarity-beta-stat"),
            )
        )
    return rows


def tracker_ideal_card(row, board_index=0, saved=False):
    row_id = str(row.get("season_player_id", ""))
    comps = historical_current_comps(row, n=HISTORICAL_TRACKER_COMP_LIMIT)
    return ui.div(
        {"class": "similarity-beta-card"},
        ui.div(
            {"class": "similarity-beta-card-head"},
            ui.div(
                ui.div(str(row.get("player_name", "Unknown player")), class_="similarity-beta-ideal-name"),
                ui.div(historical_profile_subtitle(row), class_="similarity-beta-ideal-meta"),
            ),
            ui.div("Saved" if saved else "Ideal", class_="similarity-beta-pill"),
        ),
        ui.div(
            {"class": "similarity-beta-ideal-stats"},
            ui.div(ui.span("HT"), ui.tags.b(height_str(_as_float(row.get("height_inches"), 0)))),
            ui.div(ui.span("PPG"), ui.tags.b(historical_metric(row, "pts_per_game"))),
            ui.div(ui.span("APG"), ui.tags.b(historical_metric(row, "ast_per_game"))),
            ui.div(ui.span("RPG"), ui.tags.b(historical_metric(row, "treb_per_game"))),
        ),
        tracker_table_head(),
        ui.div({"class": "similarity-beta-table"}, *tracker_comp_rows(row, comps, board_index)),
        ui.tags.button(
            "Open in Historical",
            class_="similarity-beta-more",
            onclick=f"Shiny.setInputValue('hist_select_row',{json.dumps(row_id)},{{priority:'event'}});switchTab('hist');",
        ),
    )


def make_tracker_content(saved_ids):
    pinned = tracker_default_rows()
    pinned_ids = {str(row.get("season_player_id", "")) for row in pinned}
    saved_rows = [historical_row_by_id(row_id) for row_id in sorted(saved_ids) if str(row_id) not in pinned_ids]
    saved_rows = [row for row in saved_rows if row is not None]
    return ui.div(
        {"class": "similarity-beta-shell"},
        ui.div(
            {"class": "similarity-beta-topbar"},
            ui.div(
                ui.div("Triton Tracker", class_="similarity-beta-title"),
                ui.div("Save historical UCSD ideals, then rank the current women's D-I pool with the historical-to-current similarity model.", class_="similarity-beta-subtitle"),
            ),
            ui.div("Current pool: 2026 WBB D-I", class_="similarity-beta-refresh-note"),
        ),
        ui.div({"class": "similarity-beta-section-head"}, ui.div("Pinned UCSD ideals"), ui.div("Always shown", class_="similarity-beta-section-count")),
        ui.div({"class": "similarity-beta-grid"}, *[tracker_ideal_card(row, i) for i, row in enumerate(pinned)]) if pinned else ui.div("No UC San Diego historical ideals found in the 2021-25 data.", class_="similarity-beta-tracked-empty"),
        ui.div({"class": "similarity-beta-section-head"}, ui.div("Saved historical ideals"), ui.div(f"{len(saved_rows)} saved", class_="similarity-beta-section-count")),
        ui.div({"class": "similarity-beta-grid"}, *[tracker_ideal_card(row, i, saved=True) for i, row in enumerate(saved_rows)]) if saved_rows else ui.div("Save players from Historical Players to add custom tracker ideals.", class_="similarity-beta-tracked-empty"),
    )


def triton_metric_display(metric, row):
    value = _as_float(row.get(f"triton_val_{metric['key']}"))
    if not np.isfinite(value):
        return "N/A"
    return height_str(value) if metric.get("kind") == "height" else f"{value:.1f}"


def archetype_score_rows(row):
    rows = []
    for col, label in ARCHETYPE_SCORE_LABELS.items():
        score = _as_float(row.get(col), 0)
        primary = " primary" if label == row.get("primary_archetype") else ""
        rows.append(
            ui.div(
                {"class": f"arch-score-row{primary}"},
                ui.div(ui.span(label, class_="arch-score-name"), ui.span(f"{score:.0f}", class_="arch-score-value"), class_="arch-score-head"),
                ui.div({"class": "arch-score-track"}, ui.div({"class": "arch-score-fill", "style": f"width:{max(0, min(100, score)):.1f}%;background:{ARCHETYPE_COLOR.get(label, '#c8a84b')};"})),
            )
        )
    return rows


def position_group(value):
    text = str(value or "").strip()
    for key, meta in POSITION_GROUPS.items():
        if text in meta["members"]:
            return key
    return "Unknown"


def position_label(value):
    key = position_group(value)
    return POSITION_GROUPS.get(key, {}).get("label", "Unknown")


def position_color(value):
    key = position_group(value)
    return POSITION_GROUPS.get(key, {}).get("color", "#888")


def slider_range(frame: pd.DataFrame, col: str, step: float = 1.0, fallback: tuple[float, float] = (0.0, 1.0)):
    vals = pd.to_numeric(frame.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
    if vals.empty:
        return fallback
    lo = float(np.floor(vals.min() / step) * step)
    hi = float(np.ceil(vals.max() / step) * step)
    if step >= 1:
        return int(lo), int(hi)
    decimals = len(str(step).split(".")[1].rstrip("0"))
    return round(lo, decimals), round(hi, decimals)


def stat_box(lbl, val, avg):
    delta = float(val) - float(avg)
    sign = "+" if delta >= 0 else ""
    cls = "up" if delta > 0.001 else ("down" if delta < -0.001 else "")
    return ui.div(
        {"class": "stat-cell"},
        ui.div(str(val), class_="num"),
        ui.div(lbl, class_="lbl"),
        ui.div(f"{sign}{delta:.1f} vs avg", class_=f"delta {cls}"),
    )


def bar_row(lbl, pv, av, mx, fmt=None):
    fmt = fmt or (lambda v: f"{v:.2f}")
    wp = min(100.0, (pv / mx) * 100) if mx else 0.0
    wa = min(100.0, (av / mx) * 100) if mx else 0.0
    return ui.div(
        {"class": "cmp-row"},
        ui.div(lbl, class_="lbl"),
        ui.div(
            {"class": "cmp-bar"},
            ui.div({"class": "player-mark", "style": f"left:0;width:{wp:.1f}%"}),
            ui.div({"class": "avg-mark", "style": f"left:{wa:.1f}%"}),
        ),
        ui.div(fmt(pv), class_="val"),
    )


def bio_item(label, value, mono=False):
    return ui.div(
        {"class": "bio-item"},
        ui.div(label, class_="k"),
        ui.div(value, class_="v mono" if mono else "v"),
    )


HOVER_TPL = (
    "<b>%{customdata[0]}</b><br>"
    "%{customdata[1]} · %{customdata[2]} · %{customdata[3]}<br>"
    "%{customdata[4]:.1f} PPG · %{customdata[5]:.1f} RPG · %{customdata[6]:.1f} APG"
    "<extra></extra>"
)


def cdata(d):
    return list(
        zip(
            d["name"],
            d["pos"].map(position_label),
            d["team"],
            d["cls"],
            d["ppg"],
            d["rpg"],
            d["apg"],
            d["id"],
        )
    )


PC1_LOG_LEFT_BREAK = -4.0
PC1_LOG_RIGHT_BREAK = 4.0
PC1_LOG_FACTOR = 1.0
PC2_LOG_BREAK = -4.0
PC2_LOG_FACTOR = 1.1


def transform_pc1_value(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return np.nan
    if numeric < PC1_LOG_LEFT_BREAK:
        return float(PC1_LOG_LEFT_BREAK - np.log1p(PC1_LOG_LEFT_BREAK - numeric) * PC1_LOG_FACTOR)
    if numeric > PC1_LOG_RIGHT_BREAK:
        return float(PC1_LOG_RIGHT_BREAK + np.log1p(numeric - PC1_LOG_RIGHT_BREAK) * PC1_LOG_FACTOR)
    return float(numeric)


def transform_pc1_series(series):
    values = pd.to_numeric(series, errors="coerce")
    transformed = values.copy()
    left_mask = values < PC1_LOG_LEFT_BREAK
    right_mask = values > PC1_LOG_RIGHT_BREAK
    transformed.loc[left_mask] = PC1_LOG_LEFT_BREAK - np.log1p(PC1_LOG_LEFT_BREAK - values.loc[left_mask]) * PC1_LOG_FACTOR
    transformed.loc[right_mask] = PC1_LOG_RIGHT_BREAK + np.log1p(values.loc[right_mask] - PC1_LOG_RIGHT_BREAK) * PC1_LOG_FACTOR
    return transformed


def transform_pc2_value(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return np.nan
    if numeric >= PC2_LOG_BREAK:
        return float(numeric)
    return float(PC2_LOG_BREAK - np.log1p(PC2_LOG_BREAK - numeric) * PC2_LOG_FACTOR)


def transform_pc2_series(series):
    values = pd.to_numeric(series, errors="coerce")
    transformed = values.copy()
    mask = values < PC2_LOG_BREAK
    transformed.loc[mask] = PC2_LOG_BREAK - np.log1p(PC2_LOG_BREAK - values.loc[mask]) * PC2_LOG_FACTOR
    return transformed


def build_traces(plot_df, selected_id, dimmed_positions, dot_size=9.5, dot_opacity=0.78):
    traces = []
    grouped = plot_df.assign(pos_group=plot_df["pos"].map(position_group))
    for position in POSITION_GROUP_ORDER:
        sub = grouped[grouped["pos_group"] == position]
        if sub.empty:
            continue
        alpha = 0.06 if position in dimmed_positions else dot_opacity
        rest = sub[sub["id"] != selected_id] if selected_id else sub
        sel = sub[sub["id"] == selected_id] if selected_id else sub.iloc[0:0]
        if not rest.empty:
            traces.append(
                go.Scatter(
                    x=transform_pc1_series(rest["arch_pca_PC1"]),
                    y=transform_pc2_series(rest["arch_pca_PC2"]),
                    mode="markers",
                    marker=dict(size=dot_size, color=position_color(position), opacity=alpha, line=dict(width=0)),
                    customdata=cdata(rest),
                    hovertemplate=HOVER_TPL,
                    name=position_label(position),
                    showlegend=False,
                )
            )
        if not sel.empty:
            r = sel.iloc[0]
            traces.append(
                go.Scatter(
                    x=[transform_pc1_value(r["arch_pca_PC1"])],
                    y=[transform_pc2_value(r["arch_pca_PC2"])],
                    mode="markers",
                    marker=dict(size=dot_size + 16, color="rgba(0,0,0,0)", line=dict(color="#c8a84b", width=1.5)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            traces.append(
                go.Scatter(
                    x=[transform_pc1_value(r["arch_pca_PC1"])],
                    y=[transform_pc2_value(r["arch_pca_PC2"])],
                    mode="markers",
                    marker=dict(size=dot_size + 4, color=position_color(position), opacity=1.0, line=dict(color="#0f1623", width=1.8)),
                    customdata=[cdata(sel)[0]],
                    hovertemplate=HOVER_TPL,
                    showlegend=False,
                )
            )
    return traces


def build_trace_id_map(plot_df, selected_id, dimmed_positions):
    trace_ids = []
    grouped = plot_df.assign(pos_group=plot_df["pos"].map(position_group))
    for position in POSITION_GROUP_ORDER:
        sub = grouped[grouped["pos_group"] == position]
        if sub.empty:
            continue
        rest = sub[sub["id"] != selected_id] if selected_id else sub
        sel = sub[sub["id"] == selected_id] if selected_id else sub.iloc[0:0]
        if not rest.empty:
            trace_ids.append(rest["id"].astype(str).tolist())
        if not sel.empty:
            selected_ids = sel["id"].astype(str).tolist()
            trace_ids.append(selected_ids)
            trace_ids.append(selected_ids)
    return trace_ids


def resolve_clicked_player_id(plot_df, selected_id, dimmed_positions, trace_index, point_index):
    trace_map = build_trace_id_map(plot_df, selected_id, dimmed_positions)
    if trace_index is None or point_index is None:
        return None
    try:
        trace_ids = trace_map[int(trace_index)]
        return trace_ids[int(point_index)] if 0 <= int(point_index) < len(trace_ids) else None
    except (IndexError, ValueError, TypeError):
        return None


def robust_axis_range(series, selected_value=None, min_span=1.0, pad_ratio=0.08, quantile_clip=None):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return [-1.0, 1.0]
    if quantile_clip is not None:
        q_lo, q_hi = quantile_clip
        lo = float(vals.quantile(q_lo))
        hi = float(vals.quantile(q_hi))
    else:
        lo = float(vals.min())
        hi = float(vals.max())
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    if selected_value is not None and pd.notna(selected_value):
        selected_value = float(selected_value)
        lo = min(lo, selected_value)
        hi = max(hi, selected_value)
    span = hi - lo
    if span < min_span:
        mid = (hi + lo) / 2.0
        half = min_span / 2.0
        lo, hi = mid - half, mid + half
        span = hi - lo
    pad = max(span * pad_ratio, min_span * 0.05)
    return [lo - pad, hi + pad]


def build_layout(plot_df, selected_id=None):
    axis = dict(
        gridcolor="rgba(0,0,0,0)",
        zeroline=True,
        zerolinecolor="#1e2d47",
        zerolinewidth=1.2,
        tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#4a6080"),
        linecolor="#1e2d47",
        linewidth=1,
    )
    tf = dict(size=10, family="JetBrains Mono, monospace", color="#4a6080")
    selected_row = plot_df[plot_df["id"] == selected_id] if selected_id else plot_df.iloc[0:0]
    selected_x = transform_pc1_value(selected_row["arch_pca_PC1"].iloc[0]) if not selected_row.empty else None
    selected_y = transform_pc2_value(selected_row["arch_pca_PC2"].iloc[0]) if not selected_row.empty else None
    x_plot = transform_pc1_series(plot_df["arch_pca_PC1"])
    x_range = robust_axis_range(x_plot, selected_x)
    y_plot = transform_pc2_series(plot_df["arch_pca_PC2"])
    y_range = robust_axis_range(y_plot, selected_y)
    x_ticks = [-7, -6, -5, -4, -2, 0, 2, 4, 5, 6, 7, 8]
    x_tickvals = [transform_pc1_value(v) for v in x_ticks]
    y_ticks = [-20, -15, -10, -7, -5, 0, 5]
    tickvals = [transform_pc2_value(v) for v in y_ticks]
    return go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1623",
        margin=dict(l=64, r=18, t=16, b=60),
        xaxis=dict(
            title="PC1 · spacing ↔ rebounding",
            title_font=tf,
            range=x_range,
            tickmode="array",
            tickvals=x_tickvals,
            ticktext=[str(v) for v in x_ticks],
            **axis,
        ),
        yaxis=dict(
            title="PC2 · support role ↔ usage/creation",
            title_font=tf,
            range=y_range,
            tickmode="array",
            tickvals=tickvals,
            ticktext=[str(v) for v in y_ticks],
            **axis,
        ),
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                y0=transform_pc2_value(PC2_LOG_BREAK),
                y1=transform_pc2_value(PC2_LOG_BREAK),
                line=dict(color="rgba(74,96,128,0.55)", width=1, dash="dot"),
                layer="below",
            ),
            dict(
                type="line",
                yref="paper",
                y0=0,
                y1=1,
                x0=transform_pc1_value(PC1_LOG_LEFT_BREAK),
                x1=transform_pc1_value(PC1_LOG_LEFT_BREAK),
                line=dict(color="rgba(74,96,128,0.35)", width=1, dash="dot"),
                layer="below",
            ),
            dict(
                type="line",
                yref="paper",
                y0=0,
                y1=1,
                x0=transform_pc1_value(PC1_LOG_RIGHT_BREAK),
                x1=transform_pc1_value(PC1_LOG_RIGHT_BREAK),
                line=dict(color="rgba(74,96,128,0.35)", width=1, dash="dot"),
                layer="below",
            ),
        ],
        hoverlabel=dict(
            bgcolor="#1a2540",
            bordercolor="#c8a84b",
            font=dict(family="JetBrains Mono, monospace", size=11.5, color="#c8d4e8"),
        ),
        hovermode="closest",
        dragmode="pan",
        font=dict(family="Inter, sans-serif"),
        clickmode="event",
    )


def handle_trace_click_factory(set_selected, set_modal):
    def _clicked(trace, points, selector):
        if not points or not points.point_inds:
            return
        cd = trace.customdata[points.point_inds[0]]
        if cd is not None and len(cd) >= 8:
            import random
            pid = str(cd[7])
            set_selected(pid)
            set_modal((pid, random.random()))

    return _clicked


def percentile_value(series, value):
    vals = pd.to_numeric(series, errors="coerce").dropna().sort_values().to_numpy()
    if len(vals) == 0:
        return 0.0
    return float(np.searchsorted(vals, float(value), side="right") / len(vals) * 100)


def watchlist_rows(player_ids):
    rows = []
    for pid in player_ids:
        row_ = df[df["id"] == pid]
        if row_.empty:
            continue
        rows.append((pid, row_.iloc[0]))
    return sorted(rows, key=lambda x: str(x[1]["name"]))


def make_watchlist_radar(player_ids, stat_keys=None):
    fig = go.Figure()
    rows = watchlist_rows(player_ids)
    stat_keys = DEFAULT_RADAR_STAT_KEYS if stat_keys is None else stat_keys
    stats = [RADAR_STAT_LOOKUP[key] for key in stat_keys if key in RADAR_STAT_LOOKUP]
    if not rows or not stats:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    theta = [s[0] for s in stats]
    theta_closed = theta + [theta[0]]
    for i, (_pid, row) in enumerate(rows):
        values = [percentile_value(df[col], row[col]) for _, col, _, _ in stats]
        values_closed = values + [values[0]]
        actual = [fmt.format(float(row[col])) for _, col, _, fmt in stats]
        actual_closed = actual + [actual[0]]
        labels = [label for _, _, label, _ in stats]
        labels_closed = labels + [labels[0]]
        color = RADAR_PALETTE[i % len(RADAR_PALETTE)]
        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=theta_closed,
                mode="lines+markers",
                name=row["name"],
                line=dict(color=color, width=2.4),
                marker=dict(size=8, color=color, opacity=1, line=dict(color="#0f1623", width=1.6)),
                fill="none",
                customdata=list(zip(labels_closed, actual_closed)),
                hovertemplate="<b>%{fullData.name}</b><br>%{customdata[0]}: %{customdata[1]}<br>League percentile: %{r:.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=34, r=34, t=22, b=22),
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.04,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10, family="JetBrains Mono, monospace", color="#f4f7fb"),
            bgcolor="rgba(0,0,0,0)",
        ),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#f4f7fb"),
                gridcolor="rgba(244,247,251,0.24)",
                linecolor="rgba(244,247,251,0.34)",
                angle=90,
            ),
            angularaxis=dict(
                tickfont=dict(size=11, family="Inter, sans-serif", color="#ffffff"),
                gridcolor="rgba(244,247,251,0.24)",
                linecolor="rgba(244,247,251,0.34)",
            ),
        ),
        hoverlabel=dict(
            bgcolor="#1a2540",
            bordercolor="#c8a84b",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#c8d4e8"),
        ),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def legend_html(dimmed_positions):
    parts = []
    for position in POSITION_GROUP_ORDER:
        cls = "legend-item dim" if position in dimmed_positions else "legend-item"
        col = position_color(position)
        parts.append(
            f'<div class="{cls}" onclick="Shiny.setInputValue(\'toggle_dim\',\'{position}\',{{priority:\'event\'}})">'
            f'<span class="swatch" style="background:{col}"></span><span>{position_label(position)}</span></div>'
        )
    parts.append('<span class="legend-hint"></span>')
    return "".join(parts)


def make_sidebar(prefix, frame, conference_rows):
    mpg_min, mpg_max = slider_range(frame, "mpg", 0.1, (0.0, 40.0))
    mpg_default_min = max(float(mpg_min), 10.0)
    ppg_min, ppg_max = slider_range(frame, "ppg", 0.1, (0.0, 30.0))
    efg_min, efg_max = slider_range(frame, "efg", 0.01, (0.0, 1.0))
    tp_min, tp_max = slider_range(frame, "tp", 0.01, (0.0, 1.0))
    three_share_min, three_share_max = slider_range(frame, "three_share", 0.01, (0.0, 1.0))
    apg_min, apg_max = slider_range(frame, "apg", 0.1, (0.0, 10.0))
    ato_min, ato_max = slider_range(frame, "ast_tov", 0.1, (0.0, 5.0))
    rpg_min, rpg_max = slider_range(frame, "rpg", 0.1, (0.0, 15.0))
    drb_min, drb_max = slider_range(frame, "drb_pct", 0.01, (0.0, 1.0))
    bpg_min, bpg_max = slider_range(frame, "bpg", 0.1, (0.0, 4.0))
    spg_min, spg_max = slider_range(frame, "spg", 0.1, (0.0, 4.0))
    h_min, h_max = slider_range(frame, "heightIn", 1, (60, 84))
    eligibility_min, eligibility_max = slider_range(frame, "eligibility", 1, (1, 5))
    usg_min, usg_max = slider_range(frame, "usg", 0.01, (0.0, 1.0))
    bpm_available = pd.to_numeric(frame.get("bpm"), errors="coerce").notna().any()
    bpm_min, bpm_max = slider_range(frame, "bpm", 0.1, (-10.0, 15.0)) if bpm_available else (0, 0)
    conf_choices = {c["conf"]: c["confName"] for c in sorted(conference_rows, key=lambda x: x["confName"])}
    position_values = [p for p in ["G", "G/F", "F", "F/C", "C"] if p in set(frame["pos"].fillna("").astype(str))]
    return ui.div(
        {"class": "sidebar", "id": "sidebar"},
        ui.div("Filters", class_="sb-title"),
        ui.div(ui.div("Search by name", class_="sb-section-head"), ui.input_text(f"{prefix}_q", None, placeholder="e.g. Hannah Hidalgo"), class_="sb-section"),
        ui.div(ui.div(ui.span("Position"), ui.tags.button("clear", class_="clear-btn", onclick=f"Shiny.setInputValue('{prefix}_clear_pos',Math.random())"), class_="sb-section-head"), ui.input_checkbox_group(f"{prefix}_positions", None, choices={p: p for p in position_values}), class_="sb-section"),
        ui.div(ui.div(ui.span("Class"), ui.tags.button("clear", class_="clear-btn", onclick=f"Shiny.setInputValue('{prefix}_clear_cls',Math.random())"), class_="sb-section-head"), ui.input_checkbox_group(f"{prefix}_classes", None, choices={c: c for c in CLASSES}), class_="sb-section"),
        ui.div(ui.div(ui.span("Eligibility Used"), ui.tags.button("clear", class_="clear-btn", onclick=f"Shiny.setInputValue('{prefix}_clear_eligibility',Math.random())"), class_="sb-section-head"), ui.input_slider(f"{prefix}_eligibility", None, min=eligibility_min, max=eligibility_max, value=[eligibility_min, eligibility_max], step=1), class_="sb-section"),
        ui.div(ui.div(ui.span("Conference"), ui.tags.button("clear", class_="clear-btn", onclick=f"Shiny.setInputValue('{prefix}_clear_conf',Math.random())"), class_="sb-section-head"), ui.input_checkbox_group(f"{prefix}_confs", None, choices=conf_choices), class_="sb-section"),
        ui.div(ui.div(ui.span("Team"), ui.tags.button("clear", class_="clear-btn", onclick=f"Shiny.setInputValue('{prefix}_clear_team',Math.random())"), class_="sb-section-head"), ui.input_selectize(f"{prefix}_team", None, choices=sorted(frame["team"].dropna().astype(str).unique().tolist()), multiple=True, options={"placeholder": "Search teams..."}), class_="sb-section"),
        ui.div(ui.div("MPG", class_="sb-section-head"), ui.input_slider(f"{prefix}_mpg", None, min=mpg_min, max=mpg_max, value=[mpg_default_min, mpg_max], step=0.1), class_="sb-section"),
        ui.div(ui.div("PPG", class_="sb-section-head"), ui.input_slider(f"{prefix}_ppg_range", None, min=ppg_min, max=ppg_max, value=[ppg_min, ppg_max], step=0.1), class_="sb-section"),
        ui.div(ui.div("eFG%", class_="sb-section-head"), ui.input_slider(f"{prefix}_efg", None, min=efg_min, max=efg_max, value=[efg_min, efg_max], step=0.01), class_="sb-section"),
        ui.div(ui.div("3P%", class_="sb-section-head"), ui.input_slider(f"{prefix}_tp_range", None, min=tp_min, max=tp_max, value=[tp_min, tp_max], step=0.01), class_="sb-section"),
        ui.div(ui.div("3P Share", class_="sb-section-head"), ui.input_slider(f"{prefix}_three_share", None, min=three_share_min, max=three_share_max, value=[three_share_min, three_share_max], step=0.01), class_="sb-section"),
        ui.div(ui.div("APG", class_="sb-section-head"), ui.input_slider(f"{prefix}_apg_range", None, min=apg_min, max=apg_max, value=[apg_min, apg_max], step=0.1), class_="sb-section"),
        ui.div(ui.div("Usage", class_="sb-section-head"), ui.input_slider(f"{prefix}_usg", None, min=usg_min, max=usg_max, value=[usg_min, usg_max], step=0.01), class_="sb-section"),
        ui.div(ui.div("AST/TOV ratio", class_="sb-section-head"), ui.input_slider(f"{prefix}_ast_tov", None, min=ato_min, max=ato_max, value=[ato_min, ato_max], step=0.1), class_="sb-section"),
        ui.div(ui.div("RPG", class_="sb-section-head"), ui.input_slider(f"{prefix}_rpg_range", None, min=rpg_min, max=rpg_max, value=[rpg_min, rpg_max], step=0.1), class_="sb-section"),
        ui.div(ui.div("DRB%", class_="sb-section-head"), ui.input_slider(f"{prefix}_drb_range", None, min=drb_min, max=drb_max, value=[drb_min, drb_max], step=0.01), class_="sb-section"),
        ui.div(ui.div("BPG", class_="sb-section-head"), ui.input_slider(f"{prefix}_bpg_range", None, min=bpg_min, max=bpg_max, value=[bpg_min, bpg_max], step=0.1), class_="sb-section"),
        ui.div(ui.div("SPG", class_="sb-section-head"), ui.input_slider(f"{prefix}_spg_range", None, min=spg_min, max=spg_max, value=[spg_min, spg_max], step=0.1), class_="sb-section"),
        ui.div(ui.div("Height", class_="sb-section-head"), ui.input_slider(f"{prefix}_height", None, min=h_min, max=h_max, value=[h_min, h_max], step=1), class_="sb-section"),
        ui.div(ui.div("BPM", class_="sb-section-head"), ui.input_slider(f"{prefix}_bpm", None, min=bpm_min, max=bpm_max, value=[bpm_min, bpm_max], step=0.1), class_="sb-section") if bpm_available else ui.div(),
        ui.div({"class": "sb-count"}, ui.span("Showing", class_="lbl"), ui.output_text(f"{prefix}_filter_count")),
    )


def make_plot_area(prefix):
    return ui.div(
        {"class": "plot-area"},
        ui.div({"class": "plot-toolbar"}, ui.div(ui.HTML(""), class_="plot-headline"), ui.output_ui(f"{prefix}_plot_meta")),
        ui.div({"class": "legend-bar"}, ui.output_ui(f"{prefix}_legend_ui")),
        ui.div({"class": "scatter-wrap"}, output_widget(f"{prefix}_scatter")),
    )


def make_detail_modal(player_id, frame, league_avg_map, similar_fn, watchlist, similarity_metric="mahalanobis"):
    row = frame[frame["id"] == player_id].iloc[0]
    if similarity_metric not in SIMILARITY_METRIC_LABELS:
        similarity_metric = "mahalanobis"
    sims = similar_fn(player_id, n_sim=5, metric=similarity_metric)
    hist_comps = historical_current_comps(row, n=5)
    pc = ARCHETYPE_COLOR.get(row.get("primary_archetype"), position_color(row.get("pos", "")))
    starred = player_id in watchlist
    star_icon = "\u2605" if starred else "\u2606"
    star_label = "Remove from watchlist" if starred else "Add to watchlist"
    star_style = "color:var(--accent);" if starred else "color:var(--ink-3);"
    statline = [
        stat_box("MIN", f"{row['mpg']:.1f}", league_avg_map["mpg"]),
        stat_box("PTS", f"{row['ppg']:.1f}", league_avg_map["ppg"]),
        stat_box("REB", f"{row['rpg']:.1f}", league_avg_map["rpg"]),
        stat_box("AST", f"{row['apg']:.1f}", league_avg_map["apg"]),
        stat_box("STL", f"{row['spg']:.2f}", league_avg_map["spg"]),
        stat_box("BLK", f"{row['bpg']:.2f}", league_avg_map["bpg"]),
        stat_box("TOV", f"{row['tov']:.1f}", league_avg_map["tov"]),
        stat_box("FG%", f"{row['fg']*100:.1f}", league_avg_map["fg"] * 100),
        stat_box("3P%", f"{row['tp']*100:.1f}", league_avg_map["tp"] * 100),
        stat_box("FT%", f"{row['ft']*100:.1f}", league_avg_map["ft"] * 100),
    ]
    bpm_value = pd.to_numeric(pd.Series([row.get("bpm", np.nan)]), errors="coerce").iloc[0]
    porpag_value = pd.to_numeric(pd.Series([row.get("porpag", np.nan)]), errors="coerce").iloc[0]
    if pd.notna(bpm_value):
        statline.append(stat_box("BPM", f"{bpm_value:.1f}", 0))
    efficiency_stats = [
        ("eFG%", "efg", True), ("2P%", "two_pct", True), ("3P%", "tp", True),
        ("TS%", "ts", True), ("USG%", "usg", True), ("ORB%", "orb_pct", True),
        ("DRB%", "drb_pct", True), ("AST%", "ast_pct", True), ("TOV%", "tov_pct", True),
        ("STL%", "stl_pct", True), ("BLK%", "blk_pct", True), ("FTR", "ftr", False),
        ("PF/40", "pf_per_40", False), ("Stops/40", "stops_per_40", False),
    ]
    eff_cells = []
    for label, col, is_pct in efficiency_stats:
        val = _as_float(row.get(col))
        avg = _as_float(league_avg_map.get(col), 0)
        if not np.isfinite(val):
            continue
        if is_pct:
            eff_cells.append(stat_box(label, f"{val * 100:.1f}", avg * 100))
        else:
            eff_cells.append(stat_box(label, f"{val:.1f}", avg))
    bars = [
        bar_row("PPG", row["ppg"], league_avg_map["ppg"], 30),
        bar_row("RPG", row["rpg"], league_avg_map["rpg"], 14),
        bar_row("APG", row["apg"], league_avg_map["apg"], 12),
        bar_row("SPG", row["spg"], league_avg_map["spg"], 4),
        bar_row("BPG", row["bpg"], league_avg_map["bpg"], 4),
        bar_row("3P%", row["tp"], league_avg_map["tp"], 0.55, lambda v: f"{v*100:.1f}%"),
        bar_row("TS%", row["ts"], league_avg_map["ts"], 0.75, lambda v: f"{v*100:.1f}%"),
    ]
    shot_cards = [
        ("Rim", row.get("rim_pct_of_total_attempts"), row.get("rim_pct"), row.get("rim_assisted_pct")),
        ("Mid", row.get("mid_pct_of_total_attempts"), row.get("mid_pct"), row.get("mid_assisted_pct")),
        ("3PT", row.get("three_pct_of_total_attempts"), row.get("tp"), row.get("three_assisted_pct")),
    ]
    triton_rows = []
    for metric in TRITON_ZONE_METRICS:
        ok = bool(row.get(f"triton_ok_{metric['key']}", False))
        triton_rows.append(
            ui.div(
                {"class": f"triton-mini-row {'ok' if ok else ''}"},
                ui.div(metric["label"]),
                ui.div(f"{_as_float(row.get(f'triton_val_{metric['key']}')):.1f}"),
                ui.div(f"{metric['target']:.0f}"),
            )
        )
    sim_rows = []
    for i, s in enumerate(sims):
        sim_pos = s.get("pos") or frame.loc[frame["id"] == s["id"], "pos"].iloc[0]
        sc = position_color(sim_pos)
        sim_rows.append(
            ui.div(
                {"class": "sim-row", "onclick": f"Shiny.setInputValue('d1_select_similar','{s['id']}',{{priority:'event'}})"},
                ui.div(f"{i+1:02d}", class_="sim-rank"),
                ui.div(ui.div(s["name"], class_="nm"), ui.div(ui.span(position_label(sim_pos), class_="pos-badge", style=f"color:{sc};border-color:{sc}"), ui.span(s["team"]), ui.span(f"· {s['cls']}", style="color:var(--ink-3)"), class_="meta"), class_="sim-main"),
                ui.div(f"{s['similarity_score']:.0f}", ui.span("similarity score", class_="sim-lbl"), class_="sim-pct"),
            )
        )
    hist_rows = []
    for comp in hist_comps:
        hist_rows.append(
            ui.div(
                {"class": "sim-row historical", "onclick": f"Shiny.setInputValue('d1_select_similar','{comp['id']}',{{priority:'event'}})"},
                ui.div(f"{comp['rank']:02d}", class_="sim-rank"),
                ui.div(
                    ui.div(comp["name"], class_="nm"),
                    ui.div(ui.span(comp["team"]), ui.span(f"· {comp['cls']} · {comp['primary_archetype']}", style="color:var(--ink-3)"), class_="meta"),
                    class_="sim-main",
                ),
                ui.div(f"{comp['similarity_score']:.0f}", ui.span("historical fit", class_="sim-lbl"), class_="sim-pct"),
            )
        )

    body = ui.div(
        {"id": "detail-body"},
        ui.div(
            {"class": "detail-col"},
            ui.div({"class": "player-name-row"}, ui.div(row["name"], class_="player-name"), ui.tags.button({"class": "star-btn", "title": star_label, "style": star_style, "onclick": f"Shiny.setInputValue('toggle_watchlist','{player_id}',{{priority:'event'}})"}, star_icon)),
            ui.div(ui.span({"class": "team-dot", "style": f"background:{pc}"}), f"{row['team']} · {row['confName']}", class_="player-team"),
            ui.div({"class": "bio-grid"}, bio_item("Division", "WBB D-I"), bio_item("Position", position_label(row["pos"])), bio_item("Archetype", row["primary_archetype"]), bio_item("Class", row["cls"]), bio_item("Eligibility Used", str(int(row["eligibility"])), mono=True), bio_item("Height", height_str(int(row["heightIn"])), mono=True), bio_item("Games", str(int(row["gp"])), mono=True), bio_item("Min/G", f"{row['mpg']:.1f}", mono=True), bio_item("BPM", f"{bpm_value:.1f}" if pd.notna(bpm_value) else "N/A", mono=True), bio_item("PORPAG", f"{porpag_value:.2f}" if pd.notna(porpag_value) else "N/A", mono=True)),
            ui.div(ui.div("Archetype", class_="col-title"), *archetype_score_rows(row), class_="arch-score-panel"),
            ui.div(ui.div("Triton Zone", ui.span(f"{row.get('triton_checks_passed', 0)}/{len(TRITON_ZONE_METRICS)} checks · {row.get('triton_war', 0):.0f} WAR", class_="sub"), class_="col-title"), ui.div({"class": "triton-mini"}, *triton_rows), class_="arch-score-panel"),
        ),
        ui.div(
            {"class": "detail-col"},
            ui.div("Season Statline ", ui.span("2025–26", class_="sub"), class_="col-title"),
            ui.div({"class": "statline"}, *statline),
            ui.div("Efficiency", ui.span("rates and impact", class_="sub"), class_="col-title"),
            ui.div({"class": "statline"}, *eff_cells),
            ui.div("vs. League Average ", ui.span("unweighted mean, all WBB D-I players", class_="sub"), class_="col-title"),
            *bars,
            ui.div(ui.tags.b("Bar", style="color:var(--ink-2)"), " = player value.  ", ui.tags.b("Tick", style="color:var(--ink-2)"), " = league mean.", class_="bar-note"),
            ui.div("Shot Profile", ui.span("share · FG% · assisted%", class_="sub"), class_="col-title"),
            ui.div({"class": "shot-profile-grid"}, *[
                ui.div({"class": "shot-profile-card"}, ui.div(label, class_="k"), ui.div(pct_display(share), class_="v"), ui.div(f"{pct_display(pct)} FG · {pct_display(ast)} assisted", class_="s"))
                for label, share, pct, ast in shot_cards
            ]),
        ),
        ui.div(
            {"class": "detail-col"},
            ui.div("Most Similar Current Players ", ui.span(SIMILARITY_METRIC_LABELS[similarity_metric], class_="sub"), class_="col-title"),
            ui.div(ui.input_radio_buttons("modal_similarity_metric", None, choices={"mahalanobis": "Mahalanobis", "euclidean": "Euclidean"}, selected=similarity_metric, inline=True), class_="sim-metric-control"),
            *sim_rows,
            ui.div("Historical-to-Current Fits", ui.span("2021-25 profile to 2026 pool", class_="sub"), class_="col-title"),
            *(hist_rows if hist_rows else [ui.div("No historical fit rows available.", class_="qual-note")]),
        ),
    )
    return ui.modal(body, title=ui.HTML(f"Player Profile <b>· {row['name']}</b> <span class='div-badge'>WBB D-I</span>"), easy_close=True, size="xl", footer=None)


def make_historical_tab():
    height_min, height_max = historical_slider_range("height_inches", 1, (58, 78))
    mpg_min, mpg_max = historical_slider_range("mins_per_game", .5, (0, 38))
    ppg_min, ppg_max = historical_slider_range("pts_per_game", .5, (0, 30))
    apg_min, apg_max = historical_slider_range("ast_per_game", .5, (0, 10))
    rpg_min, rpg_max = historical_slider_range("treb_per_game", .5, (0, 15))
    bpm_min, bpm_max = historical_slider_range("bpm", .5, (-20, 20))
    years = sorted(pd.to_numeric(HISTORICAL.get("year", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    confs = sorted(HISTORICAL.get("conf", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
    teams = sorted(HISTORICAL.get("team", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
    arches = sorted(HISTORICAL.get("archetype", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
    return ui.div(
        {"id": "hist-tab", "class": "tab-panel"},
        ui.div(
            {"class": "historical-shell"},
            ui.div(
                {"class": "historical-header-card"},
                ui.div("Historical Players", class_="historical-title"),
                ui.div("Search player", class_="historical-search-label"),
                ui.input_text("hist_q", None, placeholder="Search a past player..."),
                ui.div(
                    {"class": "historical-filter-row"},
                    ui.div({"class": "historical-filter-field"}, ui.div("Season", class_="historical-filter-title"), ui.input_selectize("hist_season", None, choices={str(y): str(y) for y in years}, selected=[], multiple=True, options={"placeholder": "Any season", "plugins": ["remove_button"]})),
                    ui.div({"class": "historical-filter-field"}, ui.div("Conference", class_="historical-filter-title"), ui.input_selectize("hist_conf", None, choices={c: c for c in confs}, selected=[], multiple=True, options={"placeholder": "Any conference", "plugins": ["remove_button"]})),
                    ui.div({"class": "historical-filter-field"}, ui.div("Team", class_="historical-filter-title"), ui.input_selectize("hist_team", None, choices={t: t for t in teams}, selected=[], multiple=True, options={"placeholder": "Any team", "plugins": ["remove_button"]})),
                    ui.div({"class": "historical-filter-field"}, ui.div("Pos", class_="historical-filter-title"), ui.input_selectize("hist_pos", None, choices={p: p for p in POSITION_GROUP_ORDER + ["G/F", "F/C"]}, selected=[], multiple=True, options={"placeholder": "Any position", "plugins": ["remove_button"]})),
                    ui.div({"class": "historical-filter-field"}, ui.div("Archetype", class_="historical-filter-title"), ui.input_selectize("hist_arch", None, choices={a: a for a in arches}, selected=[], multiple=True, options={"placeholder": "Any archetype", "plugins": ["remove_button"]})),
                    ui.div({"class": "historical-filter-field historical-filter-field--slider"}, ui.div("Height range", class_="historical-filter-title"), ui.input_slider("hist_height", None, min=int(height_min), max=int(height_max), value=[int(height_min), int(height_max)], step=1)),
                    ui.div({"class": "historical-filter-field historical-filter-field--slider"}, ui.div("Minutes minimum", class_="historical-filter-title"), ui.input_slider("hist_mpg", None, min=float(mpg_min), max=float(mpg_max), value=max(5.0, float(mpg_min)), step=.5)),
                ),
                ui.tags.details(
                    {"class": "historical-more-filters"},
                    ui.tags.summary("Additional filters"),
                    ui.div(
                        {"class": "historical-filter-row historical-filter-row--additional"},
                        ui.div({"class": "historical-filter-field historical-filter-field--slider"}, ui.div("Points minimum", class_="historical-filter-title"), ui.input_slider("hist_ppg_min", None, min=float(ppg_min), max=float(ppg_max), value=float(ppg_min), step=.5)),
                        ui.div({"class": "historical-filter-field historical-filter-field--slider"}, ui.div("Assists minimum", class_="historical-filter-title"), ui.input_slider("hist_apg_min", None, min=float(apg_min), max=float(apg_max), value=float(apg_min), step=.5)),
                        ui.div({"class": "historical-filter-field historical-filter-field--slider"}, ui.div("Rebounds minimum", class_="historical-filter-title"), ui.input_slider("hist_rpg_min", None, min=float(rpg_min), max=float(rpg_max), value=float(rpg_min), step=.5)),
                        ui.div({"class": "historical-filter-field historical-filter-field--slider"}, ui.div("BPM minimum", class_="historical-filter-title"), ui.input_slider("hist_bpm_min", None, min=float(bpm_min), max=float(bpm_max), value=float(bpm_min), step=.5)),
                    ),
                ),
            ),
            ui.div({"class": "historical-results-head"}, ui.output_text("hist_results_count"), ui.div("Click a row to open a profile and load current-player comps.", class_="historical-results-note")),
            ui.output_ui("historical_table_ui"),
            ui.output_ui("historical_current_comps_ui"),
        ),
    )


def make_triton_tab():
    return ui.div(
        {"id": "triton-tab", "class": "tab-panel"},
        ui.div(
            {"class": "triton-shell"},
            ui.div(
                {"class": "triton-header-card"},
                ui.div("Triton Zone", class_="triton-title"),
                ui.div(
                    "Every D-I player scored against the staff's Triton Zone targets and ranked by the weighted fit. Hitting a target is worth 70 on that metric, clearing it comfortably earns up to 100, so the board separates players who merely qualify from players who live in the zone.",
                    class_="triton-lede",
                ),
                ui.div(
                    {"class": "triton-filter-row"},
                    ui.div({"class": "triton-filter-field"}, ui.div("Search player", class_="triton-filter-title"), ui.input_text("triton_q", None, placeholder="Search a player...")),
                    ui.div({"class": "triton-filter-field"}, ui.div("Conference", class_="triton-filter-title"), ui.input_selectize("triton_conf", None, choices={r["confName"]: r["confName"] for r in conferences}, selected=[], multiple=True, options={"placeholder": "Any conference", "plugins": ["remove_button"]})),
                    ui.div({"class": "triton-filter-field"}, ui.div("Team", class_="triton-filter-title"), ui.input_selectize("triton_team", None, choices={t: t for t in sorted(df["team"].dropna().unique())}, selected=[], multiple=True, options={"placeholder": "Any team", "plugins": ["remove_button"]})),
                    ui.div({"class": "triton-filter-field"}, ui.div("Pos", class_="triton-filter-title"), ui.input_selectize("triton_pos", None, choices={p: p for p in sorted(df["pos"].dropna().unique())}, selected=[], multiple=True, options={"placeholder": "Any position", "plugins": ["remove_button"]})),
                    ui.div({"class": "triton-filter-field"}, ui.div("Class", class_="triton-filter-title"), ui.input_selectize("triton_cls", None, choices={c: c for c in CLASSES}, selected=[], multiple=True, options={"placeholder": "Any class", "plugins": ["remove_button"]})),
                ),
                ui.div(
                    {"class": "triton-filter-row triton-filter-row--second"},
                    ui.div(
                        {"class": "triton-filter-field triton-filter-field--wide"},
                        ui.div("Archetype filter", class_="triton-filter-title"),
                        ui.input_radio_buttons("triton_arch", None, choices=TRITON_ARCHETYPE_FILTERS, selected="all", inline=True),
                        ui.input_checkbox("triton_require_zone", "Archetype must also clear the full Triton Zone", value=False),
                    ),
                    ui.div({"class": "triton-filter-field triton-filter-field--slider"}, ui.div("Minutes per game minimum", class_="triton-filter-title"), ui.input_slider("triton_min_mpg", None, min=0, max=35, value=TRITON_DEFAULT_MIN_MPG, step=.5)),
                    ui.div({"class": "triton-filter-field triton-filter-field--slider"}, ui.div("Games played minimum", class_="triton-filter-title"), ui.input_slider("triton_min_gp", None, min=0, max=35, value=TRITON_DEFAULT_MIN_GP, step=1)),
                    ui.div({"class": "triton-filter-field triton-filter-field--slider"}, ui.div("Zone checks cleared minimum", class_="triton-filter-title"), ui.input_slider("triton_min_checks", None, min=0, max=len(TRITON_ZONE_METRICS), value=0, step=1)),
                    ui.div({"class": "triton-filter-field"}, ui.div("Board length", class_="triton-filter-title"), ui.input_select("triton_limit", None, choices=TRITON_TABLE_LIMITS, selected="100")),
                ),
                ui.tags.details(
                    {"class": "triton-more"},
                    ui.tags.summary("Triton Zone thresholds"),
                    ui.div("The current fixed targets match the men’s dashboard target set.", class_="triton-more-note"),
                    ui.div(
                        {"class": "triton-threshold-grid"},
                        *[
                            ui.div(
                                {"class": "triton-threshold-field"},
                                ui.div(ui.span(metric["label"], class_="triton-threshold-name"), ui.span(metric["long"], class_="triton-threshold-hint"), class_="triton-threshold-head"),
                                ui.div(f"{metric['target']:.0f}{'%' if metric['col'] != 'heightIn' else ''}", class_="triton-threshold-static"),
                            )
                            for metric in TRITON_ZONE_METRICS
                        ],
                    ),
                    ui.div("Archetype criteria", class_="triton-more-subhead"),
                    *[
                        ui.div(
                            ui.div(ui.span(archetype["label"], class_="triton-more-arch"), ui.span(archetype["note"], class_="triton-threshold-hint"), class_="triton-more-archhead"),
                            ui.div(
                                {"class": "triton-threshold-grid"},
                                *[
                                    ui.div(
                                        {"class": "triton-threshold-field"},
                                        ui.div(ui.span(criterion["label"], class_="triton-threshold-name"), class_="triton-threshold-head"),
                                        ui.div(
                                            height_str(criterion["target"]) if criterion.get("kind") == "height" else f"{criterion['target']:.0f}%",
                                            class_="triton-threshold-static",
                                        ),
                                    )
                                    for criterion in archetype["criteria"]
                                ],
                            ),
                        )
                        for archetype in TRITON_SPECIAL_ARCHETYPES.values()
                    ],
                ),
                ui.tags.details(
                    {"class": "triton-more"},
                    ui.tags.summary("Triton Zone weights"),
                    ui.div("How much each metric counts toward the score. These match the current staff-weighted setup used for the women's board.", class_="triton-more-note"),
                    ui.div(
                        {"class": "triton-threshold-grid triton-threshold-grid--weights"},
                        *[
                            ui.div(
                                {"class": "triton-weight-field"},
                                ui.div(ui.span(metric["label"], class_="triton-threshold-name"), ui.span(f"{metric['weight'] / sum(m['weight'] for m in TRITON_ZONE_METRICS):.0%}", class_="triton-threshold-hint"), class_="triton-threshold-head"),
                                ui.div(f"{metric['weight']:.0f}", class_="triton-threshold-static"),
                            )
                            for metric in TRITON_ZONE_METRICS
                        ],
                    ),
                ),
            ),
            ui.div({"class": "triton-results-head"}, ui.output_text("triton_results_count"), ui.div("Click a row to open the player profile.", class_="triton-results-note")),
            ui.output_ui("triton_table_ui"),
        ),
    )


def make_tracker_tab():
    return ui.div(
        {"id": "tracker-tab", "class": "tab-panel"},
        ui.output_ui("tracker_ui"),
    )


def make_lineup_tab():
    return ui.div(
        {"id": "lineup-tab", "class": "tab-panel"},
        ui.div(
            {"class": "historical-shell lineup-shell"},
            ui.div(
                {"class": "historical-header-card"},
                ui.div("UCSD 2026-27 Lineup Beta", class_="historical-title"),
                ui.div(ui.HTML(f"Watchlist-driven lineup sketch. Use the <a href='{UCSD_WBB_ROSTER_URL}' target='_blank'>official UC San Diego women's roster</a> as the roster reference."), class_="historical-results-note"),
            ),
            ui.output_ui("lineup_ui"),
        ),
    )


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,400;1,8..60,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"),
        ui.include_css(str(HERE / "www" / "styles.css"), method="inline"),
        ui.tags.script(
            """
            function bindD1ScatterClick() {
                var wrapper = document.getElementById('d1_scatter');
                var graph = wrapper ? wrapper.querySelector('.js-plotly-plot') : null;
                if (!graph || graph.dataset.codexPlotlyClickBound === '1' || typeof graph.on !== 'function') {
                    return;
                }
                graph.on('plotly_click', function(ev) {
                    var pt = ev && ev.points && ev.points[0];
                    if (!pt || !window.Shiny || !window.Shiny.setInputValue) return;
                    window.Shiny.setInputValue('d1_plot_click', {
                        trace_index: pt.curveNumber,
                        point_index: pt.pointNumber,
                        nonce: Date.now()
                    }, {priority: 'event'});
                });
                graph.dataset.codexPlotlyClickBound = '1';
            }

            function startD1ScatterBinding() {
                bindD1ScatterClick();
                if (window.__codexD1ScatterBindInterval) return;
                window.__codexD1ScatterBindInterval = window.setInterval(bindD1ScatterClick, 1000);
            }

            function switchTab(tab) {
                document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.remove('active'); });
                document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active-d1','active-wl','active-hist','active-triton','active-tracker','active-lineup'); });
                document.getElementById(tab+'-tab').classList.add('active');
                document.getElementById('btn-'+tab).classList.add('active-'+tab);
                if (window.Shiny && window.Shiny.setInputValue) {
                    window.Shiny.setInputValue('active_tab', tab, {priority: 'event'});
                }
                requestAnimationFrame(function() {
                    requestAnimationFrame(function() {
                        var panel = document.getElementById(tab+'-tab');
                        if (!panel) return;
                        panel.querySelectorAll('.js-plotly-plot').forEach(function(el) {
                            if (window.Plotly) Plotly.Plots.resize(el);
                        });
                        startD1ScatterBinding();
                    });
                });
            }

            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', startD1ScatterBinding);
            } else {
                startD1ScatterBinding();
            }

            document.addEventListener('shiny:connected', startD1ScatterBinding);
            """
        ),
    ),
    ui.div(
        {"id": "atlas-shell"},
        ui.div(
            {"id": "masthead"},
            ui.div({"class": "mast-left"}, ui.div(ui.HTML("NCAA Women's Basketball <span class='dot'></span> 2025–26"), class_="kicker"), ui.div(ui.HTML("Player <em>Dashboard</em>"), class_="atlas-title"), ui.div("Women's Division I transfer dashboard.", class_="dek"), ui.div(dataset_status_text(), class_="byline")),
            ui.div(
                {"class": "mast-meta"},
                ui.div(ui.div(str(TOTAL_PLAYERS), class_="mast-stat-num"), ui.div("D-I Players", class_="mast-stat-lbl"), class_="mast-stat"),
                ui.div(ui.div(str(df["team"].nunique() if not df.empty else 0), class_="mast-stat-num"), ui.div("D-I Teams", class_="mast-stat-lbl"), class_="mast-stat"),
                ui.div(ui.div(str(df["confName"].nunique() if not df.empty else 0), class_="mast-stat-num"), ui.div("Conferences", class_="mast-stat-lbl"), class_="mast-stat"),
            ),
        ),
        ui.div(
            {"id": "tab-switcher"},
            ui.tags.button("Division I", id="btn-d1", class_="tab-btn active-d1", onclick="switchTab('d1')"),
            ui.tags.button("Triton Tracker", id="btn-tracker", class_="tab-btn", onclick="switchTab('tracker')"),
            ui.tags.button("Triton Zone", id="btn-triton", class_="tab-btn", onclick="switchTab('triton')"),
            ui.tags.button("Historical Players", id="btn-hist", class_="tab-btn", onclick="switchTab('hist')"),
            ui.tags.button("Lineup Beta", id="btn-lineup", class_="tab-btn", onclick="switchTab('lineup')"),
            ui.div({"class": "tab-sep"}),
            ui.tags.button(ui.HTML('Watchlist <span id="wl-badge" class="wl-badge" style="display:none">0</span>'), id="btn-wl", class_="tab-btn", onclick="switchTab('wl')"),
        ),
        ui.div(
            {"id": "tab-content"},
            ui.div({"id": "d1-tab", "class": "tab-panel active"}, ui.div({"class": "body-grid"}, make_sidebar("d1", df, conferences), make_plot_area("d1"))),
            make_tracker_tab(),
            make_triton_tab(),
            make_historical_tab(),
            make_lineup_tab(),
            ui.div(
                {"id": "wl-tab", "class": "tab-panel"},
                ui.div(
                    {"class": "wl-shell"},
                    ui.div({"class": "wl-header"}, ui.div("Watchlist", class_="wl-title"), ui.output_text("wl_count")),
                    ui.div(
                        {"class": "wl-radar-wrap"},
                        ui.div({"class": "wl-radar-head"}, ui.div("Radar Comparison", class_="wl-radar-title"), ui.div("percentile within women's Division I", class_="wl-radar-note")),
                        ui.div({"class": "wl-radar"}, output_widget("watchlist_radar")),
                        ui.div({"class": "wl-radar-tools"}, ui.output_ui("wl_radar_picker")),
                    ),
                    ui.output_ui("watchlist_ui"),
                ),
            ),
        ),
        ui.div({"id": "site-footer"}, "Women’s Division I dashboard"),
    ),
    ui.output_ui("d1_modal_trigger"),
)


def server(input, output, session):
    d1_sel = reactive.Value(None)
    d1_dim = reactive.Value(set())
    watchlist = reactive.Value(set())
    radar_selected = reactive.Value([])
    radar_stat_selected = reactive.Value(DEFAULT_RADAR_STAT_KEYS)
    modal_req = reactive.Value(None)
    modal_player = reactive.Value(None)
    modal_similarity_metric = reactive.Value("mahalanobis")
    historical_selected = reactive.Value(None)
    tracker_ids = reactive.Value(set())

    def sync_scatter(fig, plot_df, selected_id, dimmed_arch):
        traces = build_traces(plot_df, selected_id, dimmed_arch)
        layout = build_layout(plot_df, selected_id=selected_id)
        with fig.batch_update():
            fig.data = []
            for trace in traces:
                fig.add_trace(trace)
            fig.update_layout(layout)

    def sync_radar_selection(player_ids):
        available = [pid for pid, *_ in watchlist_rows(player_ids)]
        selected = [pid for pid in radar_selected.get() if pid in available][:2]
        for pid in available:
            if len(selected) >= 2:
                break
            if pid not in selected:
                selected.append(pid)
        radar_selected.set(selected)

    @reactive.effect
    @reactive.event(input.toggle_watchlist)
    def _toggle_watchlist():
        pid = input.toggle_watchlist()
        curr = set(watchlist.get())
        curr.discard(pid) if pid in curr else curr.add(pid)
        watchlist.set(curr)
        sync_radar_selection(curr)
        import random
        modal_req.set((pid, random.random()))

    @reactive.effect
    @reactive.event(input.toggle_dim)
    def _all_dim():
        arch = input.toggle_dim()
        curr = set(d1_dim.get())
        curr.discard(arch) if arch in curr else curr.add(arch)
        d1_dim.set(curr)

    @reactive.effect
    @reactive.event(modal_req)
    def _open_modal():
        req = modal_req.get()
        if not req:
            return
        pid, _ = req
        row = df[df["id"] == pid]
        if row.empty:
            return
        modal_player.set(pid)
        ui.modal_show(make_detail_modal(pid, df, league_avg, similar_to_fn, watchlist.get(), modal_similarity_metric.get()))

    @reactive.effect
    @reactive.event(input.modal_similarity_metric)
    def _modal_similarity_metric_changed():
        metric = input.modal_similarity_metric()
        if metric not in SIMILARITY_METRIC_LABELS:
            metric = "mahalanobis"
        if metric == modal_similarity_metric.get():
            return
        modal_similarity_metric.set(metric)
        pid = modal_player.get()
        if pid:
            import random
            modal_req.set((pid, random.random()))

    @reactive.effect
    @reactive.event(input.wl_open_player)
    def _wl_open_player():
        pid = input.wl_open_player()
        if pid:
            import random
            modal_req.set((pid, random.random()))

    @reactive.effect
    @reactive.event(input.hist_select_row)
    def _hist_select_row():
        row_id = input.hist_select_row()
        if row_id:
            historical_selected.set(str(row_id))

    @reactive.effect
    @reactive.event(input.tracker_toggle)
    def _tracker_toggle():
        row_id = input.tracker_toggle()
        if not row_id:
            return
        curr = set(tracker_ids.get())
        curr.discard(str(row_id)) if str(row_id) in curr else curr.add(str(row_id))
        tracker_ids.set(curr)

    @reactive.effect
    @reactive.event(input.triton_open_player)
    def _triton_open_player():
        pid = input.triton_open_player()
        if pid:
            import random
            modal_req.set((pid, random.random()))

    @reactive.effect
    @reactive.event(input.d1_clear_pos)
    def _d1_clear_pos():
        ui.update_checkbox_group("d1_positions", selected=[])

    @reactive.effect
    @reactive.event(input.d1_clear_cls)
    def _d1_clear_cls():
        ui.update_checkbox_group("d1_classes", selected=[])

    @reactive.effect
    @reactive.event(input.d1_clear_eligibility)
    def _d1_clear_eligibility():
        vals = pd.to_numeric(df["eligibility"], errors="coerce").dropna()
        if not vals.empty:
            ui.update_slider("d1_eligibility", value=[int(vals.min()), int(vals.max())])

    @reactive.effect
    @reactive.event(input.d1_clear_conf)
    def _d1_clear_conf():
        ui.update_checkbox_group("d1_confs", selected=[])

    @reactive.effect
    @reactive.event(input.d1_clear_team)
    def _d1_clear_team():
        ui.update_selectize("d1_team", selected=[])

    @reactive.effect
    @reactive.event(input.d1_select_similar)
    def _d1_select_similar():
        sid = input.d1_select_similar()
        if sid:
            d1_sel.set(sid)
            ui.modal_remove()
            import random
            modal_req.set((sid, random.random()))

    @reactive.effect
    @reactive.event(input.d1_plot_click)
    def _d1_plot_click():
        click = input.d1_plot_click()
        if not click:
            return
        pid = resolve_clicked_player_id(
            d1_plot_df(),
            d1_sel.get(),
            d1_dim.get(),
            click.get("trace_index"),
            click.get("point_index"),
        )
        if not pid:
            return
        d1_sel.set(pid)
        ui.modal_remove()
        import random
        modal_req.set((pid, random.random()))

    @reactive.calc
    def d1_filtered():
        d = df.copy()
        q = (input.d1_q() or "").strip().lower()
        if q:
            d = d[d["name"].str.lower().str.contains(q, na=False)]
        ps = list(input.d1_positions() or [])
        if ps:
            d = d[d["pos"].isin(ps)]
        cs = list(input.d1_classes() or [])
        if cs:
            d = d[d["cls"].isin(cs)]
        lo, hi = input.d1_eligibility()
        d = d[(d["eligibility"] >= lo) & (d["eligibility"] <= hi)]
        xs = list(input.d1_confs() or [])
        if xs:
            d = d[d["conf"].isin(xs)]
        teams = list(input.d1_team() or [])
        if teams:
            d = d[d["team"].isin(teams)]
        lo, hi = input.d1_mpg()
        d = d[(d["mpg"] >= lo) & (d["mpg"] <= hi)]
        lo, hi = input.d1_ppg_range()
        d = d[(d["ppg"] >= lo) & (d["ppg"] <= hi)]
        lo, hi = input.d1_rpg_range()
        d = d[(d["rpg"] >= lo) & (d["rpg"] <= hi)]
        lo, hi = input.d1_drb_range()
        d = d[(d["drb_pct"] >= lo) & (d["drb_pct"] <= hi)]
        lo, hi = input.d1_efg()
        d = d[(d["efg"] >= lo) & (d["efg"] <= hi)]
        lo, hi = input.d1_tp_range()
        d = d[(d["tp"] >= lo) & (d["tp"] <= hi)]
        lo, hi = input.d1_three_share()
        d = d[(d["three_share"] >= lo) & (d["three_share"] <= hi)]
        lo, hi = input.d1_apg_range()
        d = d[(d["apg"] >= lo) & (d["apg"] <= hi)]
        lo, hi = input.d1_usg()
        d = d[(d["usg"] >= lo) & (d["usg"] <= hi)]
        lo, hi = input.d1_spg_range()
        d = d[(d["spg"] >= lo) & (d["spg"] <= hi)]
        lo, hi = input.d1_bpg_range()
        d = d[(d["bpg"] >= lo) & (d["bpg"] <= hi)]
        lo, hi = input.d1_ast_tov()
        d = d[(d["ast_tov"] >= lo) & (d["ast_tov"] <= hi)]
        lo, hi = input.d1_height()
        d = d[(d["heightIn"] >= lo) & (d["heightIn"] <= hi)]
        if "bpm" in d.columns and input.d1_bpm() is not None:
            lo, hi = input.d1_bpm()
            d = d[(pd.to_numeric(d["bpm"], errors="coerce").fillna(lo) >= lo) & (pd.to_numeric(d["bpm"], errors="coerce").fillna(hi) <= hi)]
        return d

    @reactive.calc
    def historical_filtered():
        if HISTORICAL.empty:
            return HISTORICAL.copy()
        d = HISTORICAL.copy()
        q = (input.hist_q() or "").strip().lower()
        if q:
            d = d[d["player_name"].str.lower().str.contains(q, na=False)]
        years = [int(y) for y in list(input.hist_season() or [])]
        if years:
            d = d[pd.to_numeric(d["year"], errors="coerce").isin(years)]
        confs = list(input.hist_conf() or [])
        if confs:
            d = d[d["conf"].isin(confs)]
        teams = list(input.hist_team() or [])
        if teams:
            d = d[d["team"].isin(teams)]
        poss = list(input.hist_pos() or [])
        if poss:
            d = d[d["pos"].isin(poss)]
        arches = list(input.hist_arch() or [])
        if arches:
            d = d[d["archetype"].isin(arches)]
        lo, hi = input.hist_height()
        d = d[(pd.to_numeric(d["height_inches"], errors="coerce") >= lo) & (pd.to_numeric(d["height_inches"], errors="coerce") <= hi)]
        d = d[pd.to_numeric(d["mins_per_game"], errors="coerce").fillna(0) >= float(input.hist_mpg())]
        d = d[pd.to_numeric(d["pts_per_game"], errors="coerce").fillna(0) >= float(input.hist_ppg_min())]
        d = d[pd.to_numeric(d["ast_per_game"], errors="coerce").fillna(0) >= float(input.hist_apg_min())]
        d = d[pd.to_numeric(d["treb_per_game"], errors="coerce").fillna(0) >= float(input.hist_rpg_min())]
        d = d[pd.to_numeric(d["bpm"], errors="coerce").fillna(-999) >= float(input.hist_bpm_min())]
        return d.sort_values(["year", "bpm", "mins_per_game"], ascending=[False, False, False]).head(HISTORICAL_TABLE_LIMIT)

    @reactive.calc
    def triton_filtered():
        d = df.copy()
        q = (input.triton_q() or "").strip().lower()
        if q:
            d = d[d["name"].str.lower().str.contains(q, na=False)]
        confs = list(input.triton_conf() or [])
        if confs:
            d = d[d["confName"].isin(confs)]
        teams = list(input.triton_team() or [])
        if teams:
            d = d[d["team"].isin(teams)]
        poss = list(input.triton_pos() or [])
        if poss:
            d = d[d["pos"].isin(poss)]
        classes = list(input.triton_cls() or [])
        if classes:
            d = d[d["cls"].isin(classes)]
        d = d[pd.to_numeric(d["mpg"], errors="coerce").fillna(0) >= float(input.triton_min_mpg())]
        d = d[pd.to_numeric(d["gp"], errors="coerce").fillna(0) >= float(input.triton_min_gp())]
        d = d[pd.to_numeric(d["triton_checks_passed"], errors="coerce").fillna(0) >= float(input.triton_min_checks())]
        arch = input.triton_arch() or "all"
        if arch == "zone":
            d = d[d["triton_zone"]]
        elif arch in {"stretch_big", "shooter"}:
            d = d[d[f"triton_is_{arch}"]]
            if bool(input.triton_require_zone()):
                d = d[d["triton_zone"]]
        return d.sort_values(["triton_war", "bpm"], ascending=[False, False])

    @reactive.calc
    def d1_plot_df():
        ids = set(d1_filtered()["id"])
        sid = d1_sel.get()
        if sid:
            ids.add(sid)
        return df[df["id"].isin(ids)]

    @output
    @render.text
    def d1_filter_count():
        return f"{len(d1_filtered())} / {TOTAL_PLAYERS}"

    @output
    @render.ui
    def d1_legend_ui():
        return ui.HTML(legend_html(d1_dim.get()))

    @output
    @render.ui
    def d1_plot_meta():
        sid = d1_sel.get()
        if sid is not None:
            row = df[df["id"] == sid]
            if not row.empty:
                return ui.div(ui.HTML(f'<span class="accent">●</span> {row.iloc[0]["name"]} selected'), class_="plot-meta")
        return ui.div("Hover a dot for details · click to expand", class_="plot-meta")

    @render_widget
    def d1_scatter():
        fig = go.Figure()
        sync_scatter(fig, d1_plot_df(), d1_sel.get(), d1_dim.get())
        return fig

    @output
    @render.ui
    def d1_modal_trigger():
        return ui.div()

    @output
    @render.text
    def hist_results_count():
        if HISTORICAL.empty:
            return "No historical data loaded"
        return f"{len(historical_filtered())} shown / {len(HISTORICAL)} historical player-seasons"

    @output
    @render.ui
    def historical_table_ui():
        rows = historical_filtered()
        if rows.empty:
            return ui.div("No historical players match those filters.", class_="historical-empty")
        body = []
        saved = tracker_ids.get()
        for _, row in rows.iterrows():
            row_id = str(row["season_player_id"])
            body.append(
                ui.tags.tr(
                    {"class": "is-selected" if row_id == historical_selected.get() else "", "onclick": f"Shiny.setInputValue('hist_select_row','{row_id}',{{priority:'event'}})"},
                    ui.tags.td(ui.div(str(row["player_name"]), class_="table-player"), ui.div(f"{row['team']} · {int(row['year'])}", class_="table-meta")),
                    ui.tags.td(str(row.get("conf", ""))),
                    ui.tags.td(str(row.get("pos", ""))),
                    ui.tags.td(str(row.get("archetype", ""))),
                    ui.tags.td(f"{_as_float(row.get('mins_per_game'), 0):.1f}"),
                    ui.tags.td(f"{_as_float(row.get('pts_per_game'), 0):.1f}"),
                    ui.tags.td(f"{_as_float(row.get('ast_per_game'), 0):.1f}"),
                    ui.tags.td(f"{_as_float(row.get('treb_per_game'), 0):.1f}"),
                    ui.tags.td(f"{_as_float(row.get('bpm'), 0):.1f}"),
                    ui.tags.td(ui.tags.button("Saved" if row_id in saved else "Save", class_="mini-btn", onclick=f"event.stopPropagation();Shiny.setInputValue('tracker_toggle','{row_id}',{{priority:'event'}})")),
                )
            )
        return ui.div({"class": "historical-table-card historical-results-table-card"}, ui.tags.table({"class": "historical-table"}, ui.tags.thead(ui.tags.tr(*[ui.tags.th(x) for x in ["Player", "Conf", "Pos", "Archetype", "MPG", "PPG", "APG", "RPG", "BPM", "Tracker"]])), ui.tags.tbody(*body)))

    @output
    @render.ui
    def historical_current_comps_ui():
        row = historical_row_by_id(historical_selected.get())
        if row is None:
            return ui.div("Select a historical player to load current 2026 comps.", class_="historical-empty")
        comps = historical_current_comps(row)
        cards = [
            ui.div(
                {"class": "comp-card", "onclick": f"Shiny.setInputValue('d1_select_similar','{comp['id']}',{{priority:'event'}})"},
                ui.div(f"{comp['rank']:02d}", class_="comp-rank"),
                ui.div(ui.div(comp["name"], class_="comp-name"), ui.div(f"{comp['team']} · {comp['cls']} · {comp['primary_archetype']}", class_="table-meta")),
                ui.div(f"{comp['similarity_score']:.0f}", class_="comp-score"),
            )
            for comp in comps
        ]
        return ui.div(
            {"class": "historical-comps-card"},
            ui.div(
                {"class": "historical-comps-head"},
                ui.div(
                    ui.div("Current players most like this profile", class_="historical-comps-title"),
                    ui.div(f"{row['player_name']} · {historical_profile_subtitle(row)}", class_="historical-comps-subtitle"),
                ),
            ),
            ui.div({"class": "historical-comp-list comp-grid"}, *cards),
        )

    @output
    @render.text
    def triton_results_count():
        total = len(triton_filtered())
        limit = input.triton_limit() or "100"
        shown = total if limit == "all" else min(total, int(limit))
        noun = "matching players" if total != shown else "players on the board"
        return f"{shown} of {total} {noun}" if total != shown else f"{shown} players on the board"

    @output
    @render.ui
    def triton_table_ui():
        rows = triton_filtered()
        limit = input.triton_limit() or "100"
        if limit != "all":
            rows = rows.head(int(limit))
        if rows.empty:
            return ui.div("No players match those Triton Zone filters.", class_="triton-empty")
        body = []
        for rank, (_, row) in enumerate(rows.iterrows(), start=1):
            cells = [
                ui.tags.td(
                    ui.span(
                        f"{_as_float(row.get(f'triton_val_{m['key']}'), 0):.1f}",
                        class_=f"triton-metric {'is-pass' if row.get(f'triton_ok_{m['key']}', False) else 'is-miss'}",
                    )
                )
                for m in TRITON_ZONE_METRICS
            ]
            war = max(0, min(100, _as_float(row.get("triton_war"), 0)))
            checks_passed = int(row["triton_checks_passed"])
            arch_tags = []
            for arch_key, arch_meta in TRITON_SPECIAL_ARCHETYPES.items():
                if bool(row.get(f"triton_is_{arch_key}", False)):
                    arch_tags.append(ui.span(arch_meta["label"], class_="triton-arch-tag"))
            body.append(
                ui.tags.tr(
                    {"onclick": f"Shiny.setInputValue('triton_open_player','{row['id']}',{{priority:'event'}})"},
                    ui.tags.td(str(rank)),
                    ui.tags.td(ui.div(row["name"], class_="table-player"), ui.div(f"{row['team']} · {row['cls']} · {row['pos']}", class_="table-meta")),
                    ui.tags.td(row.get("confName", "")),
                    ui.tags.td(height_str(_as_float(row.get("heightIn"), 0))),
                    ui.tags.td(f"{_as_float(row.get('mpg'), 0):.1f}"),
                    ui.tags.td(ui.div(f"{war:.0f}", class_="triton-war-value"), ui.div({"class": "triton-war-track"}, ui.div({"class": "triton-war-fill", "style": f"width:{war:.1f}%"}))),
                    ui.tags.td(ui.span(f"{checks_passed}/{len(TRITON_ZONE_METRICS)}", class_=f"triton-zone-badge {'is-full' if checks_passed >= len(TRITON_ZONE_METRICS) else ''}")),
                    *cells,
                    ui.tags.td(ui.div({"class": "triton-arch-tags"}, *(arch_tags or [ui.span("-", class_="triton-arch-tag is-empty")]))),
                )
            )
        headers = ["#", "Player", "Conference", "Ht", "MPG", "Triton Zone ↓", "Zone", *[m["label"] for m in TRITON_ZONE_METRICS], "Archetype"]
        return ui.div({"class": "triton-table-card"}, ui.tags.table({"class": "triton-table"}, ui.tags.thead(ui.tags.tr(*[ui.tags.th(x) for x in headers])), ui.tags.tbody(*body)))

    @output
    @render.ui
    def tracker_ui():
        return make_tracker_content(tracker_ids.get())

    @output
    @render.ui
    def lineup_ui():
        rows = [r for _, r in watchlist_rows(watchlist.get())]
        if not rows:
            return ui.div("Add current players to the watchlist to sketch lineup combinations against the UCSD roster reference.", class_="historical-empty")
        pool = pd.DataFrame(rows)
        pool = pool.assign(lineup_score=pool["triton_war"].fillna(0) + pool["bpm"].fillna(0) * 2 + pool["primary_score"].fillna(0) * .25)
        guards = pool[pool["pos"].isin(["G", "G/F"])].sort_values("lineup_score", ascending=False).head(2)
        wings = pool[pool["pos"].isin(["G/F", "F"])].drop(guards.index, errors="ignore").sort_values("lineup_score", ascending=False).head(2)
        bigs = pool[pool["pos"].isin(["F/C", "C"])].drop(guards.index.union(wings.index), errors="ignore").sort_values("lineup_score", ascending=False).head(1)
        chosen = pd.concat([guards, wings, bigs]).drop_duplicates(subset=["id"]).head(5)
        cards = [
            ui.div({"class": "lineup-card"}, ui.div(row["name"], class_="comp-name"), ui.div(f"{row['team']} · {row['pos']} · {row['primary_archetype']}", class_="table-meta"), ui.div(f"Triton WAR {row['triton_war']:.0f} · BPM {row['bpm']:.1f}", class_="table-meta"))
            for _, row in chosen.iterrows()
        ]
        return ui.div(ui.div({"class": "lineup-grid"}, *cards), ui.div(f"Roster reference: {UCSD_WBB_ROSTER_URL}", class_="beta-note"))

    @output
    @render.text
    def wl_count():
        n = len(watchlist.get())
        return f"{n} player{'s' if n != 1 else ''}"

    @reactive.effect
    @reactive.event(input.wl_radar_player_1, input.wl_radar_player_2)
    def _wl_radar_players_changed():
        available = [pid for pid, *_ in watchlist_rows(watchlist.get())]
        selected = []
        for pid in (input.wl_radar_player_1(), input.wl_radar_player_2()):
            if pid and pid in available and pid not in selected:
                selected.append(pid)
        radar_selected.set(selected[:2])

    @reactive.effect
    @reactive.event(input.wl_radar_stats)
    def _wl_radar_stats_changed():
        selected = [key for key in list(input.wl_radar_stats() or []) if key in RADAR_STAT_LOOKUP]
        radar_stat_selected.set(selected)

    @output
    @render.ui
    def wl_radar_picker():
        rows = watchlist_rows(watchlist.get())
        if not rows:
            return ui.div({"class": "wl-radar-picker"})
        selected = [pid for pid in radar_selected.get() if pid in {row[0] for row in rows}][:2]
        player_choices = {"": "Select player...", **{pid: r["name"] for pid, r in rows}}
        stat_selected = [key for key in radar_stat_selected.get() if key in RADAR_STAT_LOOKUP]
        stat_choices = {key: label for key, label, _col, _short_label, _fmt in RADAR_STATS}
        return ui.div(
            {"class": "wl-radar-picker"},
            ui.div({"class": "wl-radar-field"}, ui.div("Player 1", class_="wl-radar-field-title"), ui.input_selectize("wl_radar_player_1", None, choices=player_choices, selected=selected[0] if len(selected) >= 1 else "", options={"placeholder": "Search player 1..."})),
            ui.div({"class": "wl-radar-field"}, ui.div("Player 2", class_="wl-radar-field-title"), ui.input_selectize("wl_radar_player_2", None, choices=player_choices, selected=selected[1] if len(selected) >= 2 else "", options={"placeholder": "Search player 2..."})),
            ui.div({"class": "wl-radar-field wl-radar-stat-checks"}, ui.div("Stats", class_="wl-radar-field-title"), ui.input_checkbox_group("wl_radar_stats", None, choices=stat_choices, selected=stat_selected)),
        )

    @output
    @render.ui
    def watchlist_ui():
        wl = watchlist.get()
        if not wl:
            return ui.div(
                ui.tags.script("var b=document.getElementById('wl-badge');if(b){b.style.display='none';}"),
                ui.div({"class": "wl-empty"}, ui.div("☆", class_="wl-star"), ui.div("No players starred yet."), ui.div("Open any player profile and click ☆ to add them here.", style="color:var(--ink-3);max-width:280px;text-align:center;line-height:1.5")),
            )
        cards = []
        for pid, r in watchlist_rows(wl):
            pc = position_color(r.get("pos", ""))
            open_js = f"Shiny.setInputValue('wl_open_player','{pid}',{{priority:'event'}})"
            cards.append(
                ui.div(
                    {"class": "wl-card", "onclick": open_js},
                    ui.tags.button({"class": "wl-remove", "title": "Remove from watchlist", "onclick": f"event.stopPropagation();Shiny.setInputValue('toggle_watchlist','{pid}',{{priority:'event'}})"}, "★"),
                    ui.div(r["name"], class_="wl-card-name"),
                    ui.div(ui.span(position_label(r["pos"]), class_="pos-badge", style=f"color:{pc};border-color:{pc}"), ui.span(r["team"]), ui.span(f"· {r['cls']} · WBB D-I", style="color:var(--ink-3)"), class_="wl-card-meta"),
                    ui.div({"class": "wl-card-stats"}, ui.div(ui.div(f"{r['ppg']:.1f}", class_="n"), ui.div("PPG", class_="l"), class_="wl-stat"), ui.div(ui.div(f"{r['rpg']:.1f}", class_="n"), ui.div("RPG", class_="l"), class_="wl-stat"), ui.div(ui.div(f"{r['apg']:.1f}", class_="n"), ui.div("APG", class_="l"), class_="wl-stat"), ui.div(ui.div(f"{r['fg']*100:.0f}%", class_="n"), ui.div("FG%", class_="l"), class_="wl-stat")),
                )
            )
        n = len(wl)
        vis = "inline-block" if n else "none"
        js = f"var b=document.getElementById('wl-badge');if(b){{b.textContent='{n}';b.style.display='{vis}';}}"
        return ui.div(ui.tags.script(js), ui.div({"class": "wl-grid"}, *cards))

    @output
    @render_widget
    def watchlist_radar():
        selected = [pid for pid in radar_selected.get() if pid in watchlist.get()][:2]
        stats = [key for key in radar_stat_selected.get() if key in RADAR_STAT_LOOKUP]
        return make_watchlist_radar(selected, stats)


app = App(app_ui, server, static_assets=HERE / "www")

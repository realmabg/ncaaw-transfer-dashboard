from pathlib import Path
import json

import asttokens  # noqa: F401 - direct import lets Shinylive install this transitive dependency.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def dataset_status_text() -> str:
    if DATA["source_status"] == "loaded":
        return DATA["source_path"]
    return "No processed women's Division I dataset found yet."


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
    pc = position_color(row.get("pos", ""))
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
        stat_box("FG%", f"{row['fg']*100:.1f}", league_avg_map["fg"] * 100),
        stat_box("3P%", f"{row['tp']*100:.1f}", league_avg_map["tp"] * 100),
    ]
    bpm_value = pd.to_numeric(pd.Series([row.get("bpm", np.nan)]), errors="coerce").iloc[0]
    if pd.notna(bpm_value):
        statline.append(stat_box("BPM", f"{bpm_value:.1f}", 0))
    bars = [
        bar_row("PPG", row["ppg"], league_avg_map["ppg"], 30),
        bar_row("RPG", row["rpg"], league_avg_map["rpg"], 14),
        bar_row("APG", row["apg"], league_avg_map["apg"], 12),
        bar_row("SPG", row["spg"], league_avg_map["spg"], 4),
        bar_row("BPG", row["bpg"], league_avg_map["bpg"], 4),
        bar_row("3P%", row["tp"], league_avg_map["tp"], 0.55, lambda v: f"{v*100:.1f}%"),
        bar_row("TS%", row["ts"], league_avg_map["ts"], 0.75, lambda v: f"{v*100:.1f}%"),
    ]
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

    body = ui.div(
        {"id": "detail-body"},
        ui.div(
            {"class": "detail-col"},
            ui.div({"class": "player-name-row"}, ui.div(row["name"], class_="player-name"), ui.tags.button({"class": "star-btn", "title": star_label, "style": star_style, "onclick": f"Shiny.setInputValue('toggle_watchlist','{player_id}',{{priority:'event'}})"}, star_icon)),
            ui.div(ui.span({"class": "team-dot", "style": f"background:{pc}"}), f"{row['team']} · {row['confName']}", class_="player-team"),
            ui.div({"class": "bio-grid"}, bio_item("Division", "WBB D-I"), bio_item("Position", position_label(row["pos"])), bio_item("Class", row["cls"]), bio_item("Eligibility Used", str(int(row["eligibility"])), mono=True), bio_item("Height", height_str(int(row["heightIn"])), mono=True), bio_item("Games", str(int(row["gp"])), mono=True), bio_item("Min/G", f"{row['mpg']:.1f}", mono=True), bio_item("BPM", f"{bpm_value:.1f}" if pd.notna(bpm_value) else "N/A", mono=True)),
        ),
        ui.div({"class": "detail-col"}, ui.div("Season Statline ", ui.span("2025–26", class_="sub"), class_="col-title"), ui.div({"class": "statline"}, *statline), ui.div("vs. League Average ", ui.span("unweighted mean, all WBB D-I players", class_="sub"), class_="col-title"), *bars, ui.div(ui.tags.b("Bar", style="color:var(--ink-2)"), " = player value.  ", ui.tags.b("Tick", style="color:var(--ink-2)"), " = league mean.", class_="bar-note")),
        ui.div({"class": "detail-col"}, ui.div("Most Similar Players ", ui.span(SIMILARITY_METRIC_LABELS[similarity_metric], class_="sub"), class_="col-title"), ui.div(ui.input_radio_buttons("modal_similarity_metric", None, choices={"mahalanobis": "Mahalanobis", "euclidean": "Euclidean"}, selected=similarity_metric, inline=True), class_="sim-metric-control"), *sim_rows),
    )
    return ui.modal(body, title=ui.HTML(f"Player Profile <b>· {row['name']}</b> <span class='div-badge'>WBB D-I</span>"), easy_close=True, size="xl", footer=None)


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
                document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active-d1','active-wl'); });
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
            ui.div({"class": "mast-left"}, ui.div(ui.HTML("NCAA Women's Basketball <span class='dot'></span> 2025–26"), class_="kicker"), ui.div(ui.HTML("Player <em>Dashboard</em>"), class_="atlas-title"), ui.div("Women's Division I transfer dashboard.", class_="dek"), ui.div(f"Dataset: {dataset_status_text()}", class_="byline")),
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
            ui.div({"class": "tab-sep"}),
            ui.tags.button(ui.HTML('Watchlist <span id="wl-badge" class="wl-badge" style="display:none">0</span>'), id="btn-wl", class_="tab-btn", onclick="switchTab('wl')"),
        ),
        ui.div(
            {"id": "tab-content"},
            ui.div({"id": "d1-tab", "class": "tab-panel active"}, ui.div({"class": "body-grid"}, make_sidebar("d1", df, conferences), make_plot_area("d1"))),
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
        ui.div({"id": "site-footer"}, "Women’s Division I source of truth · adapted from the men’s product shell"),
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

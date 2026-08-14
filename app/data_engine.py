"""Utilities for loading the women's Division I processed player dataset.

This repo treats women's Division I basketball data as the source of truth.
The copied men's files are methodology references only, so the app boot path
must not depend on men's CSV names, division splits, or recruiting/portal
enrichments.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

POS_COLOR = {
    "G": "#4a9eed",
    "G/F": "#5ab87a",
    "F": "#e8963a",
    "F/C": "#c46ab0",
    "C": "#9a7adc",
}

POS_LABEL = {
    "G": "Guard",
    "G/F": "Guard / Forward",
    "F": "Forward",
    "F/C": "Forward / Center",
    "C": "Center",
}

POSITIONS = ["G", "G/F", "F", "F/C", "C"]
CLASSES = ["R", "FR", "SO", "JR", "SR", "GR"]
SIMILARITY_COLUMNS = [f"arch_pca_PC{i}" for i in range(1, 7)]
ARCHETYPE_SCORE_COLUMNS = [
    "score_pg_combo",
    "score_wing_2_4",
    "score_stretch_big",
]
ARCHETYPE_METADATA_COLUMNS = [
    "primary_score_col",
    "primary_score",
    "meets_pg_preferred",
    "meets_wing_preferred",
    "meets_big_preferred",
]
ARCHETYPE_LABELS = {
    "score_pg_combo": "PG / Combo Guard",
    "score_wing_2_4": "2-4 Wing",
    "score_stretch_big": "Stretch Big",
}
OPTIONAL_TAG_COLUMNS = [
    "transfer_available",
    "transfer_status",
    "transfer_from",
    "transfer_to",
    "recruiting_summary",
]

DEFAULT_DATASET_CANDIDATES = [
    Path("wbb_d1_processed_players.csv"),
    Path("wbb_processed_players.csv"),
    Path("data/processed/wbb_d1_processed_players.csv"),
    Path("data/processed/wbb_processed_players.csv"),
    Path("data/processed/wbb_d1_player_profiles.csv"),
    Path("data/processed/wbb_player_profiles.csv"),
    Path("data/interim/wbb_d1_processed_players.csv"),
    Path("data/interim/wbb_processed_players.csv"),
    Path("app/wbb_d1_processed_players.csv"),
    Path("app/wbb_processed_players.csv"),
]


def height_str(inches: int | float) -> str:
    value = int(pd.to_numeric(pd.Series([inches]), errors="coerce").fillna(72).iloc[0])
    feet = value // 12
    rem = value % 12
    return f"{feet}'{rem}\""


def clean_text_series(series, default: str = "") -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    return series.fillna(default).astype(str).str.strip()


def match_key(value: str) -> str:
    value = "" if pd.isna(value) else str(value)
    value = value.lower().strip()
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def normalize_class(value: str) -> str:
    if pd.isna(value):
        return "SR"
    text = str(value).lower().replace(".", "").strip()
    if text.startswith("gr") or "graduate" in text:
        return "GR"
    if text.startswith("fr") or "fresh" in text:
        return "FR"
    if text.startswith("so") or "soph" in text:
        return "SO"
    if text.startswith("jr") or "jun" in text:
        return "JR"
    if text.startswith("sr") or "sen" in text:
        return "SR"
    if text.startswith("r"):
        return "R"
    return "SR"


def eligibility_used(value: str) -> int:
    normalized = normalize_class(value)
    mapping = {"R": 1, "FR": 1, "SO": 2, "JR": 3, "SR": 4, "GR": 5}
    return mapping.get(normalized, 4)


def refine_position(raw: str) -> str:
    text = ("" if pd.isna(raw) else str(raw)).strip().upper()
    direct = {
        "G": "G",
        "G/F": "G/F",
        "GF": "G/F",
        "F": "F",
        "F/C": "F/C",
        "FC": "F/C",
        "C": "C",
        "PG": "G",
        "SG": "G",
        "CG": "G",
        "WG": "G/F",
        "SF": "F",
        "PF": "F/C",
        "POST": "C",
        "PURE PG": "G",
        "SCORING PG": "G",
        "COMBO G": "G",
        "WING G": "G/F",
        "WING F": "F",
        "STRETCH 4": "F/C",
        "PF/C": "F/C",
    }
    if text in direct:
        return direct[text]
    if text.startswith(("G/F", "GF")):
        return "G/F"
    if text.startswith(("F/C", "FC", "PF")):
        return "F/C"
    if text.startswith(("G", "PG", "SG")):
        return "G"
    if text.startswith(("F", "SF")):
        return "F"
    if text.startswith(("C", "POST")):
        return "C"
    return "G"


def parse_height_inches(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if number > 24 else np.nan
    text = str(value).strip().lower().replace(" ", "")
    match = re.match(r"^(\d+)[-'](\d+)$", text)
    if match:
        feet, inches = match.groups()
        return int(feet) * 12 + int(inches)
    match = re.match(r"^(\d+)ft(\d+)$", text)
    if match:
        feet, inches = match.groups()
        return int(feet) * 12 + int(inches)
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and numeric > 24:
        return float(numeric)
    return np.nan


def normalize_pct_series(series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return values.where(values <= 1.0, values / 100.0)


def empty_processed_frame() -> pd.DataFrame:
    columns = {
        "id": pd.Series(dtype="object"),
        "name": pd.Series(dtype="object"),
        "team": pd.Series(dtype="object"),
        "conf": pd.Series(dtype="object"),
        "confName": pd.Series(dtype="object"),
        "pos": pd.Series(dtype="object"),
        "cls": pd.Series(dtype="object"),
        "eligibility": pd.Series(dtype="int64"),
        "heightIn": pd.Series(dtype="float64"),
        "gp": pd.Series(dtype="float64"),
        "mpg": pd.Series(dtype="float64"),
        "ppg": pd.Series(dtype="float64"),
        "rpg": pd.Series(dtype="float64"),
        "apg": pd.Series(dtype="float64"),
        "spg": pd.Series(dtype="float64"),
        "bpg": pd.Series(dtype="float64"),
        "tov": pd.Series(dtype="float64"),
        "fg": pd.Series(dtype="float64"),
        "tp": pd.Series(dtype="float64"),
        "ft": pd.Series(dtype="float64"),
        "ts": pd.Series(dtype="float64"),
        "efg": pd.Series(dtype="float64"),
        "usg": pd.Series(dtype="float64"),
        "three_share": pd.Series(dtype="float64"),
        "ast_tov": pd.Series(dtype="float64"),
        "pts_per_40": pd.Series(dtype="float64"),
        "reb_per_40": pd.Series(dtype="float64"),
        "ast_per_40": pd.Series(dtype="float64"),
        "stl_per_40": pd.Series(dtype="float64"),
        "blk_per_40": pd.Series(dtype="float64"),
        "tov_per_40": pd.Series(dtype="float64"),
        "assist_creation": pd.Series(dtype="float64"),
        "dreb_arch": pd.Series(dtype="float64"),
        "orb_pct": pd.Series(dtype="float64"),
        "drb_pct": pd.Series(dtype="float64"),
        "dreb_source": pd.Series(dtype="object"),
        "bpm": pd.Series(dtype="float64"),
        "porpag": pd.Series(dtype="float64"),
        "primary_score_col": pd.Series(dtype="object"),
        "primary_score": pd.Series(dtype="float64"),
        "primary_archetype": pd.Series(dtype="object"),
    }
    for col in SIMILARITY_COLUMNS + ARCHETYPE_SCORE_COLUMNS + OPTIONAL_TAG_COLUMNS:
        columns[col] = pd.Series(dtype="float64" if "score_" in col or "arch_pca" in col else "object")
    columns["meets_pg_preferred"] = pd.Series(dtype="bool")
    columns["meets_wing_preferred"] = pd.Series(dtype="bool")
    columns["meets_big_preferred"] = pd.Series(dtype="bool")
    columns["transfer_available"] = pd.Series(dtype="bool")
    return pd.DataFrame(columns)


def resolve_default_wbb_dataset_path(base_dir: Path | None = None) -> Path | None:
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    for rel_path in DEFAULT_DATASET_CANDIDATES:
        candidate = root / rel_path
        if candidate.exists():
            return candidate
    return None


def _first_present(raw: pd.DataFrame, candidates: list[str], default=None):
    for col in candidates:
        if col in raw.columns:
            return raw[col]
    return default


def _numeric(raw: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    source = _first_present(raw, candidates)
    if source is None:
        return pd.Series(default, index=raw.index, dtype="float64")
    return pd.to_numeric(source, errors="coerce").fillna(default)


def _text(raw: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    source = _first_present(raw, candidates)
    if source is None:
        return pd.Series(default, index=raw.index, dtype="object")
    return clean_text_series(source, default=default)


def _bool(raw: pd.DataFrame, candidates: list[str]) -> pd.Series:
    source = _first_present(raw, candidates)
    if source is None:
        return pd.Series(False, index=raw.index, dtype="bool")
    text = clean_text_series(source).str.lower()
    return text.isin(["1", "true", "yes", "y", "available"])


def _pick_archetype(df: pd.DataFrame) -> pd.Series:
    existing = clean_text_series(df.get("primary_archetype"), default="")
    if existing.str.len().gt(0).any():
        return existing.where(existing.str.len() > 0, "Unassigned")
    scores = df[ARCHETYPE_SCORE_COLUMNS].copy()
    if scores.notna().any().any():
        numeric_scores = scores.apply(pd.to_numeric, errors="coerce")
        has_signal = numeric_scores.fillna(0).abs().sum(axis=1) > 0
        picked = numeric_scores.idxmax(axis=1).map(ARCHETYPE_LABELS).fillna("Unassigned")
        return picked.where(has_signal, "Unassigned")
    return pd.Series("Unassigned", index=df.index, dtype="object")


def _normalize_position(raw: pd.DataFrame) -> pd.Series:
    source = _first_present(raw, ["pos", "position"])
    if source is None:
        return pd.Series("", index=raw.index, dtype="object")
    normalized = clean_text_series(source, default="").apply(refine_position)
    return normalized.where(clean_text_series(source, default="").str.len() > 0, "")


def _normalize_frame(raw: pd.DataFrame, id_prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    raw_player_id = _first_present(raw, ["id", "player_id"])
    if raw_player_id is None:
        df["_source_id"] = [f"{id_prefix}{i}" for i in range(len(raw))]
    else:
        ids = clean_text_series(raw_player_id, default="")
        fallback_ids = pd.Series([f"{id_prefix}{i}" for i in range(len(ids))], index=ids.index)
        df["_source_id"] = ids.where(ids.str.len() > 0, fallback_ids)
    df["name"] = _text(raw, ["name", "player_name", "player"])
    df["team"] = _text(raw, ["team", "school"])
    df["conf"] = _text(raw, ["conf", "conference_abbr", "conference"])
    df["confName"] = _text(raw, ["confName", "conference_full", "conference"], default="")
    df["confName"] = df["confName"].where(df["confName"].str.len() > 0, df["conf"])
    df["pos"] = _normalize_position(raw)
    class_series = _text(raw, ["cls", "class", "yr", "year"])
    df["cls"] = class_series.apply(normalize_class)
    df["eligibility"] = _numeric(raw, ["eligibility"], default=np.nan)
    df["eligibility"] = df["eligibility"].where(df["eligibility"] > 0, class_series.apply(eligibility_used)).astype(int)
    df["heightIn"] = _first_present(raw, ["heightIn", "height_inches", "height"])
    if df["heightIn"] is None:
        df["heightIn"] = pd.Series(np.nan, index=raw.index)
    df["heightIn"] = df["heightIn"].apply(parse_height_inches)
    df["heightIn"] = df["heightIn"].fillna(df["heightIn"].median() if df["heightIn"].notna().any() else 72).round()
    df["gp"] = _numeric(raw, ["gp", "GP", "games_played"])
    df["mpg"] = _numeric(raw, ["mpg", "mins_per_game", "minutes_per_game"])
    df["ppg"] = _numeric(raw, ["ppg", "pts_per_game", "points_per_game"])
    df["rpg"] = _numeric(raw, ["rpg", "reb_per_game", "treb_per_game"])
    df["apg"] = _numeric(raw, ["apg", "ast_per_game"])
    df["spg"] = _numeric(raw, ["spg", "stl_per_game"])
    df["bpg"] = _numeric(raw, ["bpg", "blk_per_game"])
    df["tov"] = _numeric(raw, ["tov", "tov_per_game", "turnovers_per_game"])
    df["fg"] = normalize_pct_series(_first_present(raw, ["fg", "fg_pct", "FG_pct", "FG%"]))
    df["tp"] = normalize_pct_series(_first_present(raw, ["tp", "3P_pct", "3p_pct", "3P%", "3p%"]))
    df["ft"] = normalize_pct_series(_first_present(raw, ["ft", "FT_pct", "ft_pct", "FT%"]))
    df["ts"] = normalize_pct_series(_first_present(raw, ["ts", "TS_pct", "ts_pct"]))
    df["efg"] = normalize_pct_series(_first_present(raw, ["efg", "eFG", "eFG_pct", "efg_pct"]))
    df["usg"] = normalize_pct_series(_first_present(raw, ["usg", "usage", "usage_pct"]))
    df["three_share"] = normalize_pct_series(_first_present(raw, ["three_share", "3pr", "three_point_rate"]))
    df["ast_tov"] = _numeric(raw, ["ast_tov", "AST_TOV", "ast_to_ratio"])
    pace_to_40 = 40.0 / df["mpg"].replace(0, np.nan)
    df["pts_per_40"] = _numeric(raw, ["pts_per_40"], default=np.nan).fillna(df["ppg"] * pace_to_40).fillna(0)
    df["reb_per_40"] = _numeric(raw, ["reb_per_40"], default=np.nan).fillna(df["rpg"] * pace_to_40).fillna(0)
    df["ast_per_40"] = _numeric(raw, ["ast_per_40"], default=np.nan).fillna(df["apg"] * pace_to_40).fillna(0)
    df["stl_per_40"] = _numeric(raw, ["stl_per_40"], default=np.nan).fillna(df["spg"] * pace_to_40).fillna(0)
    df["blk_per_40"] = _numeric(raw, ["blk_per_40"], default=np.nan).fillna(df["bpg"] * pace_to_40).fillna(0)
    df["tov_per_40"] = _numeric(raw, ["tov_per_40"], default=np.nan).fillna(df["tov"] * pace_to_40).fillna(0)
    df["assist_creation"] = _numeric(raw, ["assist_creation", "AST_pct"], default=np.nan).fillna(df["apg"])
    df["dreb_arch"] = _numeric(raw, ["dreb_arch", "DRB_pct", "drb", "dreb_per_game"], default=np.nan).fillna(df["rpg"])
    df["orb_pct"] = _numeric(raw, ["orb_pct", "ORB_pct"], default=0.0)
    df["drb_pct"] = _numeric(raw, ["drb_pct", "DRB_pct"], default=np.nan).fillna(df["dreb_arch"])
    df["dreb_source"] = _text(raw, ["dreb_source"], default="processed")
    df["bpm"] = _numeric(raw, ["bpm", "BPM"], default=np.nan)
    df["porpag"] = _numeric(raw, ["porpag", "PORPAG"], default=np.nan)
    for i, col in enumerate(SIMILARITY_COLUMNS, start=1):
        df[col] = _numeric(raw, [col, f"PC{i}", f"arch_PC{i}"], default=0.0)
    for col in ARCHETYPE_SCORE_COLUMNS:
        df[col] = _numeric(raw, [col], default=0.0)
    df["primary_score_col"] = _text(raw, ["primary_score_col"], default="")
    df["primary_score"] = _numeric(raw, ["primary_score"], default=np.nan).fillna(df[ARCHETYPE_SCORE_COLUMNS].max(axis=1))
    df["meets_pg_preferred"] = _bool(raw, ["meets_pg_preferred"])
    df["meets_wing_preferred"] = _bool(raw, ["meets_wing_preferred"])
    df["meets_big_preferred"] = _bool(raw, ["meets_big_preferred"])
    df["transfer_available"] = _bool(raw, ["transfer_available"])
    df["transfer_status"] = _text(raw, ["transfer_status"], default="")
    df["transfer_from"] = _text(raw, ["transfer_from"], default="")
    df["transfer_to"] = _text(raw, ["transfer_to"], default="")
    df["recruiting_summary"] = _text(raw, ["recruiting_summary"], default="")
    df["primary_archetype"] = _pick_archetype(df)
    df = df[df["name"].str.len() > 0].copy().reset_index(drop=True)
    df["id"] = clean_text_series(df.pop("_source_id"), default="")
    return df


def _build_output(df: pd.DataFrame, source_path: Path | None) -> dict:
    if df.empty:
        empty = empty_processed_frame()
        return {
            "df": empty,
            "conferences": [],
            "teams": [],
            "archetypes": [],
            "league_avg": {key: 0.0 for key in ["mpg", "ppg", "rpg", "apg", "spg", "bpg", "fg", "tp", "ft", "ts"]},
            "similar_to": lambda player_id, n_sim=5, metric="euclidean": [],
            "source_path": str(source_path) if source_path else "",
            "source_status": "missing" if source_path is None else "empty",
        }

    conferences = (
        df[["conf", "confName"]]
        .drop_duplicates()
        .sort_values(["confName", "conf"])
        .to_dict("records")
    )
    teams = sorted(df["team"].dropna().astype(str).unique().tolist())
    archetypes = sorted(df["primary_archetype"].dropna().astype(str).unique().tolist())
    league_avg = {key: float(pd.to_numeric(df[key], errors="coerce").fillna(0).mean()) for key in ["mpg", "ppg", "rpg", "apg", "spg", "bpg", "fg", "tp", "ft", "ts"]}

    def similar_to(player_id: str, n_sim: int = 5, metric: str = "euclidean"):
        idx = df.index[df["id"] == player_id]
        if len(idx) == 0:
            return []
        frame = df[SIMILARITY_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        matrix = frame.to_numpy(dtype=float)
        target = matrix[idx[0]].reshape(1, -1)
        if metric == "mahalanobis" and len(df) > 1:
            cov = np.cov(matrix, rowvar=False)
            cov += np.eye(matrix.shape[1]) * 1e-6
            vi = np.linalg.inv(cov)
            dists = cdist(target, matrix, metric="mahalanobis", VI=vi).flatten()
        else:
            dists = cdist(target, matrix, metric="euclidean").flatten()
        dists[idx[0]] = np.inf
        order = np.argsort(dists)[:n_sim]
        output = []
        finite = dists[np.isfinite(dists)]
        finite_sorted = np.sort(finite) if finite.size else np.array([], dtype=float)
        benchmark_idx = min(9, finite_sorted.size - 1) if finite_sorted.size else -1
        benchmark_dist = float(finite_sorted[benchmark_idx]) if benchmark_idx >= 0 else 1.0
        benchmark_dist = benchmark_dist if benchmark_dist > 1e-9 else 1.0
        for row_idx in order:
            row = df.iloc[row_idx]
            dist = float(dists[row_idx])
            score = 100.0 * float(np.exp(-np.log(2.0) * dist / benchmark_dist))
            output.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "team": row["team"],
                    "conf": row["confName"],
                    "pos": row["pos"],
                    "cls": row["cls"],
                    "similarity_score": float(max(0.0, min(100.0, score))),
                    "distance": dist,
                }
            )
        return output

    return {
        "df": df,
        "conferences": conferences,
        "teams": teams,
        "archetypes": archetypes,
        "league_avg": league_avg,
        "similar_to": similar_to,
        "source_path": str(source_path) if source_path else "",
        "source_status": "loaded",
    }


def load_wbb_dataset(csv_path: str | Path | None = None, id_prefix: str = "wbbp") -> dict:
    path = Path(csv_path) if csv_path is not None else resolve_default_wbb_dataset_path()
    if path is None or not path.exists():
        return _build_output(empty_processed_frame(), None)
    raw = pd.read_csv(path)
    df = _normalize_frame(raw, id_prefix=id_prefix)
    return _build_output(df, path)

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


UNKNOWN_YEAR_VALUES = {"", "--", "na", "nan", "none", "null"}
ROLE_TO_POS = {
    "Pure PG": "G",
    "Scoring PG": "G",
    "Combo G": "G",
    "Wing G": "G/F",
    "Wing F": "F",
    "Stretch 4": "F/C",
    "PF/C": "F/C",
    "C": "C",
}
ARCHETYPE_LABELS = {
    "score_pg_combo": "PG / Combo Guard",
    "score_wing_2_4": "2-4 Wing",
    "score_stretch_big": "Stretch Big",
}
THRESHOLDS = {
    "baseline_efg": 0.500,
    "baseline_3p_pct": 0.300,
    "pg_3p_pct": 0.330,
    "pg_3p_rate": 0.300,
    "wing_3p_pct": 0.330,
    "wing_3p_rate": 0.300,
    "big_3p_pct": 0.300,
    "big_3p_rate": 0.250,
    "positive_ast_tov": 1.000,
    "high_percentile": 70.0,
    "d1_dreb_rate": 15.0,
}
PCA_FEATURES = [
    "pts_per_40",
    "ts",
    "usg",
    "three_share",
    "ft_rate",
    "ast_per_40",
    "ast_tov",
    "tov_per_40",
    "orb_pct",
    "drb_pct",
    "stl_per_40",
    "blk_per_40",
    "heightIn",
]


def clean_text(series: pd.Series, default: str = "") -> pd.Series:
    return series.fillna(default).astype(str).str.strip()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_pct(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.where(values <= 1.0, values / 100.0)


def normalize_class(value: str) -> str:
    if pd.isna(value):
        return "SR"
    text = str(value).lower().replace(".", "").strip()
    if text in UNKNOWN_YEAR_VALUES:
        return "SR"
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


def eligibility_used(cls_value: str) -> int:
    mapping = {"R": 1, "FR": 1, "SO": 2, "JR": 3, "SR": 4, "GR": 5}
    return mapping.get(cls_value, 4)


def parse_height_inches(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if number > 24 else np.nan
    text = str(value).strip().lower().replace(" ", "")
    if "'" in text or "-" in text:
        parts = text.replace("'", "-").split("-")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            return int(parts[0]) * 12 + int(parts[1])
    numeric_value = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric_value) and numeric_value > 24:
        return float(numeric_value)
    return np.nan


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_values = numeric(den).replace(0, np.nan)
    return numeric(num).div(den_values)


def map_role_to_pos(value: str) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return ROLE_TO_POS.get(text, "")


def percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average") * 100.0


def percentile_by_pos(df: pd.DataFrame, value_col: str, pos_col: str = "pos") -> pd.Series:
    return df.groupby(pos_col)[value_col].rank(pct=True, method="average") * 100.0


def bool_bonus(flag: pd.Series, points: float) -> pd.Series:
    return np.where(flag.fillna(False), points, 0.0)


def add_pca_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    feature_frame = df[PCA_FEATURES].apply(pd.to_numeric, errors="coerce")
    filled = feature_frame.fillna(feature_frame.median())
    x_raw = filled.to_numpy(dtype=float)
    std = np.where(x_raw.std(axis=0) == 0, 1, x_raw.std(axis=0))
    x = (x_raw - x_raw.mean(axis=0)) / std
    n_components = min(10, x.shape[1], x.shape[0])
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(x)
    vt = pca.components_.copy()

    if vt[0, PCA_FEATURES.index("heightIn")] < 0:
        coords[:, 0] *= -1
        vt[0, :] *= -1
    if n_components > 1:
        pc2_creator_polarity = (
            vt[1, PCA_FEATURES.index("ast_per_40")]
            + vt[1, PCA_FEATURES.index("ast_tov")]
            + 0.5 * vt[1, PCA_FEATURES.index("pts_per_40")]
            - vt[1, PCA_FEATURES.index("heightIn")]
            - vt[1, PCA_FEATURES.index("blk_per_40")]
        )
        if pc2_creator_polarity > 0:
            coords[:, 1] *= -1
            vt[1, :] *= -1

    out = df.copy()
    for i in range(n_components):
        out[f"arch_pca_PC{i+1}"] = coords[:, i]

    loadings = pd.DataFrame(
        vt[:n_components].T,
        index=PCA_FEATURES,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
    )
    return out, {
        "features": PCA_FEATURES,
        "explained_ratio": pca.explained_variance_ratio_,
        "loadings": loadings,
    }


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    out["id"] = clean_text(raw["player_id"], default="")
    fallback_ids = pd.Series([f"wbbp{i}" for i in range(len(out))], index=out.index)
    out["id"] = out["id"].where(out["id"].str.len() > 0, fallback_ids)
    out["name"] = clean_text(raw["name"], default="")
    out["team"] = clean_text(raw["team"], default="")
    out["conf"] = clean_text(raw["conf"], default="")
    out["confName"] = out["conf"]
    out["pos"] = clean_text(raw["role"], default="").apply(map_role_to_pos)

    year_text = clean_text(raw["yr"], default="")
    year_normalized = year_text.str.lower()
    review_needed = year_normalized.isin(UNKNOWN_YEAR_VALUES)
    out["cls"] = year_text.apply(normalize_class)
    out["eligibility"] = out["cls"].apply(eligibility_used)
    out["class_year_review_needed"] = review_needed
    out["class_year_review_note"] = np.where(review_needed, "Unknown source class year; manual review required.", "")

    out["heightIn"] = raw["ht"].apply(parse_height_inches)
    height_fill = out["heightIn"].median() if out["heightIn"].notna().any() else 72
    out["heightIn"] = out["heightIn"].fillna(height_fill).round()

    out["gp"] = numeric(raw["GP"]).fillna(0)
    out["mpg"] = numeric(raw["minutes_per_game"]).fillna(0)
    out["ppg"] = numeric(raw["pts_per_game"]).fillna(0)
    out["rpg"] = numeric(raw["treb_per_game"]).fillna(0)
    out["apg"] = numeric(raw["ast_per_game"]).fillna(0)
    out["spg"] = numeric(raw["stl_per_game"]).fillna(0)
    out["bpg"] = numeric(raw["blk_per_game"]).fillna(0)
    out["tov"] = np.nan

    fg_made = numeric(raw["twoPM"]).fillna(0) + numeric(raw["TPM"]).fillna(0)
    fg_att = numeric(raw["twoPA"]).fillna(0) + numeric(raw["TPA"]).fillna(0)
    out["fg"] = safe_ratio(fg_made, fg_att)
    out["tp"] = normalize_pct(raw["TP_%"])
    out["ft"] = normalize_pct(raw["FT_%"])
    out["ts"] = normalize_pct(raw["TS_%"])
    out["efg"] = normalize_pct(raw["eFG_%"])
    out["usg"] = normalize_pct(raw["usg_%"])

    three_attempted = numeric(raw["three_attempted"]).fillna(np.nan)
    total_attempted = numeric(raw["total_attempted"]).fillna(np.nan)
    fallback_three_share = safe_ratio(raw["TPA"], numeric(raw["twoPA"]).fillna(0) + numeric(raw["TPA"]).fillna(0))
    out["three_share"] = safe_ratio(three_attempted, total_attempted).fillna(fallback_three_share)
    out["ast_tov"] = numeric(raw["ast_tov"])
    estimated_tov = safe_ratio(out["apg"], out["ast_tov"])
    out["tov"] = estimated_tov.fillna(0)

    pace_to_40 = 40.0 / out["mpg"].replace(0, np.nan)
    out["pts_per_40"] = (out["ppg"] * pace_to_40).fillna(0)
    out["reb_per_40"] = (out["rpg"] * pace_to_40).fillna(0)
    out["ast_per_40"] = (out["apg"] * pace_to_40).fillna(0)
    out["stl_per_40"] = (out["spg"] * pace_to_40).fillna(0)
    out["blk_per_40"] = (out["bpg"] * pace_to_40).fillna(0)
    out["tov_per_40"] = (out["tov"] * pace_to_40).fillna(0)

    out["assist_creation"] = numeric(raw["AST_%"]).fillna(0)
    out["dreb_arch"] = numeric(raw["DRB_%"]).fillna(0)
    out["dreb_source"] = "DRB_%"
    out["bpm"] = numeric(raw["bpm"])
    out["porpag"] = numeric(raw["porpag"])
    out["adjoe"] = numeric(raw["adjoe"])
    out["drtg"] = numeric(raw["drtg"])
    out["adrtg"] = numeric(raw["adrtg"])
    out["dporpag"] = numeric(raw["dporpag"])
    out["stops_per_40"] = numeric(raw["stops_per_40"])
    out["has_season_row"] = raw["has_season_row"].fillna(False).astype(bool)
    out["has_pbp_row"] = raw["has_pbp_row"].fillna(False).astype(bool)

    out["orb_pct"] = normalize_pct(raw["ORB_%"])
    out["drb_pct"] = normalize_pct(raw["DRB_%"])
    out["ast_pct"] = normalize_pct(raw["AST_%"])
    out["to_pct"] = normalize_pct(raw["TO_%"])
    out["blk_pct"] = normalize_pct(raw["blk_%"])
    out["stl_pct"] = normalize_pct(raw["stl_%"])
    out["ft_rate"] = numeric(raw["FT_rate"])
    out["three_pa_per_100"] = numeric(raw["threePA_per_100"])

    shot_profile_columns = [
        "FTM",
        "FTA",
        "twoPM",
        "twoPA",
        "twoP_%",
        "TPM",
        "TPA",
        "rimmade",
        "rimatt",
        "rim_%",
        "midmade",
        "midatt",
        "mid_%",
        "rim_attempted",
        "mid_attempted",
        "three_attempted",
        "total_attempted",
        "rim_pct_of_total_attempts",
        "mid_pct_of_total_attempts",
        "three_pct_of_total_attempts",
        "pct_rim_made_assisted",
        "pct_mid_made_assisted",
        "pct_three_made_assisted",
    ]
    for column in shot_profile_columns:
        target = column.lower().replace("%", "pct").replace(".", "_")
        out[target] = numeric(raw[column])

    out["pct_assist_creation"] = percentile(out["assist_creation"].fillna(0))
    out["pct_three_pct"] = percentile(out["tp"].fillna(0))
    out["pct_three_rate"] = percentile(out["three_share"].fillna(0))
    out["pct_ast_tov"] = percentile(out["ast_tov"].fillna(0))
    out["pct_efg"] = percentile(out["efg"].fillna(0))
    out["pct_dreb"] = percentile(out["dreb_arch"].fillna(0))
    out["pct_dreb_pos_adj"] = percentile_by_pos(out.assign(pos=out["pos"].replace("", "Unknown")), "dreb_arch", pos_col="pos")
    out["pct_size"] = percentile(out["heightIn"].fillna(0))

    f_heights = out.loc[out["pos"] == "F", "heightIn"].dropna()
    stretch_big_height_gate = float(f_heights.min()) if not f_heights.empty else float(out["heightIn"].median())
    out["stretch_big_height_gate"] = stretch_big_height_gate

    out["meets_high_assist"] = out["pct_assist_creation"] >= THRESHOLDS["high_percentile"]
    out["meets_efg_500"] = out["efg"] >= THRESHOLDS["baseline_efg"]
    out["meets_3p_300"] = out["tp"] >= THRESHOLDS["baseline_3p_pct"]
    out["meets_high_ast_tov"] = out["pct_ast_tov"] >= THRESHOLDS["high_percentile"]
    out["meets_positive_ast_tov"] = out["ast_tov"] > THRESHOLDS["positive_ast_tov"]
    out["meets_dreb_profile"] = out["dreb_arch"] >= THRESHOLDS["d1_dreb_rate"]

    out["meets_pg_preferred"] = (
        out["meets_high_assist"]
        & (out["tp"] >= THRESHOLDS["pg_3p_pct"])
        & (out["three_share"] >= THRESHOLDS["pg_3p_rate"])
        & out["meets_positive_ast_tov"]
    )
    out["meets_wing_preferred"] = (
        out["meets_dreb_profile"]
        & (out["tp"] >= THRESHOLDS["wing_3p_pct"])
        & (out["three_share"] >= THRESHOLDS["wing_3p_rate"])
        & out["meets_positive_ast_tov"]
    )
    out["meets_big_preferred"] = (
        out["pos"].isin(["F/C", "C"])
        & (out["heightIn"] >= stretch_big_height_gate)
        & out["meets_dreb_profile"]
        & (out["tp"] >= THRESHOLDS["big_3p_pct"])
        & (out["three_share"] >= THRESHOLDS["big_3p_rate"])
        & out["meets_positive_ast_tov"]
    )

    out["score_pg_combo"] = (
        0.35 * out["pct_assist_creation"]
        + 0.20 * out["pct_three_pct"]
        + 0.15 * out["pct_three_rate"]
        + 0.20 * out["pct_ast_tov"]
        + 0.10 * out["pct_efg"]
        + bool_bonus(out["meets_pg_preferred"], 8.0)
    ).clip(0, 100)
    out["score_wing_2_4"] = (
        0.30 * out["pct_dreb_pos_adj"]
        + 0.25 * out["pct_three_pct"]
        + 0.20 * out["pct_three_rate"]
        + 0.15 * out["pct_ast_tov"]
        + 0.10 * out["pct_size"]
        + bool_bonus(out["meets_wing_preferred"], 8.0)
    ).clip(0, 100)
    out["score_stretch_big"] = (
        0.30 * out["pct_dreb_pos_adj"]
        + 0.25 * out["pct_size"]
        + 0.20 * out["pct_three_pct"]
        + 0.15 * out["pct_three_rate"]
        + 0.10 * out["pct_ast_tov"]
        + bool_bonus(out["meets_big_preferred"], 8.0)
    ).clip(0, 100)

    score_cols = list(ARCHETYPE_LABELS)
    out["primary_score_col"] = out[score_cols].idxmax(axis=1)
    out["primary_score"] = out[score_cols].max(axis=1)
    out["primary_archetype"] = out["primary_score_col"].map(ARCHETYPE_LABELS).fillna("Unassigned")

    out["transfer_available"] = False
    out["transfer_status"] = ""
    out["transfer_from"] = ""
    out["transfer_to"] = ""

    out = out[out["name"].str.len() > 0].copy()
    out = out.sort_values(["team", "name", "id"]).reset_index(drop=True)
    out, pca_meta = add_pca_columns(out)
    out.attrs["pca_meta"] = pca_meta
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the first WBB Division I processed dataset.")
    parser.add_argument(
        "--input",
        default="merged-players-2026 copy.csv",
        help="Path to the merged raw women’s Division I CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/wbb_d1_processed_players.csv",
        help="Path for the processed app-ready CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    raw = pd.read_csv(input_path)
    processed = build_dataset(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    review_count = int(processed["class_year_review_needed"].sum())
    pca_meta = processed.attrs.get("pca_meta", {})
    print(f"Wrote {len(processed)} players to {output_path}")
    print(f"Class-year manual review rows: {review_count}")
    print(f"Processed columns: {len(processed.columns)}")
    if pca_meta:
        ratios = pca_meta["explained_ratio"]
        print("PCA explained variance ratio:")
        for i, value in enumerate(ratios, start=1):
            print(f"  PC{i}: {value:.4f} ({value * 100:.2f}%)")
        print(f"  Cumulative PC1-PC{len(ratios)}: {ratios.sum():.4f} ({ratios.sum() * 100:.2f}%)")
        print("Top loadings by component:")
        loadings = pca_meta["loadings"]
        for col in loadings.columns[:4]:
            top = loadings[col].abs().sort_values(ascending=False).head(5).index.tolist()
            parts = [f"{feature}={loadings.loc[feature, col]:+.3f}" for feature in top]
            print(f"  {col}: " + ", ".join(parts))


if __name__ == "__main__":
    main()

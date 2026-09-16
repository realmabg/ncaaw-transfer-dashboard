from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_wbb_d1_dataset import build_dataset


DEFAULT_SOURCE_DIR = Path("womens_player_pbp_and_regular_stats_2021_2026")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_K8_ARCHETYPES_PATH = Path("all-player-seasons-2021-2026-core-v1-k8-archetypes.csv")


K8_ARCHETYPE_LABELS = {
    "A0": "Traditional Big",
    "A1": "Midrange-Heavy Role Player",
    "A2": "Two-Way Star Big",
    "A3": "Three-Point Specialist",
    "A4": "Combo Guard",
    "A5": "Two-Way Star Guard",
    "A6": "Low-Production Player",
    "A7": "Efficient Off-Ball Finisher",
}


LEGACY_ARCHETYPE_COLUMNS = [
    "score_pg_combo",
    "score_wing_2_4",
    "score_stretch_big",
    "qual_pg_reason",
    "qual_wing_reason",
    "qual_big_reason",
]


def apply_k8_archetypes(processed: pd.DataFrame, archetype_path: Path, current_season: int | None = None) -> pd.DataFrame:
    out = processed.copy()
    score_cols = [f"score_{code.lower()}" for code in K8_ARCHETYPE_LABELS]
    metadata_cols = [
        "dominant_archetype_code",
        "dominant_archetype",
        "top_1_code",
        "top_1_archetype",
        "top_1_pct",
        "top_2_code",
        "top_2_archetype",
        "top_2_pct",
        "top_3_code",
        "top_3_archetype",
        "top_3_pct",
    ]
    for col in [*score_cols, *metadata_cols]:
        if col not in out.columns:
            out[col] = pd.NA

    if not archetype_path.exists():
        target_mask = pd.Series(True, index=out.index)
        if current_season is not None:
            target_mask = pd.to_numeric(out["season"], errors="coerce").eq(current_season)
        out.loc[target_mask, score_cols] = 0.0
        out.loc[target_mask, "primary_archetype"] = "Unassigned"
        out.loc[target_mask, "primary_score"] = 0.0
        out.loc[target_mask, "primary_score_col"] = ""
        return out

    archetypes = pd.read_csv(archetype_path)
    archetypes["__player_id"] = archetypes["player_id"].astype(str)
    archetypes["__season"] = pd.to_numeric(archetypes["season"], errors="coerce").astype("Int64")
    source_cols = [
        "__season",
        "__player_id",
        *metadata_cols,
        *K8_ARCHETYPE_LABELS.keys(),
    ]
    target = out[["season", "id"]].copy()
    target["__row_index"] = target.index
    target["__season"] = pd.to_numeric(target["season"], errors="coerce").astype("Int64")
    target["__player_id"] = target["id"].astype(str)
    merged = target.merge(archetypes[source_cols], on=["__season", "__player_id"], how="left")
    matched = merged["dominant_archetype"].notna()
    idx = merged["__row_index"]

    out.loc[idx, score_cols] = 0.0
    for code in K8_ARCHETYPE_LABELS:
        out.loc[idx, f"score_{code.lower()}"] = pd.to_numeric(merged[code], errors="coerce").fillna(0.0).to_numpy() * 100.0
    for col in metadata_cols:
        out.loc[idx, col] = merged[col].to_numpy()
    out.loc[idx, "primary_archetype"] = merged["dominant_archetype"].fillna("Unassigned").to_numpy()
    out.loc[idx, "primary_score"] = pd.to_numeric(merged["top_1_pct"], errors="coerce").fillna(0.0).to_numpy() * 100.0
    out.loc[idx, "primary_score_col"] = merged["dominant_archetype_code"].str.lower().radd("score_").where(matched, "").to_numpy()
    return out


def read_yearly_sources(source_dir: Path) -> pd.DataFrame:
    files = sorted(source_dir.glob("merged-players-*.csv"))
    if not files:
        raise FileNotFoundError(f"No merged-players-*.csv files found in {source_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "season" not in frame.columns:
            year_text = path.stem.rsplit("-", 1)[-1]
            frame["season"] = int(year_text)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def historical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "name": "player_name",
        "heightIn": "height_inches",
        "gp": "GP",
        "mpg": "mins_per_game",
        "ppg": "pts_per_game",
        "rpg": "treb_per_game",
        "apg": "ast_per_game",
        "spg": "stl_per_game",
        "bpg": "blk_per_game",
        "tp": "3P_pct",
        "ft": "FT_pct",
        "efg": "eFG",
        "ast_tov": "AST_TOV",
        "ast_pct": "AST_pct",
        "tov_pct": "TOV_pct",
        "orb_pct": "ORB_pct",
        "drb_pct": "DRB_pct",
        "stl_pct": "Stl_pct",
        "blk_pct": "Blk_pct",
        "ftr": "FTR",
        "pf_per_40": "personal_fouls_per_40",
        "primary_archetype": "archetype",
    }
    cols = [
        "season_player_id",
        "player_name",
        "team",
        "conf",
        "season",
        "cls",
        "pos",
        "archetype",
        "height_inches",
        "GP",
        "mins_per_game",
        "pts_per_game",
        "treb_per_game",
        "ast_per_game",
        "stl_per_game",
        "blk_per_game",
        "bpm",
        "porpag",
        "eFG",
        "ts",
        "FT_pct",
        "3P_pct",
        "three_pa_per_100",
        "AST_TOV",
        "AST_pct",
        "TOV_pct",
        "ORB_pct",
        "DRB_pct",
        "Stl_pct",
        "Blk_pct",
        "FTR",
        "personal_fouls_per_40",
        "usg",
        "three_share",
        "rim_share",
        "mid_share",
        "rim_pct",
        "mid_pct",
        "assisted_fg_pct",
        "rim_assisted_pct",
        "mid_assisted_pct",
        "three_assisted_pct",
        "stops_per_40",
        "score_a0",
        "score_a1",
        "score_a2",
        "score_a3",
        "score_a4",
        "score_a5",
        "score_a6",
        "score_a7",
        "primary_score",
        "dominant_archetype_code",
        "dominant_archetype",
        "top_1_code",
        "top_1_archetype",
        "top_1_pct",
        "top_2_code",
        "top_2_archetype",
        "top_2_pct",
        "top_3_code",
        "top_3_archetype",
        "top_3_pct",
    ]
    out = frame.rename(columns=rename).copy()
    out["year"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["class"] = out["cls"]
    out["role"] = out["pos"]
    out["height"] = out["height_inches"].apply(lambda v: f"{int(v)//12}'{int(v)%12}\"" if pd.notna(v) else "")
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[[*cols, "year", "class", "role", "height"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WBB current and historical dashboard datasets.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--current-season", type=int, default=2026)
    parser.add_argument("--k8-archetypes", default=str(DEFAULT_K8_ARCHETYPES_PATH))
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = read_yearly_sources(source_dir)
    processed = build_dataset(raw)
    processed["season"] = pd.to_numeric(processed["season"], errors="coerce").astype("Int64")
    processed = apply_k8_archetypes(processed, Path(args.k8_archetypes), args.current_season)
    processed = processed.drop(columns=LEGACY_ARCHETYPE_COLUMNS, errors="ignore")

    current = processed[processed["season"].eq(args.current_season)].copy()
    historical = processed[processed["season"].lt(args.current_season)].copy()
    historical_index = historical_columns(historical)

    all_path = processed_dir / "wbb_all_player_seasons_2021_2026.csv"
    current_path = processed_dir / "wbb_d1_processed_players.csv"
    historical_path = processed_dir / "wbb_historical_player_index.csv"

    processed.to_csv(all_path, index=False)
    current.to_csv(current_path, index=False)
    historical_index.to_csv(historical_path, index=False)

    print(f"Wrote {len(processed)} all-season rows to {all_path}")
    print(f"Wrote {len(current)} current rows to {current_path}")
    print(f"Wrote {len(historical_index)} historical rows to {historical_path}")


if __name__ == "__main__":
    main()

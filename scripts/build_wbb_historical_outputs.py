from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_wbb_d1_dataset import build_dataset


DEFAULT_SOURCE_DIR = Path("womens_player_pbp_and_regular_stats_2021_2026")
DEFAULT_PROCESSED_DIR = Path("data/processed")


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
        "score_pg_combo",
        "score_wing_2_4",
        "score_stretch_big",
        "primary_score",
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
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = read_yearly_sources(source_dir)
    processed = build_dataset(raw)
    processed["season"] = pd.to_numeric(processed["season"], errors="coerce").astype("Int64")

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

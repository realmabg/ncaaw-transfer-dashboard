# Archetype Process Explainer

The current-player dashboard uses the frozen **core v1 K8** archetype model. The model assigns each player a membership weight across eight basketball profiles and uses the highest weight as the player’s primary archetype.

## Current Archetypes

The eight displayed archetypes are:

- Traditional Big
- Midrange-Heavy Role Player
- Two-Way Star Big
- Three-Point Specialist
- Combo Guard
- Two-Way Star Guard
- Low-Production Player
- Efficient Off-Ball Finisher

The player profile shows all eight membership scores. The highlighted archetype is the row with the highest K8 membership.

## Source File

Current-season archetypes come from:

`all-player-seasons-2021-2026-core-v1-k8-archetypes.csv`

That file is scored from the frozen K8 model and covers every 2021-26 player-season used by the dashboard. Its 2026 rows were compared against the existing 3,572-player K8 export; all overlapping players matched the provided dominant archetype, with only tiny rounding differences in membership weights.

## Historical Players

The frozen K8 scoring file covers the 2021-26 historical and current player pools. The K8 labels are used for current-player profile cards, filters, current-player lists, historical-player filters, historical comparison cards, and archetype score bars.

## Map Axes

The archetype map uses PCA coordinates from the frozen model. PC1 runs from spacing-guard profiles on the low side toward size and rebounding profiles on the high side. PC2 is displayed with lower-usage support roles lower on the map and higher-usage creation profiles higher on the map.

## Triton Zone Thresholds

The Triton Zone thresholds and special tags are separate from the K8 archetype model. Those controls are rule-based evaluation tools for the tracker and should not be read as the source of the main player archetype label.

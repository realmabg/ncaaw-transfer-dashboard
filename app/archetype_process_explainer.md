# Archetype Process Explainer

The current-player dashboard now uses the 2025-26 K8 archetype file as the source of player archetypes. The model assigns each player a probability-like score across eight basketball profiles and uses the highest score as the player’s primary archetype.

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

The player profile shows all eight scores. The highlighted archetype is the row with the highest K8 score.

## Source File

Current-season archetypes come from:

`all-players-2025-26-core-v1-k8-archetypes copy.csv`

The dashboard matches that file to the current player pool by player id. Players in the current pool that do not have a matching row in the K8 file are labeled **Unassigned** and receive zeroes for the K8 archetype score bars.

## Historical Players

The K8 file only covers the 2025-26 current player pool, so historical comparison records keep their existing historical labels and comparison data. The K8 labels are used for current-player profile cards, filters, current-player lists, and current-player archetype score bars.

## Triton Zone Thresholds

The Triton Zone thresholds and special tags are separate from the K8 archetype model. Those controls are still rule-based evaluation tools for the tracker and should not be read as the source of the main player archetype label.

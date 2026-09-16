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

`all-players-2025-26-core-v1-k8-archetypes copy.csv`

That file is scored from the frozen K8 model and covers the full 2025-26 current-player dashboard pool. It was compared against the existing 3,572-player K8 export; all overlapping players matched the provided dominant archetype, with only tiny rounding differences in membership weights.

## Historical Players

The frozen K8 scoring file covers the 2025-26 current player pool. Historical comparison records keep their existing historical labels and comparison data. The K8 labels are used for current-player profile cards, filters, current-player lists, and current-player archetype score bars.

## Triton Zone Thresholds

The Triton Zone thresholds and special tags are separate from the K8 archetype model. Those controls are rule-based evaluation tools for the tracker and should not be read as the source of the main player archetype label.

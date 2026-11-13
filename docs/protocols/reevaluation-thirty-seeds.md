# Supplemental final-checkpoint evaluation

Evaluate the five learned methods (BC, DAgger, SAC, SAC with symmetric sampling,
and RLPD), each at training seeds 0–2, with 30 fixed evaluation seeds
20000–20029. Evaluate the expert once on the same seeds as a shared reference.
Use the original four training and nine held-out tracks from `kartstudy.tracks`.
Keep the split and each track explicit in the episode records.

Use completed final checkpoints at matched method-specific training budgets:
8,000 BC updates or 400,000 online actions. Replication actors supersede
interrupted original-suite checkpoints; do not select an actor by evaluation
performance. The explicit input roster, checkpoint hashes, configurations and
source snapshot are saved with the supplemental run.

Run `scripts/reevaluate.py --roots <run directories> --output <fresh directory>`.
Each CPU worker owns a native headless simulator and uses one PyTorch thread.
Reuse deterministic inference, observation/action decoding, the 0.1-second action
interval, and the 1,500-action episode limit. No training or parameter updates
occur. Results go to a fresh directory and preserve all original evaluations.
Full trajectories are optional (`--trajectories`); episode metrics are always saved.

Each worker saves per-track episodes incrementally and writes its final
`evaluations.jsonl` only after all requested tracks complete. `manifest.json`
records input selection and worker outcomes; each worker's `status.json` records
progress and completion. A failed job prevents the suite being marked complete.

Average evaluation laps within each training run before comparing methods.
Independent training runs remain the statistical unit. Report completion beside
successful lap time, weight off-road percentages by measured race time, and keep
training and held-out summaries separate. The expert has no training replicates.
The original four-seed study protocol and completed artifacts remain unchanged.

For the report's eight-track selection, collect full trajectories on the four
training tracks and Hacienda, Northern Resort, Ravenbridge Mansion and Zen
Garden with the same frozen actors and 30 evaluation seeds. Zen Garden is a
supplemental held-out track; it does not change the original thirteen-track
suite. Store this collection separately. `report/scripts/prepare_performance.py`
computes the report's lap and local-driving summaries with the existing metrics.
The capped-time bars show one training run each, with sample SD across the 30
evaluation-seed means (each averaging the four tracks in that split). Table
ranges continue to describe the three independent training runs.

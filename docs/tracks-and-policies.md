# Tracks and policies

Retain **one policy per learned method, training seed 0** (five learned methods
with the SAC symmetric sampling add-on). This choice is fixed
across collaborators; it is not the seed with the best evaluation score.
The viewer reads `runs/study-polar-four/selected-policies.json`.

| CLI name (`--policy`) | Viewer name | Retained training |
|---|---|---|
| `expert` | Expert | Original hand-written Team 4 controller; no training seed |
| `bc` | Behavior cloning | Seed 0; 8,000 supervised updates |
| `dagger` | DAgger | Seed 0; 400,000 online actions |
| `sac` | Online SAC | Seed 0; 400,000 online actions |
| `sac_sym` | SAC + symmetric sampling | Seed 0; 400,000 online actions |
| `rlpd` | RLPD | Seed 0; 400,000 online actions |

Demonstration seeds 1000–1019 provide training laps; 1020–1021 provide validation
laps on each training track. These differ from the single retained training seed.
Every learned policy is shared across the four training tracks.

## Expert-relative difficulty

These are Kat's qualitative study groups, informed jointly by harmful-row contacts
and time spent off-track. They describe this expert under our solo conditions,
not intrinsic game difficulty or formal metric cutoffs. Hazard exposure varies:
a high contact percentage on one row is less stable than on many rows. Read both
metrics together; completion and native rescues remain additional checks.
All 84 diagnostic laps (21 tracks × evaluation seeds 0–3) completed.

Contacts are inferred banana contacts / initial harmful rows, averaged per seed.
Off-track is the percentage of race time flagged off-road. No initial harmful
rows means N/A. Newly dropped bubblegum contacts are not fully measured.
See [measurement definitions and limits](expert-evaluation.md).

The table lists all 21 installed race tracks, all available in the viewer.
Thirteen are study tracks (training or held out); eight are outside the study. Use the exact simulator
ID with `--track`. Difficulty is never passed to the policy.

| Track | Simulator ID | Expert difficulty | Study role | Contacts / rows (%) | Off-track time (%) |
|---|---|---|---|---:|---:|
| Antediluvian Abyss | `abyss` | Simple | Held out | 0.00 | 0.00 |
| Around the Lighthouse | `lighthouse` | Simple | Training | N/A | 0.00 |
| Candela City | `candela_city` | Simple | Held out | 0.00 | 0.22 |
| Cocoa Temple | `cocoa_temple` | Simple | Held out | 0.00 | 0.63 |
| Cornfield Crossing | `cornfield_crossing` | Simple | Training | 19.44 | 0.04 |
| Gran Paradiso Island | `gran_paradiso_island` | Simple | Unused | 0.00 | 0.93 |
| Hacienda | `hacienda` | Simple | Held out | N/A | 1.41 |
| Nessie's Pond | `scotland` | Simple | Unused | 12.50 | 0.00 |
| Shifting Sands | `sandtrack` | Simple | Unused | 15.00 | 0.00 |
| Snow Peak | `snowtuxpeak` | Simple | Unused | 0.00 | 0.40 |
| Volcan Island | `volcano_island` | Simple | Held out | 0.00 | 1.22 |
| XR591 | `xr591` | Simple | Unused | 15.38 | 0.00 |
| Black Forest | `black_forest` | Medium | Training | 3.00 | 3.89 |
| Northern Resort | `snowmountain` | Medium | Held out | 6.82 | 2.20 |
| Old Mine | `mines` | Medium | Held out | 7.69 | 3.38 |
| Oliver's Math Class | `olivermath` | Medium | Unused | 25.00 | 3.41 |
| Fort Magma | `fortmagma` | Difficult | Held out | 0.00 | 12.30 |
| Minigolf | `minigolf` | Difficult | Unused | 0.00 | 23.20 |
| Ravenbridge Mansion | `ravenbridge_mansion` | Difficult | Held out | 0.00 | 10.34 |
| STK Enterprise | `stk_enterprise` | Difficult | Training | 15.38 | 13.26 |
| Zen Garden | `zengarden` | Difficult | Unused | 0.00 | 10.57 |

Diagnostic JSON is distributed in `study-data.zip` under
`runs/expert-road-quality-final/metrics.json`. The distinction between native
rescues and the expert’s reversing recovery routine is preserved.

## Separate ablation

`sac_sym` (viewer name **SAC + symmetric sampling**) is the additional 20-demo
ablation, using training seed 0. It is separate from the original four release
policies. Its completed seed-0 checkpoint is included in the default viewer
when the SAC symmetric sampling add-on archive is restored. Open it directly with:

```sh
uv run python -m kartstudy.playback --root runs/ablation-sac-sym-demo20 --policy sac_sym
```

See the [ablation protocol](study.md#additional-20-demo-symmetric-sampling-ablation).

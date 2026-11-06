# Team 4 original policy

Source: https://github.com/osigaud/pystk_project/tree/ddf2cb6573850f890bba26d4e5f82888d9dac82a/src/agents/team4

The coordinator, active subagents, steering, speed, item enum, and YAML parameters
are copied from this pinned commit. `kart_agent.py` comes from `src/agents/`.
`manifest.json` records upstream and local SHA-256 checksums.

Compatibility changes only:

- Absolute imports of `agents.kart_agent` and `utils.track_utils` become relative.
- `track_utils.py` contains the original `compute_curvature` function verbatim,
  with its NumPy import; unrelated visualization/CSV utilities are omitted.
- Episode reset plumbing lives outside these files in `../expert.py` and avoids
  the original reset hook issuing a second, unseeded simulator reset.

No steering, recovery, avoidance, item, nitro, or parameter changes are applied.
In particular, the original curvature calculation is retained without unwrapping.
The upstream repository contains no LICENSE file; attribution is not a license grant.

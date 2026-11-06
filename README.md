# Learning to drive: DAgger vs. RLPD

How should a driving agent learn from an expert: by asking for corrections as
it drives, or by combining demonstrations with its own experience?

We compare behavior cloning, DAgger, online SAC, and RLPD in solo SuperTuxKart
driving. Each policy learns across four training tracks; nine held-out tracks test
generalization. Difficulty groups are used only for reporting. Evaluation uses
normal driving only.

The expert is Team 4's original coordinator at a pinned source commit, including
recovery, hazard avoidance, item use, and nitro. Every method has the same
84-value observations and five controls: steer, throttle, brake, nitro, and fire.
Completion, off-track time, and rescue counts
characterize driving quality alongside lap time; harmful-item contacts are
measured relative to rows of hazards.

- [Study protocol](docs/study.md)
- [Expert metrics and measurement limits](docs/expert-evaluation.md)
- [Watch policies](docs/visualization.md)
- [Share artifacts](docs/experiment-data.md)
- [Development and commands](AGENTS.md)

Built with [rl_mind](https://master-dac.isir.upmc.fr/rl/rl-mind/),
[PySTK2 Gymnasium](https://github.com/bpiwowar/pystk2-gymnasium), and
[Team 4](https://github.com/osigaud/pystk_project/tree/ddf2cb6573850f890bba26d4e5f82888d9dac82a/src/agents/team4).

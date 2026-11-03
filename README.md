# Learning to drive: DAgger vs. RLPD

How should a driving agent learn from an expert: by asking for corrections as
it drives, or by combining past demonstrations with its own experience?

This project explores that question in **SuperTuxKart**. The task is simple:
one kart, one track, one lap—as fast and reliably as possible. Opponents and
items are outside the study.

## Two ways to learn from experience

**DAgger** learns to imitate an expert. As the learner drives, the expert labels
the states it visits, including situations missing from the original
demonstrations.

**RLPD** learns from rewards using both a fixed dataset and new driving
experience. Its symmetric sampling strategy draws half of each training batch
from prior data and half from the agent's own experience.

We compare these approaches with behavior cloning, online SAC, and SAC with
symmetric sampling alone to understand what each ingredient contributes.

## What we investigate

- **Speed and reliability:** which agents finish laps, and how fast?
- **Learning efficiency:** how much interaction and expert assistance do they need?
- **Dependence on demonstrations:** what changes with less data or a slower expert?
- **Robustness and transfer:** how do they handle steering disturbances and an unseen track?

The empirical study uses repeated training runs and reports uncertainty across
training seeds. Lap times are interpreted alongside completion rates, so fast
successful laps do not hide failures. Our RLPD configuration is adapted to the
available compute; the implementation differences are documented in the protocol.

## Explore the project

- [Project brief](TASK.md)
- [Initial experiments](docs/initial-experiments.md)
- [Development and experiment commands](AGENTS.md)

Built with [rl_mind](https://master-dac.isir.upmc.fr/rl/rl-mind/) and
[PySTK2 Gymnasium](https://github.com/bpiwowar/pystk2-gymnasium), using an expert
adapted from the [SuperTuxKart agents project](https://github.com/osigaud/pystk_project/tree/master/src/agents).

## Papers

- [DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)
- [RLPD: Efficient Online Reinforcement Learning with Offline Data](https://proceedings.mlr.press/v202/ball23a.html)

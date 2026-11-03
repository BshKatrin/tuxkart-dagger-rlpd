The goal of this project is to investigate the relative performance of DAgger and symmetric sampling (from RLPD) in Super Tux Kart driving agents.

To run DAgger, you will need an expert policy. You can use any policy available in this project:

https://github.com/osigaud/pystk_project/tree/master/src/agents

To run symmetric sampling, you will need a strong dataset. You can generate it using the agents available on the link above.

Your study should focus on a single agent alone on a track and trying to perform one lap as fast as possible, without taking care of opponent-related items.

Using the rl-mind toolkit is recommended, but not mandatory. You may use another RL library, such as Stable baselines 3, Tianshu or CleanRL. 

Your report should contain your empirical study, a more conceptual comparison is also welcome. Think of clearly specifying your methodology and all the hyper-parameters you used to favor reproducibility. Think also of trying to establish statistically valid conclusions. You may also provide a link to a public github project with your sources.
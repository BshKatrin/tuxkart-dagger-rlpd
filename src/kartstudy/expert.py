"""Original Team 4 coordinator, with episode/interface plumbing only."""

from types import SimpleNamespace

from ._team4.agent4 import Agent4


class Expert:
    name = "team4-original"
    source_commit = "ddf2cb6573850f890bba26d4e5f82888d9dac82a"

    def reset(self, observation):
        # Agent4.reset normally resets its environment. The caller already made
        # the seeded simulator reset: return that observation without a second,
        # unseeded simulator reset, while running all original component resets.
        self.agent = Agent4(SimpleNamespace(reset=lambda: (observation, {})))
        self.agent.reset()

    def act(self, observation):
        return self.agent.choose_action(observation)

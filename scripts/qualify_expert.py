import json
import time
from pathlib import Path

import torch

from kartstudy.environment import DrivingEnv


def main():
    results = []
    for track in ["lighthouse", "zengarden", "snowtuxpeak"]:
        env = DrivingEnv(track=track)
        try:
            for conservative in [False, True]:
                for seed in range(3):
                    _obs, _ = env.reset(seed=seed)
                    start = time.perf_counter()
                    for step in range(env.max_steps):
                        _obs, _reward, done, trunc, info = env.step(
                            env.expert_action(conservative)
                        )
                        if done or trunc:
                            break
                    result = dict(
                        track=track,
                        conservative=conservative,
                        seed=seed,
                        steps=step + 1,
                        wall_seconds=time.perf_counter() - start,
                        **{
                            k: info[k]
                            for k in ["success", "elapsed_seconds", "distance"]
                        },
                    )
                    results.append(result)
                    print(json.dumps(result), flush=True)
        finally:
            env.close()
    Path("runs").mkdir(exist_ok=True)
    Path("runs/expert_qualification.json").write_text(json.dumps(results, indent=2))
    print("Metal available:", torch.backends.mps.is_available(), flush=True)


if __name__ == "__main__":
    main()

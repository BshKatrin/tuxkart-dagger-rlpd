"""Summarize completed checkpoints without treating evaluation laps as seeds."""

import argparse
import json
from pathlib import Path

from kartstudy.evaluation import summarize_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="runs")
    parser.add_argument(
        "--output", default=None, help="JSON path; defaults to ROOT/summary.json"
    )
    args = parser.parse_args()
    destination = Path(args.output) if args.output else Path(args.root) / "summary.json"
    summaries = summarize_runs(args.root, output=destination)
    lines = [
        "# Experiment summary",
        "",
        "Final available checkpoints; confidence intervals bootstrap independent training seeds.",
        "Lap times are conditional on successful completion. A single seed has no interval.",
        "",
        "| Configuration | Evaluation | Checkpoint steps | Training seeds | Completion | Successful lap time (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def fmt(estimate, percentage=False):
        if estimate["mean"] is None:
            return "—"
        scale = 100 if percentage else 1
        suffix = "%" if percentage else ""
        value = f"{estimate['mean'] * scale:.2f}{suffix}"
        if estimate["ci95"] is not None:
            low, high = estimate["ci95"]
            value += f" [{low * scale:.2f}, {high * scale:.2f}]"
        return value

    for index, result in enumerate(summaries):
        config = result["config"]
        label = config.get("algo", config.get("algorithm", "run"))
        n = result["completion_rate"]["n_training_seeds"]
        low_step, high_step = result["checkpoint_step_range"]
        step_label = (
            str(low_step)
            if low_step == high_step
            else f"{low_step}–{high_step} (unequal)"
        )
        lines.append(
            f"| {index + 1}. {label} | {result['condition']} | {step_label} | {n} | {fmt(result['completion_rate'], True)} | {fmt(result['conditional_lap_time'])} |"
        )
    lines.extend(["", "## Configurations", ""])
    for index, result in enumerate(summaries):
        lines.extend(
            [
                f"### {index + 1}",
                "",
                "```json",
                json.dumps(result["config"], indent=2),
                "```",
                "",
            ]
        )
    markdown_path = destination.with_suffix(".md")
    markdown_path.write_text("\n".join(lines))
    print(
        f"{len(summaries)} experimental groups\n{destination.resolve()}\n{markdown_path.resolve()}"
    )


if __name__ == "__main__":
    main()

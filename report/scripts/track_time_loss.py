"""Create one 2 x 4 expert-relative road map per retained learned policy.

From the repository root:
    uv run python report/scripts/track_time_loss.py collect
    uv run --with matplotlib --with seaborn python report/scripts/track_time_loss.py plot
    uv run python report/scripts/track_time_loss.py summary
    uv run python report/scripts/track_time_loss.py local-summary

Tracks come from the Training/Held out rows of docs/tracks-and-policies.md.
Collection writes separate expert references, geometry, and missing policy
evaluations under report/generated/track-time-loss/data/. Existing final policy
trajectories are read in place. No training, checkpoint selection, or original
evaluation changes are performed. Plotting writes vector PDFs for the report,
150-dpi PNG previews and a machine-readable analysis under
report/generated/track-time-loss/.
Summary writes aggregate-metrics.json, retaining every failed attempt.
Local-summary writes local-metrics.json using available matched segment traversals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "report/generated/track-time-loss"
SEEDS = (20000, 20001, 20002, 20003)
METHODS = {
    "bc": "Behavior cloning",
    "dagger": "DAgger",
    "sac": "Online SAC",
    "sac_sym": "SAC + symmetric sampling",
    "rlpd": "RLPD",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tracks(path):
    tracks = []
    for line in Path(path).read_text().splitlines():
        fields = [field.strip() for field in line.split("|")[1:-1]]
        if len(fields) == 6 and fields[3] in ("Training", "Held out"):
            tracks.append(
                {"name": fields[0], "id": fields[1].strip("`"), "split": fields[3]}
            )
    training = [track for track in tracks if track["split"] == "Training"]
    held_out = [track for track in tracks if track["split"] == "Held out"]
    if len(training) != 4 or len(held_out) != 4:
        raise ValueError("Expected exactly 4 training and 4 held-out tracks")
    return training + held_out


def collect(args):
    import torch

    from kartstudy.environment import SCHEMA_VERSION, DrivingEnv
    from kartstudy.evaluation import evaluate
    from kartstudy.playback import load_actor

    torch.set_num_threads(1)
    tracks = read_tracks(args.tracks)
    selection = json.loads((args.root / "selected-policies.json").read_text())
    directories = {
        method: (args.root / selection[method]).resolve() for method in METHODS
    }
    for method, directory in directories.items():
        config = json.loads((directory / "config.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        if (
            config["seed"] != 0
            or config["schema_version"] != SCHEMA_VERSION
            or config["algo"] != method
            or status["status"] != "completed"
        ):
            raise ValueError(
                f"Expected retained completed schema-3 seed-0 run: {directory}"
            )
    data_root = args.output / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "tracks": tracks,
        "seeds": list(SEEDS),
        "schema_version": SCHEMA_VERSION,
        "track_list_sha256": digest(args.tracks),
        "selection_sha256": digest(args.root / "selected-policies.json"),
        "environment_sha256": digest(PROJECT_ROOT / "src/kartstudy/environment.py"),
        "expert_sha256": digest(PROJECT_ROOT / "src/kartstudy/expert.py"),
        "sources": {},
    }
    actors = {}
    env = DrivingEnv(track=tracks[0]["id"])
    try:
        for track in tracks:
            track_id = track["id"]
            directory = data_root / track_id
            directory.mkdir(exist_ok=True)
            env.track = track_id
            env.env.unwrapped.default_track = track_id
            geometry_path = directory / "geometry.json"

            def expert_actor(obs, path=geometry_path):
                if not path.exists():
                    native = env.env.unwrapped.track
                    path.write_text(
                        json.dumps(
                            {
                                "length": float(native.length),
                                "nodes": np.asarray(native.path_nodes).tolist(),
                                "distances": np.asarray(native.path_distance).tolist(),
                                "widths": np.asarray(native.path_width).tolist(),
                                "successors": [
                                    list(map(int, values))
                                    for values in native.successors
                                ],
                            }
                        )
                    )
                return env.expert_action()

            metadata["sources"][track_id] = {}
            for method in ("expert", *METHODS):
                if method != "expert":
                    original = directories[method] / f"trajectories_{track_id}.npz"
                else:
                    original = None
                if original is not None and original.exists():
                    records = [
                        episode
                        for line in (directories[method] / "evaluations.jsonl")
                        .read_text()
                        .splitlines()
                        for row in [json.loads(line)]
                        if row.get("final")
                        for episode in row["episodes"]
                        if episode["track"] == track_id
                    ]
                    trace_path = original
                    origin = "original final evaluation"
                else:
                    trace_path = directory / f"{method}-trajectories.npz"
                    records_path = directory / f"{method}-episodes.json"
                    if trace_path.exists() != records_path.exists():
                        raise ValueError(
                            f"Incomplete collection cache: {directory}/{method}"
                        )
                    if records_path.exists():
                        records = json.loads(records_path.read_text())
                    else:
                        if method == "expert":
                            actor = expert_actor
                        else:
                            if method not in actors:
                                actors[method] = load_actor(directories[method])[0]
                            actor = actors[method]
                        records = evaluate(
                            env, actor, SEEDS, trajectory_path=trace_path
                        )
                        records_path.write_text(json.dumps(records, indent=2))
                    origin = "supplemental deterministic CPU evaluation"
                if sorted(episode["seed"] for episode in records) != list(SEEDS):
                    raise ValueError(
                        f"Unexpected evaluation seeds: {track_id}/{method}"
                    )
                metadata["sources"][track_id][method] = {
                    "path": str(trace_path.relative_to(PROJECT_ROOT)),
                    "sha256": digest(trace_path),
                    "origin": origin,
                    "episodes": records,
                }
                print(
                    f"{track_id}: {method}, {sum(e['success'] for e in records)}/4 finished ({origin})",
                    flush=True,
                )
            (args.output / "sources.json").write_text(json.dumps(metadata, indent=2))
    finally:
        env.close()


def reference_route(geometry):
    """Follow the native first-successor reference route, preserving 3D nodes."""
    route, seen, current = [], set(), 0
    while current not in seen:
        seen.add(current)
        route.append(current)
        current = geometry["successors"][current][0]
    if current != 0:
        raise ValueError("Native reference route does not close at its start")
    distances = np.asarray(geometry["distances"])[route]
    if np.any(np.diff(distances[:, 0]) < 0):
        raise ValueError("Native reference progress is not ordered")
    return route


def timing_gates(geometry, spacing=20.0):
    """Use ~20 m sectors, merging route choices between common junctions."""
    route = reference_route(geometry)
    order = {node: rank for rank, node in enumerate(route)}
    distances = np.asarray(geometry["distances"])
    length = geometry["length"]
    branches = []
    for node in route:
        successors = geometry["successors"][node]
        if len(successors) < 2:
            continue
        pending = list(successors[1:])
        visited, joins = set(), []
        while pending:
            alternative = pending.pop()
            if alternative in order:
                joins.append(alternative)
            elif alternative not in visited:
                visited.add(alternative)
                pending.extend(geometry["successors"][alternative])
        if not joins:
            raise ValueError("Alternative route does not reconnect to reference")
        end_rank = max((order[join] - order[node]) % len(route) for join in joins)
        join = route[(order[node] + end_rank) % len(route)]
        start, end = float(distances[node, 1]), float(distances[join, 0])
        if end <= start:
            raise ValueError(
                "A route split crosses the start line; explicit gates needed"
            )
        branches.append([start, end])
    merged = []
    for start, end in sorted(branches):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gates = [
        gate
        for gate in np.arange(0, length, spacing)
        if not any(start < gate < end for start, end in merged)
    ]
    gates.extend(gate for bounds in merged for gate in bounds)
    gates.append(length)
    return np.unique(gates), merged


def first_passage_sectors(progress, times, gates, rescue_active, finish_time=None):
    """Keep stopping/reversing delays; never infer a fast rescue traversal."""
    progress, times = np.asarray(progress), np.asarray(times)
    rescue_active = np.asarray(rescue_active, dtype=bool)
    if len(progress) != len(times) or len(times) != len(rescue_active):
        raise ValueError("Inconsistent telemetry lengths")
    if np.any(np.diff(times) < -1e-6):
        raise ValueError("Simulation time goes backward")
    crossings = np.full(len(gates), np.nan)
    crossings[0] = 0.0
    for index, gate in enumerate(gates[1:], start=1):
        if index == len(gates) - 1 and finish_time is not None:
            crossings[index] = finish_time
            continue
        reached = np.flatnonzero(progress >= gate)
        if not len(reached):
            continue
        step = int(reached[0])
        before_progress = progress[step - 1] if step else 0.0
        before_time = times[step - 1] if step else 0.0
        if progress[step] <= before_progress:
            raise ValueError("First forward crossing has no forward displacement")
        fraction = (gate - before_progress) / (progress[step] - before_progress)
        crossings[index] = before_time + fraction * (times[step] - before_time)
    durations = np.diff(crossings)
    if np.any(durations < -1e-6):
        raise ValueError("Negative sector time")
    before_times = np.r_[0.0, times[:-1]]
    for index, (start, stop) in enumerate(pairwise(crossings)):
        # Status belongs to the preceding action interval, rather than a point.
        if not np.isfinite(start + stop) or np.any(
            rescue_active & (times > start) & (before_times < stop)
        ):
            durations[index] = np.nan
    if (
        finish_time is not None
        and np.all(np.isfinite(durations))
        and not np.isclose(durations.sum(), finish_time, atol=1e-5)
    ):
        raise ValueError("Sector times do not reconcile with official finish time")
    return durations


def analyze_track(track, sources, geometry):
    gates, branches = timing_gates(geometry)
    durations, episodes = {}, {}
    for method, source in sources.items():
        path = PROJECT_ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise ValueError(f"Input trajectory changed: {path}")
        by_seed = {row["seed"]: row for row in source["episodes"]}
        episodes[method] = [by_seed[seed] for seed in SEEDS]
        laps = []
        with np.load(path) as data:
            for seed in SEEDS:
                mask = data["seed"] == seed
                if not np.any(mask):
                    raise ValueError(f"Missing seed {seed}: {path}")
                record = by_seed[seed]
                laps.append(
                    first_passage_sectors(
                        data["progress"][mask],
                        data["elapsed_seconds"][mask],
                        gates,
                        data["rescue_active"][mask],
                        finish_time=record["lap_time"] if record["success"] else None,
                    )
                )
        durations[method] = np.asarray(laps)
    comparison = {}
    for method in METHODS:
        delta = durations[method] - durations["expert"]
        counts = np.isfinite(delta).sum(axis=0)
        local_mean = np.full(delta.shape[1], np.nan)
        local_sd = np.full(delta.shape[1], np.nan)
        complete = counts == len(SEEDS)
        local_mean[complete] = delta[:, complete].mean(axis=0)
        local_sd[complete] = delta[:, complete].std(axis=0, ddof=1)
        # Express longer branch sectors on the same physical scale as 20 m bins.
        rate = 20 / np.diff(gates)
        gap = [
            policy["lap_time"] - expert["lap_time"]
            for policy, expert in zip(episodes[method], episodes["expert"])
            if policy["success"] and expert["success"]
        ]
        for index, (policy, expert) in enumerate(
            zip(episodes[method], episodes["expert"])
        ):
            if (
                policy["success"]
                and expert["success"]
                and np.isfinite(delta[index]).all()
                and not np.isclose(
                    delta[index].sum(),
                    policy["lap_time"] - expert["lap_time"],
                    atol=1e-5,
                )
            ):
                raise ValueError("Local time gaps do not reconcile with lap-time gap")
        comparison[method] = {
            "mean_seconds_per_20m": local_mean * rate,
            "sd_seconds_per_20m": local_sd * rate,
            "matched_pairs": counts,
            "finished": sum(row["success"] for row in episodes[method]),
            "expert_finished": sum(row["success"] for row in episodes["expert"]),
            "lap_gap_mean": float(np.mean(gap)) if gap else None,
            "lap_gap_sd": float(np.std(gap, ddof=1)) if len(gap) > 1 else None,
            "lap_gap_n": len(gap),
        }
    return {
        "track": track,
        "gates": gates,
        "branch_sectors": branches,
        "comparison": comparison,
    }


def clean_json(value):
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    return value


def aggregate_episodes(episodes_by_track, seeds, horizon):
    """Equal track/seed weights for time; actual exposure weights for off-road."""
    by_track = {}
    for track, episodes in episodes_by_track.items():
        if sorted(episode["seed"] for episode in episodes) != sorted(seeds):
            raise ValueError(f"Expected one attempt per evaluation seed: {track}")
        by_track[track] = {episode["seed"]: episode for episode in episodes}
    episodes = [episode for rows in by_track.values() for episode in rows.values()]
    if not episodes:
        raise ValueError("No episodes to aggregate")
    if any(
        not episode["success"]
        and (
            not episode["timeout"]
            or not np.isclose(episode["sim_seconds"], horizon, atol=1e-5)
        )
        for episode in episodes
    ):
        raise ValueError("Failed attempts must share the full timeout budget")
    success_times = [episode["lap_time"] for episode in episodes if episode["success"]]
    if any(time is None or not 0 < time <= horizon for time in success_times):
        raise ValueError("Successful finish times must be within the budget")
    capped_by_seed = [
        float(
            np.mean(
                [
                    rows[seed]["lap_time"] if rows[seed]["success"] else horizon
                    for rows in by_track.values()
                ]
            )
        )
        for seed in seeds
    ]
    exposure = sum(episode["measurement_seconds"] for episode in episodes)
    if exposure <= 0:
        raise ValueError("No measured race time")
    return {
        "n_tracks": len(by_track),
        "attempts": len(episodes),
        "completed": len(success_times),
        "completion_pct": 100 * len(success_times) / len(episodes),
        "successful_lap_time_mean": float(np.mean(success_times))
        if success_times
        else None,
        "capped_time_mean": float(np.mean(capped_by_seed)),
        "capped_time_sd": float(np.std(capped_by_seed, ddof=1)),
        "capped_time_by_evaluation_seed": capped_by_seed,
        "offroad_pct": 100
        * sum(episode["off_track_seconds"] for episode in episodes)
        / exposure,
        "rescues_per_attempt": float(
            np.mean([episode["rescue_count"] for episode in episodes])
        ),
    }


def summary(args):
    metadata = json.loads((args.output / "sources.json").read_text())
    failed = [
        episode
        for methods in metadata["sources"].values()
        for source in methods.values()
        for episode in source["episodes"]
        if not episode["success"]
    ]
    if not failed:
        raise ValueError("Cannot verify the race-clock timeout from these records")
    horizon = float(failed[0]["sim_seconds"])
    metrics = {
        "race_clock_budget_seconds": horizon,
        "evaluation_seeds": metadata["seeds"],
        "groups": {},
    }
    for group in ("All tracks", "Training", "Held out"):
        tracks = [
            track
            for track in metadata["tracks"]
            if group == "All tracks" or track["split"] == group
        ]
        metrics["groups"][group] = {}
        for method in ("expert", *METHODS):
            episodes_by_track = {
                track["id"]: metadata["sources"][track["id"]][method]["episodes"]
                for track in tracks
            }
            metrics["groups"][group][method] = aggregate_episodes(
                episodes_by_track, metadata["seeds"], horizon
            )
    path = args.output / "aggregate-metrics.json"
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(path.relative_to(PROJECT_ROOT))
    for group, rows in metrics["groups"].items():
        for method, row in rows.items():
            print(
                f"{group}: {method}, finished {row['completed']}/{row['attempts']}, capped {row['capped_time_mean']:.2f} ± {row['capped_time_sd']:.2f} s"
            )


def first_crossing_values(progress, values, positions):
    """Interpolate the original adjacent samples at each first forward crossing."""
    progress, values = np.asarray(progress), np.asarray(values)
    result = np.full((len(positions), values.shape[1]), np.nan)
    indices = np.searchsorted(np.maximum.accumulate(progress), positions)
    valid = (indices > 0) & (indices < len(progress))
    after = indices[valid]
    fraction = (positions[valid] - progress[after - 1]) / (
        progress[after] - progress[after - 1]
    )
    result[valid] = values[after - 1] + fraction[:, None] * (
        values[after] - values[after - 1]
    )
    return result


def road_lateral_positions(positions, progress, geometry):
    """Signed horizontal road-frame displacement in nominal road half-widths.

    Native 3D segments disambiguate overpasses; native progress limits the search
    to the relevant road region. Global up defines the lateral axis, so jumps
    and kart heading do not change its sign. Banked-road edges are approximate.
    """
    nodes = np.asarray(geometry["nodes"], dtype=float)
    distances = np.asarray(geometry["distances"], dtype=float)
    widths = np.asarray(geometry["widths"], dtype=float).reshape(-1)
    start, tangent = nodes[:, 0], nodes[:, 1] - nodes[:, 0]
    squared_length = np.sum(tangent * tangent, axis=1)
    horizontal_length = np.linalg.norm(tangent[:, [0, 2]], axis=1)
    result = np.full(len(positions), np.nan)
    for lo in range(0, len(positions), 128):
        hi = min(lo + 128, len(positions))
        relative = positions[lo:hi, None, :] - start[None, :, :]
        fraction = np.clip(
            np.sum(relative * tangent, axis=2) / np.maximum(squared_length, 1e-12),
            0,
            1,
        )
        offset = relative - fraction[:, :, None] * tangent
        distance_squared = np.sum(offset * offset, axis=2)
        nearby = (
            (distances[None, :, 0] <= progress[lo:hi, None] + 20)
            & (distances[None, :, 1] >= progress[lo:hi, None] - 20)
            & (widths[None, :] > 0)
            & (horizontal_length[None, :] > 1e-8)
        )
        distance_squared[~nearby] = np.inf
        indices = np.argmin(distance_squared, axis=1)
        rows = np.arange(hi - lo)
        valid = np.isfinite(distance_squared[rows, indices])
        road_offset = offset[rows, indices]
        sideways = (
            road_offset[:, 0] * tangent[indices, 2]
            - road_offset[:, 2] * tangent[indices, 0]
        ) / np.maximum(horizontal_length[indices], 1e-12)
        result[lo:hi] = np.where(valid, 2 * sideways / widths[indices], np.nan)
    return result


def reduce_local_sectors(policy_time, expert_time, speeds, offsets, lengths, usable):
    """Summarize a seed's available sectors, never zero-filling missing values."""
    valid = (
        usable
        & np.isfinite(policy_time)
        & np.isfinite(expert_time)
        & np.isfinite(speeds)
        & np.isfinite(offsets)
    )
    if not valid.any():
        return {"signed_gap": np.nan, "segments": 0, "speed": np.nan, "lateral": np.nan}
    length = lengths[valid].sum()
    return {
        "signed_gap": float(
            20 * (policy_time[valid] - expert_time[valid]).sum() / length
        ),
        "segments": int(valid.sum()),
        "speed": float(np.sum(speeds[valid] * lengths[valid]) / length),
        "lateral": float(100 * np.sum(offsets[valid] * lengths[valid]) / length),
    }


def local_summary(args):
    metadata = json.loads((args.output / "sources.json").read_text())
    tracks = {}
    for track in metadata["tracks"]:
        track_id = track["id"]
        geometry = json.loads(
            (args.output / "data" / track_id / "geometry.json").read_text()
        )
        gates, _branches = timing_gates(geometry)
        lengths = np.diff(gates)
        positions, sector_ids = [], []
        for index, (start, stop) in enumerate(pairwise(gates)):
            count = max(1, int(np.ceil((stop - start) / 2)))
            positions.extend(start + (np.arange(count) + 0.5) * (stop - start) / count)
            sector_ids.extend([index] * count)
        positions, sector_ids = np.asarray(positions), np.asarray(sector_ids)
        channels = {}
        for method, source in metadata["sources"][track_id].items():
            path = PROJECT_ROOT / source["path"]
            if digest(path) != source["sha256"]:
                raise ValueError(f"Input trajectory changed: {path}")
            records = {episode["seed"]: episode for episode in source["episodes"]}
            channels[method] = []
            with np.load(path) as data:
                for seed in SEEDS:
                    mask = data["seed"] == seed
                    progress = data["progress"][mask]
                    durations = first_passage_sectors(
                        progress,
                        data["elapsed_seconds"][mask],
                        gates,
                        data["rescue_active"][mask],
                        records[seed]["lap_time"],
                    )
                    sampled = first_crossing_values(
                        progress,
                        np.column_stack(
                            [data["speed"][mask], data["world_position"][mask]]
                        ),
                        positions,
                    )
                    lateral = road_lateral_positions(
                        sampled[:, 1:], positions, geometry
                    )
                    speeds, offsets = [], []
                    for index in range(len(lengths)):
                        selected = sector_ids == index
                        speeds.append(np.mean(sampled[selected, 0]))
                        offsets.append(np.mean(np.abs(lateral[selected])))
                    channels[method].append(
                        {
                            "times": durations,
                            "speeds": np.asarray(speeds),
                            "offsets": np.asarray(offsets),
                        }
                    )
        rows = {}
        for method in ("expert", *METHODS):
            rows[method] = []
            for policy, expert in zip(channels[method], channels["expert"]):
                usable = np.isfinite(expert["speeds"]) & np.isfinite(expert["offsets"])
                rows[method].append(
                    reduce_local_sectors(
                        policy["times"],
                        expert["times"],
                        policy["speeds"],
                        policy["offsets"],
                        lengths,
                        usable,
                    )
                )
        tracks[track_id] = {"total_segments": len(lengths), "by_seed": rows}
        print(f"Local summaries: {track_id}", flush=True)
    result = {"tracks": tracks, "groups": {}, "evaluation_seeds": list(SEEDS)}
    for group in ("All tracks", "Training", "Held out"):
        selected = [
            track["id"]
            for track in metadata["tracks"]
            if group == "All tracks" or track["split"] == group
        ]
        result["groups"][group] = {
            "total_segments_mean": float(
                np.mean([tracks[track]["total_segments"] for track in selected])
            ),
            "methods": {},
        }
        for method in ("expert", *METHODS):
            fields = {}
            for field in ("signed_gap", "segments", "speed", "lateral"):
                # Each track has equal weight within a fixed evaluation seed.
                values = np.asarray(
                    [
                        [row[field] for row in tracks[track]["by_seed"][method]]
                        for track in selected
                    ]
                )
                by_seed = np.mean(values, axis=0)
                fields[field] = {
                    "mean": float(np.mean(by_seed)),
                    "sd": float(np.std(by_seed, ddof=1)),
                    "by_seed": by_seed.tolist(),
                }
            result["groups"][group]["methods"][method] = fields
    path = args.output / "local-metrics.json"
    path.write_text(json.dumps(clean_json(result), indent=2, allow_nan=False))
    print(path.relative_to(PROJECT_ROOT))


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    metadata = json.loads((args.output / "sources.json").read_text())
    tracks = read_tracks(args.tracks)
    if tracks != metadata["tracks"]:
        raise ValueError("Track selection changed since collection")
    analysis, geometries = {}, {}
    for track in tracks:
        track_id = track["id"]
        geometry = json.loads(
            (args.output / "data" / track_id / "geometry.json").read_text()
        )
        geometries[track_id] = geometry
        analysis[track_id] = analyze_track(
            track, metadata["sources"][track_id], geometry
        )
    (args.output / "analysis.json").write_text(
        json.dumps(clean_json(analysis), indent=2, allow_nan=False)
    )
    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
        }
    )
    # Fixed color semantics: green = faster, neutral = equal, red = slower.
    cmap = LinearSegmentedColormap.from_list(
        "time_gap", ["#15803d", "#808080", "#c62828"]
    )
    norm = Normalize(-0.5, 0.5, clip=True)
    missing = "#d6d6d6"
    for method, name in METHODS.items():
        fig, axes = plt.subplots(2, 4, figsize=(17 / 2.54, 10.8 / 2.54))
        fig.subplots_adjust(
            left=0.025, right=0.985, top=0.86, bottom=0.21, wspace=0.24, hspace=0.57
        )
        fig.text(0.025, 0.97, f"{name} versus expert", fontsize=10)
        for ax, track in zip(axes.flat, tracks):
            track_id = track["id"]
            geometry, result = geometries[track_id], analysis[track_id]
            values = result["comparison"][method]
            route = reference_route(geometry)
            nodes = np.asarray(geometry["nodes"])
            distances = np.asarray(geometry["distances"])
            origin = nodes[route[0], 0]
            # Plot all geometry quietly, including alternative routes.
            background = nodes[:, :, [0, 2]] - origin[[0, 2]]
            ax.add_collection(
                LineCollection(background, colors="0.90", linewidths=0.45)
            )
            segments, sector_ids = [], []
            for node in route:
                start, stop = distances[node]
                if stop <= start:
                    continue
                internal = result["gates"][
                    (result["gates"] > start) & (result["gates"] < stop)
                ]
                boundaries = np.r_[start, internal, stop]
                for lo, hi in pairwise(boundaries):
                    fraction = (np.array([lo, hi]) - start) / (stop - start)
                    segment = nodes[node, 0] + fraction[:, None] * (
                        nodes[node, 1] - nodes[node, 0]
                    )
                    segments.append(segment[:, [0, 2]] - origin[[0, 2]])
                    sector_ids.append(
                        min(
                            len(result["gates"]) - 2,
                            np.searchsorted(
                                result["gates"], (lo + hi) / 2, side="right"
                            )
                            - 1,
                        )
                    )
            segments, sector_ids = np.asarray(segments), np.asarray(sector_ids)
            means = values["mean_seconds_per_20m"][sector_ids]
            deviations = values["sd_seconds_per_20m"][sector_ids]
            valid = np.isfinite(means)
            ax.add_collection(
                LineCollection(segments[~valid], colors=missing, linewidths=2.0)
            )
            stable = valid & (np.abs(means) > deviations)
            variable = valid & ~stable
            for mask, style in ((stable, "solid"), (variable, (0, (1.6, 1.2)))):
                collection = LineCollection(
                    segments[mask],
                    cmap=cmap,
                    norm=norm,
                    linewidths=2.0,
                    linestyles=style,
                )
                collection.set_array(means[mask])
                ax.add_collection(collection)
            ax.plot(
                0,
                0,
                marker="o",
                color="0.15",
                markersize=3,
                markeredgecolor="white",
                markeredgewidth=0.4,
                zorder=5,
            )
            ax.update_datalim(background.reshape(-1, 2))
            ax.autoscale_view()
            ax.margins(0.07)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            ax.set_title(f"{track['name']}\nFinished {values['finished']}/4", pad=5)
        color_axis = fig.add_axes((0.23, 0.13, 0.54, 0.025))
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=color_axis,
            orientation="horizontal",
            extend="both",
            ticks=[-0.5, 0, 0.5],
        )
        colorbar.ax.set_xticklabels(["≤ −0.5", "0", "≥ +0.5"])
        colorbar.set_label(
            "Local time gap (s / 20 m): green faster, red slower", labelpad=3
        )
        colorbar.outline.set_visible(False)
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
        legend = [
            Line2D(
                [],
                [],
                color="0.3",
                linewidth=1.5,
                linestyle=(0, (1.6, 1.2)),
                label="±1 SD spans zero",
            ),
            Line2D(
                [], [], color=missing, linewidth=2, label="Fewer than 4 valid pairs"
            ),
            Line2D(
                [],
                [],
                color="0.15",
                marker="o",
                linestyle="none",
                markersize=3,
                label="Start / finish",
            ),
        ]
        fig.legend(
            handles=legend,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.005),
            ncol=3,
            frameon=False,
            fontsize=7,
        )
        output = args.output / f"track-time-loss-{method}.pdf"
        fig.savefig(output, facecolor="white")
        fig.savefig(output.with_suffix(".png"), dpi=150, facecolor="white")
        plt.close(fig)
        print(output.relative_to(PROJECT_ROOT), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("collect", "plot", "summary", "local-summary")
    )
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT / "runs/study-polar-four"
    )
    parser.add_argument(
        "--tracks", type=Path, default=PROJECT_ROOT / "docs/tracks-and-policies.md"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "plot":
        plot(args)
    elif args.command == "summary":
        summary(args)
    else:
        local_summary(args)


if __name__ == "__main__":
    main()

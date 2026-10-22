import argparse
import json
from pathlib import Path

import numpy as np

from src.config_utils import get_video


def generate_windows(duration_sec: float, window_sec: float, stride_sec: float):
    windows = []
    start = 0.0
    idx = 1

    while start < duration_sec:
        end = min(duration_sec, start + window_sec)

        windows.append({
            "window_id": idx,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
        })

        if end >= duration_sec:
            break

        start += stride_sec
        idx += 1

    return windows


def cut_times_in_window(cut_times, start, end):
    return [t for t in cut_times if start < t < end]


def local_shot_durations(start, end, cuts):
    boundaries = [start] + list(cuts) + [end]
    durations = []

    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b > a:
            durations.append(b - a)

    return durations


def local_deltas(frame_deltas, start, end):
    values = []

    for item in frame_deltas:
        midpoint = (float(item["from_sec"]) + float(item["to_sec"])) / 2.0

        if start <= midpoint < end:
            values.append(float(item["delta"]))

    return values


def classify_regime(features, cfg):
    cuts_per_min = features["cuts_per_minute"]
    median_shot = features["median_shot_duration_sec"]
    static_ratio = features["static_pair_ratio"]

    static_cfg = cfg["static"]
    high_cfg = cfg["high_cut"]
    low_cfg = cfg["low_cut"]

    if (
        cuts_per_min <= float(static_cfg["max_cuts_per_min"])
        and static_ratio >= float(static_cfg["min_static_pair_ratio"])
    ):
        return "static"

    if (
        cuts_per_min >= float(high_cfg["min_cuts_per_min"])
        or median_shot <= float(high_cfg["max_median_shot_sec"])
    ):
        return "high_cut"

    if (
        cuts_per_min <= float(low_cfg["max_cuts_per_min"])
        or median_shot >= float(low_cfg["min_median_shot_sec"])
    ):
        return "low_cut"

    return "medium_cut"


def soft_regime_scores(features, cfg):
    """
    Simple interpretable soft scores for diagnostics.
    These are not probabilities and are not trained.
    """
    cpm = features["cuts_per_minute"]
    med = max(features["median_shot_duration_sec"], 1e-6)
    static_ratio = features["static_pair_ratio"]

    static_ref = float(cfg["static"]["max_cuts_per_min"])
    high_ref = float(cfg["high_cut"]["min_cuts_per_min"])
    low_ref = float(cfg["low_cut"]["max_cuts_per_min"])
    high_med = float(cfg["high_cut"]["max_median_shot_sec"])
    low_med = float(cfg["low_cut"]["min_median_shot_sec"])

    static_score = max(
        0.0,
        min(
            1.0,
            0.6 * static_ratio
            + 0.4 * (1.0 - min(cpm / max(static_ref, 1e-6), 1.0))
        )
    )

    high_score = max(
        0.0,
        min(
            1.0,
            0.6 * min(cpm / max(high_ref, 1e-6), 1.0)
            + 0.4 * min(high_med / med, 1.0)
        )
    )

    low_score = max(
        0.0,
        min(
            1.0,
            0.6 * (1.0 - min(cpm / max(low_ref, 1e-6), 1.0))
            + 0.4 * min(med / max(low_med, 1e-6), 1.0)
        )
    )

    medium_score = max(
        0.0,
        1.0 - max(static_score, high_score, low_score) * 0.85
    )

    return {
        "static": static_score,
        "low_cut": low_score,
        "medium_cut": medium_score,
        "high_cut": high_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()

    cfg = get_video(args.video_id)
    profiling_cfg = cfg["base"]["profiling"]

    analysis_path = Path(
        f"data/analysis/{cfg['video_id']}_analysis.json"
    )

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"R0 output not found: {analysis_path.resolve()}\n"
            f"Run R0 first."
        )

    analysis = json.loads(
        analysis_path.read_text(encoding="utf-8")
    )

    duration_sec = float(
        analysis["metadata"]["duration_sec"]
    )
    window_sec = float(profiling_cfg["window_sec"])
    stride_sec = float(profiling_cfg["stride_sec"])
    static_delta_threshold = float(
        profiling_cfg["static_delta_threshold"]
    )

    cut_times = analysis["shot_analysis"]["cut_times_sec"]
    frame_deltas = analysis["visual_change"]["frame_deltas"]

    windows = generate_windows(
        duration_sec,
        window_sec,
        stride_sec,
    )

    local_profiles = []
    counts = {
        "static": 0,
        "low_cut": 0,
        "medium_cut": 0,
        "high_cut": 0,
    }

    for window in windows:
        start = float(window["start_sec"])
        end = float(window["end_sec"])
        duration = float(window["duration_sec"])

        cuts = cut_times_in_window(
            cut_times, start, end
        )
        shot_durations = local_shot_durations(
            start, end, cuts
        )
        deltas = local_deltas(
            frame_deltas, start, end
        )

        cuts_per_minute = len(cuts) / max(duration / 60.0, 1e-9)

        median_shot = (
            float(np.median(shot_durations))
            if shot_durations
            else duration
        )

        mean_shot = (
            float(np.mean(shot_durations))
            if shot_durations
            else duration
        )

        static_pair_ratio = (
            sum(d < static_delta_threshold for d in deltas) / len(deltas)
            if deltas
            else 1.0
        )

        mean_delta = (
            float(np.mean(deltas))
            if deltas
            else 0.0
        )

        p90_delta = (
            float(np.percentile(deltas, 90))
            if deltas
            else 0.0
        )

        features = {
            "cut_count": len(cuts),
            "cuts_per_minute": cuts_per_minute,
            "mean_shot_duration_sec": mean_shot,
            "median_shot_duration_sec": median_shot,
            "mean_frame_delta": mean_delta,
            "p90_frame_delta": p90_delta,
            "static_pair_ratio": static_pair_ratio,
        }

        regime = classify_regime(
            features,
            profiling_cfg,
        )

        scores = soft_regime_scores(
            features,
            profiling_cfg,
        )

        counts[regime] += 1

        local_profiles.append({
            **window,
            "features": {
                key: round(value, 6)
                if isinstance(value, float)
                else value
                for key, value in features.items()
            },
            "regime": regime,
            "diagnostic_scores": {
                key: round(value, 6)
                for key, value in scores.items()
            },
        })

    payload = {
        "video_id": cfg["video_id"],
        "video_path": analysis["video_path"],
        "profiling": {
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "overlapping": stride_sec < window_sec,
            "static_delta_threshold": static_delta_threshold,
        },
        "num_windows": len(local_profiles),
        "regime_counts": counts,
        "windows": local_profiles,
        "notes": [
            "Regimes describe local temporal dynamics, not semantic video categories.",
            "Diagnostic scores are heuristic scores, not calibrated probabilities.",
            "R1 windows are profiling windows only; they are not final retrieval segments."
        ],
    }

    out = Path(
        f"data/profiles/{cfg['video_id']}_local_profile.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"\n[R1 OK] Video ID       : {cfg['video_id']}")
    print(f"[R1 OK] Windows        : {len(local_profiles)}")
    print(f"[R1 OK] Regime counts  : {counts}")
    print(f"[R1 OK] Output         : {out.resolve()}")

    print("\nFirst local windows:")
    for item in local_profiles[:8]:
        f = item["features"]
        print(
            f"  {item['start_sec']:7.1f}-{item['end_sec']:7.1f}s "
            f"| {item['regime']:10s} "
            f"| cuts/min={f['cuts_per_minute']:.2f} "
            f"| median_shot={f['median_shot_duration_sec']:.2f}s "
            f"| static_ratio={f['static_pair_ratio']:.2f}"
        )


if __name__ == "__main__":
    main()

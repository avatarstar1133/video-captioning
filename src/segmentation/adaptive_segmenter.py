import argparse
import json
import math
from collections import Counter
from pathlib import Path

from src.config_utils import get_video

EPS = 1e-9


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_policy(cfg, regime):
    return cfg["policies"][regime]


def regime_at_time(t, windows, fallback):
    covering = [w for w in windows if float(w["start_sec"]) <= t <= float(w["end_sec"]) + EPS]
    if not covering:
        return fallback

    scores = {k: 0.0 for k in ["static", "low_cut", "medium_cut", "high_cut"]}
    has_scores = False

    for w in covering:
        d = w.get("diagnostic_scores")
        if isinstance(d, dict):
            has_scores = True
            for k in scores:
                scores[k] += float(d.get(k, 0.0))

    order = ["static", "low_cut", "medium_cut", "high_cut"]
    if has_scores:
        return max(order, key=lambda k: (scores[k], -order.index(k)))

    counts = Counter(str(w.get("regime", fallback)) for w in covering)
    return max(order, key=lambda k: (counts.get(k, 0), -order.index(k)))


def build_regime_runs(duration, windows, resolution, fallback):
    if duration <= 0:
        return []

    runs = []
    start = 0.0
    current = regime_at_time(min(duration, resolution / 2.0), windows, fallback)
    t = resolution

    while t < duration - EPS:
        probe = min(duration, t + resolution / 2.0)
        regime = regime_at_time(probe, windows, fallback)
        if regime != current:
            runs.append({"start_sec": start, "end_sec": t, "regime": current})
            start = t
            current = regime
        t += resolution

    runs.append({"start_sec": start, "end_sec": duration, "regime": current})
    return runs


def cuts_inside(cuts, start, end):
    return [float(x) for x in cuts if start + EPS < float(x) < end - EPS]


def raw_units(start, end, cuts):
    bounds = [start] + sorted(cuts) + [end]
    return [{"start_sec": a, "end_sec": b} for a, b in zip(bounds[:-1], bounds[1:]) if b > a + EPS]


def split_equal(start, end, max_sec):
    duration = end - start
    if duration <= max_sec + EPS:
        return [{"start_sec": start, "end_sec": end}]
    n = int(math.ceil(duration / max_sec))
    size = duration / n
    out = []
    for i in range(n):
        a = start + i * size
        b = end if i == n - 1 else start + (i + 1) * size
        out.append({"start_sec": a, "end_sec": b})
    return out


def merge_short(units, min_sec):
    if not units:
        return []
    out, bucket = [], None
    for u in units:
        if bucket is None:
            bucket = dict(u)
        else:
            bucket["end_sec"] = u["end_sec"]
        if bucket["end_sec"] - bucket["start_sec"] >= min_sec - EPS:
            out.append(bucket)
            bucket = None
    if bucket is not None:
        if out:
            out[-1]["end_sec"] = bucket["end_sec"]
        else:
            out.append(bucket)
    return out


def cap_long(units, max_sec):
    out = []
    for u in units:
        out.extend(split_equal(float(u["start_sec"]), float(u["end_sec"]), max_sec))
    return out


def segment_run(run, cuts, seg_cfg):
    start, end, regime = float(run["start_sec"]), float(run["end_sec"]), run["regime"]
    policy = get_policy(seg_cfg, regime)
    mode = policy["mode"]
    local_cuts = cuts_inside(cuts, start, end)

    if mode == "fixed":
        units = split_equal(start, end, float(policy["max_segment_sec"]))
    elif mode in ("shot_aware", "merge_short_shots"):
        units = raw_units(start, end, local_cuts)
        units = merge_short(units, float(policy["min_segment_sec"]))
        units = cap_long(units, float(policy["max_segment_sec"]))
    else:
        raise ValueError(f"Unknown segmentation mode: {mode}")

    return [
        {
            "start_sec": float(u["start_sec"]),
            "end_sec": float(u["end_sec"]),
            "regime": regime,
            "policy_mode": mode,
        }
        for u in units
    ]


def merge_neighbors(segments, seg_cfg):
    if not segments:
        return []
    out = [dict(segments[0])]
    for s in segments[1:]:
        prev = out[-1]
        same = prev["regime"] == s["regime"]
        touching = abs(float(prev["end_sec"]) - float(s["start_sec"])) <= 1e-6
        max_sec = float(get_policy(seg_cfg, s["regime"])["max_segment_sec"])
        combined = float(s["end_sec"]) - float(prev["start_sec"])
        if same and touching and combined <= max_sec + EPS:
            prev["end_sec"] = s["end_sec"]
        else:
            out.append(dict(s))
    return out


def add_ids(segments):
    out = []
    for i, s in enumerate(segments, start=1):
        a, b = float(s["start_sec"]), float(s["end_sec"])
        out.append({
            "segment_id": i,
            "start_sec": round(a, 3),
            "end_sec": round(b, 3),
            "duration_sec": round(b - a, 3),
            "regime": s["regime"],
            "policy_mode": s["policy_mode"],
        })
    return out


def validate_coverage(segments, duration, tolerance):
    problems = []
    if not segments:
        return {"valid": False, "problems": ["No segments generated"]}

    if abs(float(segments[0]["start_sec"])) > tolerance:
        problems.append("Timeline does not start at 0")
    if abs(float(segments[-1]["end_sec"]) - duration) > tolerance:
        problems.append("Timeline does not end at video duration")

    for a, b in zip(segments[:-1], segments[1:]):
        delta = float(b["start_sec"]) - float(a["end_sec"])
        if delta > tolerance:
            problems.append(f"Gap {delta:.3f}s after segment {a['segment_id']}")
        elif delta < -tolerance:
            problems.append(f"Overlap {-delta:.3f}s after segment {a['segment_id']}")

    return {"valid": len(problems) == 0, "problems": problems}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()

    cfg = get_video(args.video_id)
    seg_cfg = cfg["base"]["segmentation"]

    analysis_path = Path(f"data/analysis/{cfg['video_id']}_analysis.json")
    profile_path = Path(f"data/profiles/{cfg['video_id']}_local_profile.json")
    analysis = load_json(analysis_path)
    profile = load_json(profile_path)

    duration = float(analysis["metadata"]["duration_sec"])
    cuts = analysis["validated_shot_analysis"]["cut_times_sec"]
    windows = profile["windows"]

    runs = build_regime_runs(
        duration,
        windows,
        float(seg_cfg.get("regime_resolution_sec", 1.0)),
        str(seg_cfg.get("fallback_regime", "medium_cut")),
    )

    segments = []
    for run in runs:
        segments.extend(segment_run(run, cuts, seg_cfg))

    if bool(seg_cfg.get("merge_same_regime_neighbors", True)):
        segments = merge_neighbors(segments, seg_cfg)

    segments = add_ids(segments)
    validation = validate_coverage(
        segments,
        duration,
        float(seg_cfg.get("coverage_tolerance_sec", 0.05)),
    )

    counts = Counter(s["regime"] for s in segments)
    durations = [float(s["duration_sec"]) for s in segments]

    payload = {
        "video_id": cfg["video_id"],
        "video_path": analysis["video_path"],
        "duration_sec": duration,
        "method": "adaptive_normalized_segmentation",
        "num_regime_runs": len(runs),
        "regime_runs": [
            {
                "start_sec": round(float(r["start_sec"]), 3),
                "end_sec": round(float(r["end_sec"]), 3),
                "regime": r["regime"],
            }
            for r in runs
        ],
        "num_segments": len(segments),
        "segment_regime_counts": dict(counts),
        "segment_duration_stats": {
            "min_sec": round(min(durations), 3) if durations else 0.0,
            "mean_sec": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "max_sec": round(max(durations), 3) if durations else 0.0,
        },
        "coverage_validation": validation,
        "segments": segments,
    }

    out = Path(f"data/segments/{cfg['video_id']}_segments.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n========================================")
    print("R2 ADAPTIVE NORMALIZED SEGMENTATION")
    print("========================================")
    print(f"[R2 OK] Video ID         : {cfg['video_id']}")
    print(f"[R2 OK] Duration         : {duration:.2f}s")
    print(f"[R2 OK] Regime runs      : {len(runs)}")
    print(f"[R2 OK] Segments         : {len(segments)}")
    print(f"[R2 OK] Regime counts    : {dict(counts)}")
    print(f"[R2 OK] Segment duration : {payload['segment_duration_stats']}")
    print(f"[R2 OK] Coverage valid   : {validation['valid']}")
    print(f"[R2 OK] Output           : {out.resolve()}")

    if not validation["valid"]:
        for p in validation["problems"]:
            print(f"[R2 WARN] {p}")

    print("\nFirst segments:")
    for s in segments[:15]:
        print(
            f"  #{s['segment_id']:04d} "
            f"{s['start_sec']:8.2f}-{s['end_sec']:8.2f}s "
            f"| {s['regime']:10s} "
            f"| {s['policy_mode']:18s} "
            f"| {s['duration_sec']:6.2f}s"
        )


if __name__ == "__main__":
    main()

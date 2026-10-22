import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector

from src.config_utils import get_video


def probe_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {Path(video_path).resolve()}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS: {fps}")

    duration_sec = frame_count / fps if frame_count > 0 else 0.0

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
    }


def make_detector(cfg: dict):
    name = str(cfg.get("shot_detector", "adaptive")).lower()
    min_len = int(cfg.get("min_scene_len_frames", 8))

    if name == "adaptive":
        detector = AdaptiveDetector(
            adaptive_threshold=float(cfg.get("adaptive_threshold", 3.0)),
            min_scene_len=min_len,
        )
    elif name == "content":
        detector = ContentDetector(
            threshold=float(cfg.get("content_threshold", 27.0)),
            min_scene_len=min_len,
        )
    else:
        raise ValueError("shot_detector must be 'adaptive' or 'content'")

    return name, detector


def detect_shots(video_path: str, duration_sec: float, cfg: dict):
    detector_name, detector = make_detector(cfg)

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(detector)

    print(f"[R0] Detecting shots with {detector.__class__.__name__}...")
    manager.detect_scenes(video=video, show_progress=True)

    scenes = manager.get_scene_list(start_in_scene=True)

    shots = []
    cut_times = []

    for i, (start_tc, end_tc) in enumerate(scenes, start=1):
        start = max(0.0, float(start_tc.get_seconds()))
        end = min(duration_sec, float(end_tc.get_seconds()))

        if end <= start:
            continue

        shots.append({
            "shot_id": i,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(end - start, 3),
        })

        if i > 1:
            cut_times.append(round(start, 3))

    if not shots:
        shots = [{
            "shot_id": 1,
            "start_sec": 0.0,
            "end_sec": round(duration_sec, 3),
            "duration_sec": round(duration_sec, 3),
        }]

    return detector_name, shots, cut_times


def sample_visual_change(video_path: str, cfg: dict):
    interval = float(cfg.get("sample_interval_sec", 0.5))
    target_w = int(cfg.get("resize_width", 160))
    target_h = int(cfg.get("resize_height", 90))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0

    sample_times = []
    deltas = []

    previous_gray = None
    t = 0.0

    while t <= duration + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()

        if not ok:
            t += interval
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )
        gray = gray.astype(np.float32) / 255.0

        if previous_gray is not None:
            delta = float(np.mean(np.abs(gray - previous_gray)))
            deltas.append({
                "from_sec": round(max(0.0, t - interval), 3),
                "to_sec": round(t, 3),
                "delta": delta,
            })

        previous_gray = gray
        sample_times.append(round(t, 3))
        t += interval

    cap.release()

    values = [x["delta"] for x in deltas]

    return {
        "sample_interval_sec": interval,
        "num_sampled_frames": len(sample_times),
        "mean_frame_delta": float(np.mean(values)) if values else 0.0,
        "median_frame_delta": float(np.median(values)) if values else 0.0,
        "p90_frame_delta": float(np.percentile(values, 90)) if values else 0.0,
        "frame_deltas": deltas,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()

    cfg = get_video(args.video_id)
    analysis_cfg = cfg["base"]["analysis"]

    info = probe_video(cfg["video_path"])
    detector_name, shots, cut_times = detect_shots(
        cfg["video_path"],
        info["duration_sec"],
        analysis_cfg,
    )
    visual = sample_visual_change(
        cfg["video_path"],
        analysis_cfg,
    )

    shot_durations = [x["duration_sec"] for x in shots]

    payload = {
        "video_id": cfg["video_id"],
        "video_path": str(Path(cfg["video_path"]).as_posix()),
        "metadata": info,
        "shot_analysis": {
            "detector": detector_name,
            "num_shots": len(shots),
            "num_cuts": len(cut_times),
            "cuts_per_minute": (
                len(cut_times) / max(info["duration_sec"] / 60.0, 1e-9)
            ),
            "mean_shot_duration_sec": (
                float(np.mean(shot_durations)) if shot_durations else 0.0
            ),
            "median_shot_duration_sec": (
                float(np.median(shot_durations)) if shot_durations else 0.0
            ),
            "cut_times_sec": cut_times,
            "shots": shots,
        },
        "visual_change": visual,
    }

    out = Path(f"data/analysis/{cfg['video_id']}_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n[R0 OK] Video ID       : {cfg['video_id']}")
    print(f"[R0 OK] Duration       : {info['duration_sec']:.2f}s")
    print(f"[R0 OK] Shots          : {len(shots)}")
    print(f"[R0 OK] Cuts/min       : {payload['shot_analysis']['cuts_per_minute']:.3f}")
    print(f"[R0 OK] Median shot    : {payload['shot_analysis']['median_shot_duration_sec']:.3f}s")
    print(f"[R0 OK] Mean delta     : {visual['mean_frame_delta']:.5f}")
    print(f"[R0 OK] Output         : {out.resolve()}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scenedetect import (
    SceneManager,
    open_video,
)

from scenedetect.detectors import (
    AdaptiveDetector,
    ContentDetector,
)

from src.config_utils import get_video


# ============================================================
# BASIC VIDEO INFO
# ============================================================

def probe_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: "
            f"{Path(video_path).resolve()}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    if fps <= 0:
        raise RuntimeError(
            f"Invalid FPS: {fps}"
        )

    duration_sec = (
        frame_count / fps
        if frame_count > 0
        else 0.0
    )

    return {
        "fps": fps,

        "frame_count":
            frame_count,

        "duration_sec":
            duration_sec,

        "width":
            width,

        "height":
            height,

        "resolution":
            f"{width}x{height}",
    }


# ============================================================
# SCENE DETECTOR
# ============================================================

def make_detector(cfg: dict):

    name = str(
        cfg.get(
            "shot_detector",
            "adaptive"
        )
    ).lower()

    min_len = int(
        cfg.get(
            "min_scene_len_frames",
            8
        )
    )

    if name == "adaptive":

        detector = AdaptiveDetector(
            adaptive_threshold=float(
                cfg.get(
                    "adaptive_threshold",
                    3.0
                )
            ),

            min_scene_len=
                min_len,
        )

    elif name == "content":

        detector = ContentDetector(
            threshold=float(
                cfg.get(
                    "content_threshold",
                    27.0
                )
            ),

            min_scene_len=
                min_len,
        )

    else:

        raise ValueError(
            "shot_detector must be "
            "'adaptive' or 'content'"
        )

    return (
        name,
        detector
    )


# ============================================================
# RAW / CANDIDATE CUT DETECTION
# ============================================================

def detect_candidate_cuts(
    video_path: str,
    duration_sec: float,
    cfg: dict,
):

    detector_name, detector = (
        make_detector(cfg)
    )

    video = open_video(
        video_path
    )

    manager = SceneManager()

    manager.add_detector(
        detector
    )

    print(
        f"[R0] Detecting candidate cuts "
        f"with "
        f"{detector.__class__.__name__}..."
    )

    manager.detect_scenes(
        video=video,
        show_progress=True,
    )

    scene_list = (
        manager.get_scene_list(
            start_in_scene=True
        )
    )

    candidate_cuts = []

    raw_shots = []

    for i, (
        start_tc,
        end_tc
    ) in enumerate(
        scene_list,
        start=1
    ):

        start = max(
            0.0,
            float(
                start_tc.get_seconds()
            )
        )

        end = min(
            duration_sec,
            float(
                end_tc.get_seconds()
            )
        )

        if end <= start:
            continue

        raw_shots.append(
            {
                "shot_id": i,

                "start_sec":
                    round(
                        start,
                        3
                    ),

                "end_sec":
                    round(
                        end,
                        3
                    ),

                "duration_sec":
                    round(
                        end - start,
                        3
                    ),
            }
        )

        if i > 1:

            candidate_cuts.append(
                round(
                    start,
                    3
                )
            )

    if not raw_shots:

        raw_shots = [
            {
                "shot_id": 1,

                "start_sec": 0.0,

                "end_sec":
                    round(
                        duration_sec,
                        3
                    ),

                "duration_sec":
                    round(
                        duration_sec,
                        3
                    ),
            }
        ]

    return {
        "detector":
            detector_name,

        "raw_shots":
            raw_shots,

        "candidate_cut_times_sec":
            candidate_cuts,
    }


# ============================================================
# FRAME READING
# ============================================================

def read_frame_at(
    cap,
    timestamp_sec: float,
):

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        timestamp_sec * 1000.0
    )

    ok, frame = cap.read()

    if not ok:
        return None

    return frame


# ============================================================
# CUT VALIDATION METRICS
# ============================================================

def compare_frames(
    frame_before,
    frame_after,
    resize_width: int,
    resize_height: int,
    changed_pixel_threshold: float,
):

    before_small = cv2.resize(
        frame_before,
        (
            resize_width,
            resize_height
        ),

        interpolation=
            cv2.INTER_AREA
    )

    after_small = cv2.resize(
        frame_after,
        (
            resize_width,
            resize_height
        ),

        interpolation=
            cv2.INTER_AREA
    )

    before_gray = cv2.cvtColor(
        before_small,
        cv2.COLOR_BGR2GRAY
    )

    after_gray = cv2.cvtColor(
        after_small,
        cv2.COLOR_BGR2GRAY
    )

    before_gray = (
        before_gray.astype(
            np.float32
        )
        / 255.0
    )

    after_gray = (
        after_gray.astype(
            np.float32
        )
        / 255.0
    )

    diff = np.abs(
        before_gray
        -
        after_gray
    )

    mean_abs_diff = float(
        np.mean(diff)
    )

    changed_pixel_ratio = float(
        np.mean(
            diff
            >= changed_pixel_threshold
        )
    )

    # --------------------------------------------
    # HSV histogram correlation
    # --------------------------------------------

    before_hsv = cv2.cvtColor(
        before_small,
        cv2.COLOR_BGR2HSV
    )

    after_hsv = cv2.cvtColor(
        after_small,
        cv2.COLOR_BGR2HSV
    )

    hist_before = cv2.calcHist(
        [before_hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    hist_after = cv2.calcHist(
        [after_hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        hist_before,
        hist_before
    )

    cv2.normalize(
        hist_after,
        hist_after
    )

    histogram_correlation = float(
        cv2.compareHist(
            hist_before,
            hist_after,
            cv2.HISTCMP_CORREL
        )
    )

    return {
        "mean_abs_diff":
            mean_abs_diff,

        "changed_pixel_ratio":
            changed_pixel_ratio,

        "histogram_correlation":
            histogram_correlation,
    }


# ============================================================
# VALIDATE CANDIDATE CUTS
# ============================================================

def validate_candidate_cuts(
    video_path: str,
    candidate_cuts,
    duration_sec: float,
    cfg: dict,
):

    validation_cfg = (
        cfg.get(
            "cut_validation",
            {}
        )
    )

    enabled = bool(
        validation_cfg.get(
            "enabled",
            True
        )
    )

    if not enabled:

        return {
            "validated_cut_times_sec":
                list(candidate_cuts),

            "rejected_cut_times_sec":
                [],

            "details":
                [],
        }

    offset = float(
        validation_cfg.get(
            "sample_offset_sec",
            0.15
        )
    )

    changed_pixel_threshold = float(
        validation_cfg.get(
            "changed_pixel_threshold",
            0.10
        )
    )

    min_changed_pixel_ratio = float(
        validation_cfg.get(
            "min_changed_pixel_ratio",
            0.50
        )
    )

    min_mean_abs_diff = float(
        validation_cfg.get(
            "min_mean_abs_diff",
            0.055
        )
    )

    max_hist_corr = float(
        validation_cfg.get(
            "max_histogram_correlation",
            0.82
        )
    )

    histogram_min_diff = float(
        validation_cfg.get(
            "histogram_min_mean_abs_diff",
            0.035
        )
    )

    resize_width = int(
        cfg.get(
            "resize_width",
            160
        )
    )

    resize_height = int(
        cfg.get(
            "resize_height",
            90
        )
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
        )

    validated = []

    rejected = []

    details = []

    print(
        f"[R0] Validating "
        f"{len(candidate_cuts)} "
        f"candidate cuts..."
    )

    for cut_time in candidate_cuts:

        before_time = max(
            0.0,
            cut_time - offset
        )

        after_time = min(
            duration_sec,
            cut_time + offset
        )

        frame_before = read_frame_at(
            cap,
            before_time
        )

        frame_after = read_frame_at(
            cap,
            after_time
        )

        if (
            frame_before is None
            or
            frame_after is None
        ):

            rejected.append(
                cut_time
            )

            details.append(
                {
                    "cut_time_sec":
                        cut_time,

                    "accepted":
                        False,

                    "reason":
                        "frame_read_failed",
                }
            )

            continue

        metrics = compare_frames(
            frame_before=
                frame_before,

            frame_after=
                frame_after,

            resize_width=
                resize_width,

            resize_height=
                resize_height,

            changed_pixel_threshold=
                changed_pixel_threshold,
        )

        global_change_rule = (
            metrics[
                "changed_pixel_ratio"
            ]
            >=
            min_changed_pixel_ratio

            and

            metrics[
                "mean_abs_diff"
            ]
            >=
            min_mean_abs_diff
        )

        histogram_rule = (
            metrics[
                "histogram_correlation"
            ]
            <=
            max_hist_corr

            and

            metrics[
                "mean_abs_diff"
            ]
            >=
            histogram_min_diff
        )

        accepted = (
            global_change_rule
            or
            histogram_rule
        )

        if accepted:

            validated.append(
                cut_time
            )

        else:

            rejected.append(
                cut_time
            )

        details.append(
            {
                "cut_time_sec":
                    cut_time,

                "accepted":
                    accepted,

                "mean_abs_diff":
                    round(
                        metrics[
                            "mean_abs_diff"
                        ],
                        6
                    ),

                "changed_pixel_ratio":
                    round(
                        metrics[
                            "changed_pixel_ratio"
                        ],
                        6
                    ),

                "histogram_correlation":
                    round(
                        metrics[
                            "histogram_correlation"
                        ],
                        6
                    ),

                "global_change_rule":
                    global_change_rule,

                "histogram_rule":
                    histogram_rule,
            }
        )

    cap.release()

    return {
        "validated_cut_times_sec":
            validated,

        "rejected_cut_times_sec":
            rejected,

        "details":
            details,
    }


# ============================================================
# BUILD SHOTS FROM VALIDATED CUTS
# ============================================================

def build_shots_from_cuts(
    cut_times,
    duration_sec,
):

    boundaries = (
        [0.0]
        +
        sorted(
            float(x)
            for x
            in cut_times
        )
        +
        [duration_sec]
    )

    shots = []

    for i in range(
        len(boundaries) - 1
    ):

        start = boundaries[i]
        end = boundaries[i + 1]

        if end <= start:
            continue

        shots.append(
            {
                "shot_id":
                    len(shots) + 1,

                "start_sec":
                    round(
                        start,
                        3
                    ),

                "end_sec":
                    round(
                        end,
                        3
                    ),

                "duration_sec":
                    round(
                        end - start,
                        3
                    ),
            }
        )

    return shots


# ============================================================
# VISUAL CHANGE SAMPLING
# ============================================================

def sample_visual_change(
    video_path: str,
    cfg: dict,
):

    interval = float(
        cfg.get(
            "sample_interval_sec",
            0.5
        )
    )

    target_w = int(
        cfg.get(
            "resize_width",
            160
        )
    )

    target_h = int(
        cfg.get(
            "resize_height",
            90
        )
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0.0
    )

    deltas = []

    previous_gray = None

    previous_t = None

    t = 0.0

    while (
        t
        <=
        duration + 1e-6
    ):

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            t * 1000.0
        )

        ok, frame = cap.read()

        if not ok:

            t += interval
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (
                target_w,
                target_h
            ),

            interpolation=
                cv2.INTER_AREA
        )

        gray = (
            gray.astype(
                np.float32
            )
            /
            255.0
        )

        if previous_gray is not None:

            delta = float(
                np.mean(
                    np.abs(
                        gray
                        -
                        previous_gray
                    )
                )
            )

            deltas.append(
                {
                    "from_sec":
                        round(
                            previous_t,
                            3
                        ),

                    "to_sec":
                        round(
                            t,
                            3
                        ),

                    "delta":
                        delta,
                }
            )

        previous_gray = gray

        previous_t = t

        t += interval

    cap.release()

    values = [
        x["delta"]
        for x
        in deltas
    ]

    return {
        "sample_interval_sec":
            interval,

        "num_sampled_pairs":
            len(values),

        "mean_frame_delta":
            (
                float(
                    np.mean(values)
                )
                if values
                else 0.0
            ),

        "median_frame_delta":
            (
                float(
                    np.median(values)
                )
                if values
                else 0.0
            ),

        "p90_frame_delta":
            (
                float(
                    np.percentile(
                        values,
                        90
                    )
                )
                if values
                else 0.0
            ),

        "frame_deltas":
            deltas,
    }


# ============================================================
# STATISTICS
# ============================================================

def shot_statistics(
    shots,
    cut_times,
    duration_sec,
):

    durations = [
        float(
            x["duration_sec"]
        )
        for x
        in shots
    ]

    duration_minutes = max(
        duration_sec / 60.0,
        1e-9
    )

    return {
        "num_shots":
            len(shots),

        "num_cuts":
            len(cut_times),

        "cuts_per_minute":
            (
                len(cut_times)
                /
                duration_minutes
            ),

        "mean_shot_duration_sec":
            (
                float(
                    np.mean(
                        durations
                    )
                )
                if durations
                else 0.0
            ),

        "median_shot_duration_sec":
            (
                float(
                    np.median(
                        durations
                    )
                )
                if durations
                else 0.0
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video-id",
        required=True
    )

    args = parser.parse_args()

    cfg = get_video(
        args.video_id
    )

    analysis_cfg = (
        cfg["base"]["analysis"]
    )

    # --------------------------------------------------------
    # Video metadata
    # --------------------------------------------------------

    info = probe_video(
        cfg["video_path"]
    )

    # --------------------------------------------------------
    # Candidate cuts
    # --------------------------------------------------------

    candidate_data = (
        detect_candidate_cuts(
            video_path=
                cfg["video_path"],

            duration_sec=
                info["duration_sec"],

            cfg=
                analysis_cfg,
        )
    )

    candidate_cut_times = (
        candidate_data[
            "candidate_cut_times_sec"
        ]
    )

    # --------------------------------------------------------
    # Validate cuts
    # --------------------------------------------------------

    validation = (
        validate_candidate_cuts(
            video_path=
                cfg["video_path"],

            candidate_cuts=
                candidate_cut_times,

            duration_sec=
                info["duration_sec"],

            cfg=
                analysis_cfg,
        )
    )

    validated_cut_times = (
        validation[
            "validated_cut_times_sec"
        ]
    )

    # --------------------------------------------------------
    # Validated shot timeline
    # --------------------------------------------------------

    validated_shots = (
        build_shots_from_cuts(
            cut_times=
                validated_cut_times,

            duration_sec=
                info["duration_sec"],
        )
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    raw_stats = (
        shot_statistics(
            shots=
                candidate_data[
                    "raw_shots"
                ],

            cut_times=
                candidate_cut_times,

            duration_sec=
                info["duration_sec"],
        )
    )

    validated_stats = (
        shot_statistics(
            shots=
                validated_shots,

            cut_times=
                validated_cut_times,

            duration_sec=
                info["duration_sec"],
        )
    )

    # --------------------------------------------------------
    # Sparse visual-change statistics
    # --------------------------------------------------------

    visual = sample_visual_change(
        video_path=
            cfg["video_path"],

        cfg=
            analysis_cfg,
    )

    # --------------------------------------------------------
    # Final payload
    # --------------------------------------------------------

    payload = {

        "video_id":
            cfg["video_id"],

        "video_path":
            str(
                Path(
                    cfg["video_path"]
                ).as_posix()
            ),

        "metadata":
            info,

        "candidate_shot_analysis": {

            "detector":
                candidate_data[
                    "detector"
                ],

            **raw_stats,

            "cut_times_sec":
                candidate_cut_times,

            "shots":
                candidate_data[
                    "raw_shots"
                ],
        },

        "cut_validation": {

            "num_candidate_cuts":
                len(
                    candidate_cut_times
                ),

            "num_validated_cuts":
                len(
                    validated_cut_times
                ),

            "num_rejected_cuts":
                len(
                    validation[
                        "rejected_cut_times_sec"
                    ]
                ),

            "acceptance_ratio":
                (
                    len(
                        validated_cut_times
                    )
                    /
                    len(
                        candidate_cut_times
                    )

                    if
                    candidate_cut_times

                    else
                    0.0
                ),

            "validated_cut_times_sec":
                validated_cut_times,

            "rejected_cut_times_sec":
                validation[
                    "rejected_cut_times_sec"
                ],

            "details":
                validation[
                    "details"
                ],
        },

        "validated_shot_analysis": {

            **validated_stats,

            "cut_times_sec":
                validated_cut_times,

            "shots":
                validated_shots,
        },

        "visual_change":
            visual,
    }

    output_path = Path(
        f"data/analysis/"
        f"{cfg['video_id']}"
        f"_analysis.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2
        ),

        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        "R0 VIDEO ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        f"[R0 OK] Video ID             : "
        f"{cfg['video_id']}"
    )

    print(
        f"[R0 OK] Duration             : "
        f"{info['duration_sec']:.2f}s"
    )

    print(
        f"[R0 RAW] Candidate cuts       : "
        f"{len(candidate_cut_times)}"
    )

    print(
        f"[R0 RAW] Candidate cuts/min   : "
        f"{raw_stats['cuts_per_minute']:.3f}"
    )

    print(
        f"[R0 VAL] Validated cuts       : "
        f"{len(validated_cut_times)}"
    )

    print(
        f"[R0 VAL] Rejected cuts        : "
        f"{len(validation['rejected_cut_times_sec'])}"
    )

    print(
        f"[R0 VAL] Acceptance ratio     : "
        f"{payload['cut_validation']['acceptance_ratio']:.3f}"
    )

    print(
        f"[R0 VAL] Validated cuts/min   : "
        f"{validated_stats['cuts_per_minute']:.3f}"
    )

    print(
        f"[R0 VAL] Median shot          : "
        f"{validated_stats['median_shot_duration_sec']:.3f}s"
    )

    print(
        f"[R0 OK] Mean frame delta      : "
        f"{visual['mean_frame_delta']:.5f}"
    )

    print(
        f"[R0 OK] Output                : "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
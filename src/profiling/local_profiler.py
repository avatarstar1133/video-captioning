import argparse
import json
from pathlib import Path

import numpy as np

from src.config_utils import (
    get_video
)


# ============================================================
# PROFILING WINDOWS
# ============================================================

def generate_windows(
    duration_sec: float,
    window_sec: float,
    stride_sec: float,
):

    windows = []

    start = 0.0

    window_id = 1

    while start < duration_sec:

        end = min(
            duration_sec,
            start + window_sec
        )

        windows.append(
            {
                "window_id":
                    window_id,

                "start_sec":
                    start,

                "end_sec":
                    end,

                "duration_sec":
                    end - start,
            }
        )

        if end >= duration_sec:
            break

        start += stride_sec

        window_id += 1

    return windows


# ============================================================
# LOCAL CUTS
# ============================================================

def cut_times_in_window(
    cut_times,
    start,
    end,
):

    return [
        float(t)

        for t
        in cut_times

        if start < float(t) < end
    ]


# ============================================================
# LOCAL SHOT DURATIONS
# ============================================================

def local_shot_durations(
    start,
    end,
    cuts,
):

    boundaries = (
        [start]
        +
        list(cuts)
        +
        [end]
    )

    durations = []

    for a, b in zip(
        boundaries[:-1],
        boundaries[1:]
    ):

        if b > a:

            durations.append(
                b - a
            )

    return durations


# ============================================================
# LOCAL FRAME DELTAS
# ============================================================

def local_deltas(
    frame_deltas,
    start,
    end,
):

    values = []

    for item in frame_deltas:

        midpoint = (
            float(
                item["from_sec"]
            )
            +
            float(
                item["to_sec"]
            )
        ) / 2.0

        if (
            start
            <= midpoint
            < end
        ):

            values.append(
                float(
                    item["delta"]
                )
            )

    return values


# ============================================================
# REGIME CLASSIFICATION
# ============================================================

def classify_regime(
    features,
    cfg,
):

    cuts_per_min = (
        features[
            "validated_cuts_per_minute"
        ]
    )

    median_shot = (
        features[
            "median_validated_shot_duration_sec"
        ]
    )

    static_ratio = (
        features[
            "static_pair_ratio"
        ]
    )

    static_cfg = (
        cfg["static"]
    )

    high_cfg = (
        cfg["high_cut"]
    )

    low_cfg = (
        cfg["low_cut"]
    )

    # --------------------------------------------------------
    # STATIC
    # --------------------------------------------------------

    if (
        cuts_per_min
        <=
        float(
            static_cfg[
                "max_validated_cuts_per_min"
            ]
        )

        and

        static_ratio
        >=
        float(
            static_cfg[
                "min_static_pair_ratio"
            ]
        )
    ):

        return "static"

    # --------------------------------------------------------
    # HIGH CUT
    #
    # Important:
    # High cut requires BOTH actual validated editing activity
    # and a scene timeline dominated by short shots.
    #
    # A highly static CCTV sequence should NOT become high_cut
    # only because a detector generated false candidates.
    # --------------------------------------------------------

    if (
        cuts_per_min
        >=
        float(
            high_cfg[
                "min_validated_cuts_per_min"
            ]
        )

        and

        median_shot
        <=
        float(
            high_cfg[
                "max_median_shot_sec"
            ]
        )

        and

        static_ratio
        <=
        float(
            high_cfg[
                "max_static_pair_ratio"
            ]
        )
    ):

        return "high_cut"

    # --------------------------------------------------------
    # LOW CUT
    # --------------------------------------------------------

    if (
        cuts_per_min
        <=
        float(
            low_cfg[
                "max_validated_cuts_per_min"
            ]
        )

        or

        median_shot
        >=
        float(
            low_cfg[
                "min_median_shot_sec"
            ]
        )
    ):

        return "low_cut"

    # --------------------------------------------------------
    # OTHERWISE
    # --------------------------------------------------------

    return "medium_cut"


# ============================================================
# DIAGNOSTIC SCORES
# ============================================================

def diagnostic_scores(
    features,
    cfg,
):

    cpm = float(
        features[
            "validated_cuts_per_minute"
        ]
    )

    median_shot = max(
        float(
            features[
                "median_validated_shot_duration_sec"
            ]
        ),
        1e-6
    )

    static_ratio = float(
        features[
            "static_pair_ratio"
        ]
    )

    static_max_cpm = float(
        cfg["static"][
            "max_validated_cuts_per_min"
        ]
    )

    high_min_cpm = float(
        cfg["high_cut"][
            "min_validated_cuts_per_min"
        ]
    )

    low_max_cpm = float(
        cfg["low_cut"][
            "max_validated_cuts_per_min"
        ]
    )

    high_max_median = float(
        cfg["high_cut"][
            "max_median_shot_sec"
        ]
    )

    low_min_median = float(
        cfg["low_cut"][
            "min_median_shot_sec"
        ]
    )

    static_score = (
        0.70
        *
        static_ratio

        +

        0.30
        *
        (
            1.0
            -
            min(
                cpm
                /
                max(
                    static_max_cpm,
                    1e-6
                ),
                1.0
            )
        )
    )

    high_score = (
        0.50
        *
        min(
            cpm
            /
            max(
                high_min_cpm,
                1e-6
            ),
            1.0
        )

        +

        0.30
        *
        min(
            high_max_median
            /
            median_shot,
            1.0
        )

        +

        0.20
        *
        (
            1.0
            -
            static_ratio
        )
    )

    low_score = (
        0.50
        *
        (
            1.0
            -
            min(
                cpm
                /
                max(
                    low_max_cpm,
                    1e-6
                ),
                1.0
            )
        )

        +

        0.30
        *
        min(
            median_shot
            /
            max(
                low_min_median,
                1e-6
            ),
            1.0
        )

        +

        0.20
        *
        static_ratio
    )

    static_score = min(
        max(
            static_score,
            0.0
        ),
        1.0
    )

    high_score = min(
        max(
            high_score,
            0.0
        ),
        1.0
    )

    low_score = min(
        max(
            low_score,
            0.0
        ),
        1.0
    )

    medium_score = max(
        0.0,
        1.0
        -
        max(
            static_score,
            high_score,
            low_score
        )
        * 0.85
    )

    return {
        "static":
            static_score,

        "low_cut":
            low_score,

        "medium_cut":
            medium_score,

        "high_cut":
            high_score,
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

    profiling_cfg = (
        cfg["base"][
            "profiling"
        ]
    )

    analysis_path = Path(
        f"data/analysis/"
        f"{cfg['video_id']}"
        f"_analysis.json"
    )

    if not analysis_path.exists():

        raise FileNotFoundError(
            f"R0 output not found:\n"
            f"{analysis_path.resolve()}\n"
            f"Run R0 first."
        )

    analysis = json.loads(
        analysis_path.read_text(
            encoding="utf-8"
        )
    )

    duration_sec = float(
        analysis[
            "metadata"
        ][
            "duration_sec"
        ]
    )

    window_sec = float(
        profiling_cfg[
            "window_sec"
        ]
    )

    stride_sec = float(
        profiling_cfg[
            "stride_sec"
        ]
    )

    static_delta_threshold = float(
        profiling_cfg[
            "static_delta_threshold"
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use VALIDATED cuts, not raw candidate cuts.
    # --------------------------------------------------------

    validated_cut_times = (
        analysis[
            "validated_shot_analysis"
        ][
            "cut_times_sec"
        ]
    )

    candidate_cut_times = (
        analysis[
            "candidate_shot_analysis"
        ][
            "cut_times_sec"
        ]
    )

    frame_deltas = (
        analysis[
            "visual_change"
        ][
            "frame_deltas"
        ]
    )

    windows = generate_windows(
        duration_sec=
            duration_sec,

        window_sec=
            window_sec,

        stride_sec=
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

        start = float(
            window[
                "start_sec"
            ]
        )

        end = float(
            window[
                "end_sec"
            ]
        )

        duration = float(
            window[
                "duration_sec"
            ]
        )

        # ----------------------------------------------------
        # Candidate cuts for diagnostics
        # ----------------------------------------------------

        local_candidate_cuts = (
            cut_times_in_window(
                candidate_cut_times,
                start,
                end,
            )
        )

        # ----------------------------------------------------
        # Validated cuts used by algorithm
        # ----------------------------------------------------

        local_validated_cuts = (
            cut_times_in_window(
                validated_cut_times,
                start,
                end,
            )
        )

        shot_durations = (
            local_shot_durations(
                start=
                    start,

                end=
                    end,

                cuts=
                    local_validated_cuts,
            )
        )

        deltas = local_deltas(
            frame_deltas=
                frame_deltas,

            start=
                start,

            end=
                end,
        )

        candidate_cpm = (
            len(
                local_candidate_cuts
            )
            /
            max(
                duration / 60.0,
                1e-9
            )
        )

        validated_cpm = (
            len(
                local_validated_cuts
            )
            /
            max(
                duration / 60.0,
                1e-9
            )
        )

        median_shot = (
            float(
                np.median(
                    shot_durations
                )
            )

            if shot_durations

            else duration
        )

        mean_shot = (
            float(
                np.mean(
                    shot_durations
                )
            )

            if shot_durations

            else duration
        )

        static_pair_ratio = (
            sum(
                delta
                <
                static_delta_threshold

                for delta
                in deltas
            )
            /
            len(deltas)

            if deltas

            else 1.0
        )

        mean_delta = (
            float(
                np.mean(
                    deltas
                )
            )

            if deltas

            else 0.0
        )

        p90_delta = (
            float(
                np.percentile(
                    deltas,
                    90
                )
            )

            if deltas

            else 0.0
        )

        features = {

            "candidate_cut_count":
                len(
                    local_candidate_cuts
                ),

            "validated_cut_count":
                len(
                    local_validated_cuts
                ),

            "candidate_cuts_per_minute":
                candidate_cpm,

            "validated_cuts_per_minute":
                validated_cpm,

            "mean_validated_shot_duration_sec":
                mean_shot,

            "median_validated_shot_duration_sec":
                median_shot,

            "mean_frame_delta":
                mean_delta,

            "p90_frame_delta":
                p90_delta,

            "static_pair_ratio":
                static_pair_ratio,
        }

        regime = classify_regime(
            features=
                features,

            cfg=
                profiling_cfg,
        )

        scores = diagnostic_scores(
            features=
                features,

            cfg=
                profiling_cfg,
        )

        counts[
            regime
        ] += 1

        local_profiles.append(
            {
                **window,

                "features": {
                    key:
                        round(
                            value,
                            6
                        )

                    if isinstance(
                        value,
                        float
                    )

                    else value

                    for key, value
                    in features.items()
                },

                "regime":
                    regime,

                "diagnostic_scores": {
                    key:
                        round(
                            value,
                            6
                        )

                    for key, value
                    in scores.items()
                },
            }
        )

    payload = {

        "video_id":
            cfg["video_id"],

        "video_path":
            analysis[
                "video_path"
            ],

        "profiling": {

            "window_sec":
                window_sec,

            "stride_sec":
                stride_sec,

            "overlapping":
                (
                    stride_sec
                    <
                    window_sec
                ),

            "static_delta_threshold":
                static_delta_threshold,

            "cut_source":
                "validated_cuts",
        },

        "num_windows":
            len(
                local_profiles
            ),

        "regime_counts":
            counts,

        "windows":
            local_profiles,

        "notes": [

            "Regimes represent local temporal dynamics, not semantic video categories.",

            "R1 uses validated shot cuts rather than raw SceneDetect candidates.",

            "Profiling windows are not final retrieval segments.",

            "Diagnostic scores are heuristic scores and are not calibrated probabilities.",
        ],
    }

    output_path = Path(
        f"data/profiles/"
        f"{cfg['video_id']}"
        f"_local_profile.json"
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
    # Console output
    # --------------------------------------------------------

    print(
        f"\n[R1 OK] Video ID       : "
        f"{cfg['video_id']}"
    )

    print(
        f"[R1 OK] Windows        : "
        f"{len(local_profiles)}"
    )

    print(
        f"[R1 OK] Regime counts  : "
        f"{counts}"
    )

    print(
        f"[R1 OK] Output         : "
        f"{output_path.resolve()}"
    )

    print(
        "\nFirst local windows:"
    )

    for item in (
        local_profiles[:10]
    ):

        f = (
            item[
                "features"
            ]
        )

        print(
            f"  "
            f"{item['start_sec']:7.1f}"
            f"-"
            f"{item['end_sec']:7.1f}s "
            f"| "
            f"{item['regime']:10s} "
            f"| raw="
            f"{f['candidate_cuts_per_minute']:.2f}/min "
            f"| val="
            f"{f['validated_cuts_per_minute']:.2f}/min "
            f"| median="
            f"{f['median_validated_shot_duration_sec']:.2f}s "
            f"| static="
            f"{f['static_pair_ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, AutoProcessor

from src.config_utils import get_video


# ============================================================
# CONFIG DEFAULTS
# ============================================================

DEFAULT_POSITIONS = [0.25, 0.50, 0.75]
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_BATCH_SIZE = 32
DEFAULT_JPEG_QUALITY = 92


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n  {path.resolve()}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def l2_normalize_np(matrix: np.ndarray):
    norms = np.linalg.norm(
        matrix,
        axis=-1,
        keepdims=True
    )

    norms = np.clip(
        norms,
        1e-12,
        None
    )

    return matrix / norms


def get_representation_config(base_cfg: dict):
    cfg = base_cfg.get(
        "representation",
        {}
    )

    positions = cfg.get(
        "representative_positions",
        DEFAULT_POSITIONS
    )

    positions = [
        float(x)
        for x in positions
    ]

    for p in positions:
        if not 0.0 < p < 1.0:
            raise ValueError(
                "representative_positions must be strictly "
                "between 0 and 1."
            )

    if not positions:
        raise ValueError(
            "At least one representative position is required."
        )

    return {
        "representative_positions": positions,

        "model_name":
            str(
                cfg.get(
                    "model_name",
                    DEFAULT_MODEL
                )
            ),

        "batch_size":
            int(
                cfg.get(
                    "batch_size",
                    DEFAULT_BATCH_SIZE
                )
            ),

        "jpeg_quality":
            int(
                cfg.get(
                    "jpeg_quality",
                    DEFAULT_JPEG_QUALITY
                )
            ),

        "save_representative_frames":
            bool(
                cfg.get(
                    "save_representative_frames",
                    True
                )
            ),
    }


# ============================================================
# FRAME EXTRACTION
# ============================================================

def representative_timestamp(
    start_sec: float,
    end_sec: float,
    position: float,
):
    duration = end_sec - start_sec

    return (
        start_sec
        +
        duration * position
    )


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


def frame_to_pil(frame):
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(rgb)


def save_frame(
    frame,
    output_path: Path,
    jpeg_quality: int,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ok = cv2.imwrite(
        str(output_path),
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            int(jpeg_quality)
        ]
    )

    if not ok:
        raise RuntimeError(
            f"Could not save frame: "
            f"{output_path.resolve()}"
        )


def extract_representative_frames(
    video_id: str,
    video_path: str,
    segments,
    positions,
    jpeg_quality: int,
    save_images: bool,
):
    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video:\n"
            f"  {Path(video_path).resolve()}"
        )

    samples = []

    failed_reads = []

    for segment in segments:
        segment_id = int(
            segment["segment_id"]
        )

        start_sec = float(
            segment["start_sec"]
        )

        end_sec = float(
            segment["end_sec"]
        )

        for position in positions:
            timestamp_sec = (
                representative_timestamp(
                    start_sec,
                    end_sec,
                    position
                )
            )

            frame = read_frame_at(
                cap,
                timestamp_sec
            )

            # Rare decoder-seek failures can occur near boundaries.
            # Fall back to the midpoint of the segment.
            used_fallback = False

            if frame is None:
                fallback_sec = (
                    start_sec + end_sec
                ) / 2.0

                frame = read_frame_at(
                    cap,
                    fallback_sec
                )

                used_fallback = True

                timestamp_sec = (
                    fallback_sec
                )

            if frame is None:
                failed_reads.append(
                    {
                        "segment_id":
                            segment_id,

                        "position":
                            position,

                        "timestamp_sec":
                            timestamp_sec,
                    }
                )

                continue

            position_label = int(
                round(
                    position * 100
                )
            )

            frame_path = Path(
                "data/frames/"
                f"{video_id}_"
                f"seg_{segment_id:05d}_"
                f"p{position_label:02d}.jpg"
            )

            if save_images:
                save_frame(
                    frame=frame,

                    output_path=
                        frame_path,

                    jpeg_quality=
                        jpeg_quality,
                )

            samples.append(
                {
                    "segment_id":
                        segment_id,

                    "position":
                        position,

                    "position_label":
                        f"p{position_label:02d}",

                    "timestamp_sec":
                        round(
                            timestamp_sec,
                            6
                        ),

                    "frame_path":
                        (
                            frame_path.as_posix()
                            if save_images
                            else None
                        ),

                    "fallback_used":
                        used_fallback,

                    "image":
                        frame_to_pil(
                            frame
                        ),
                }
            )

    cap.release()

    return samples, failed_reads


# ============================================================
# CLIP IMAGE ENCODER
# ============================================================

def load_clip(
    model_name: str,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[R3] Device     : {device}"
    )

    print(
        f"[R3] CLIP model : {model_name}"
    )

    model = (
        CLIPModel
        .from_pretrained(
            model_name
        )
        .to(device)
        .eval()
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            model_name
        )
    )

    return (
        model,
        processor,
        device
    )


def encode_images(
    samples,
    model,
    processor,
    device,
    batch_size: int,
):
    all_embeddings = []

    total = len(samples)

    for start in range(
        0,
        total,
        batch_size
    ):
        end = min(
            total,
            start + batch_size
        )

        batch = samples[
            start:end
        ]

        images = [
            item["image"]
            for item
            in batch
        ]

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = (
            inputs[
                "pixel_values"
            ]
            .to(device)
        )

        with torch.inference_mode():
            vision_outputs = (
                model.vision_model(
                    pixel_values=
                        pixel_values,

                    return_dict=True
                )
            )

            pooled_output = (
                vision_outputs
                .pooler_output
            )

            features = (
                model
                .visual_projection(
                    pooled_output
                )
            )

            features = (
                features
                /
                features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        all_embeddings.append(
            features
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        print(
            f"[R3] Encoded "
            f"{end}/{total} frames"
        )

    if not all_embeddings:
        raise RuntimeError(
            "No representative frames "
            "were encoded."
        )

    return np.concatenate(
        all_embeddings,
        axis=0
    )


# ============================================================
# SEGMENT POOLING
# ============================================================

def pool_segment_embeddings(
    segments,
    samples,
    frame_embeddings,
):
    by_segment = {}

    for sample_index, sample in enumerate(
        samples
    ):
        segment_id = int(
            sample["segment_id"]
        )

        by_segment.setdefault(
            segment_id,
            []
        ).append(
            sample_index
        )

    segment_vectors = []

    items = []

    missing_segments = []

    for segment in segments:
        segment_id = int(
            segment["segment_id"]
        )

        indices = by_segment.get(
            segment_id,
            []
        )

        if not indices:
            missing_segments.append(
                segment_id
            )
            continue

        vectors = frame_embeddings[
            indices
        ]

        pooled = np.mean(
            vectors,
            axis=0,
            keepdims=True
        )

        pooled = l2_normalize_np(
            pooled
        )[0].astype(
            np.float32
        )

        segment_vectors.append(
            pooled
        )

        reps = []

        for idx in indices:
            sample = samples[idx]

            reps.append(
                {
                    "position":
                        sample["position"],

                    "position_label":
                        sample[
                            "position_label"
                        ],

                    "timestamp_sec":
                        sample[
                            "timestamp_sec"
                        ],

                    "frame_path":
                        sample[
                            "frame_path"
                        ],

                    "fallback_used":
                        sample[
                            "fallback_used"
                        ],
                }
            )

        items.append(
            {
                "segment_id":
                    segment_id,

                "start_sec":
                    float(
                        segment[
                            "start_sec"
                        ]
                    ),

                "end_sec":
                    float(
                        segment[
                            "end_sec"
                        ]
                    ),

                "duration_sec":
                    float(
                        segment[
                            "duration_sec"
                        ]
                    ),

                "regime":
                    segment[
                        "regime"
                    ],

                "policy_mode":
                    segment[
                        "policy_mode"
                    ],

                "representative_frames":
                    reps,

                "num_representative_frames":
                    len(reps),
            }
        )

    if missing_segments:
        raise RuntimeError(
            "Some segments have no usable "
            "representative frames: "
            f"{missing_segments[:20]}"
        )

    matrix = np.stack(
        segment_vectors,
        axis=0
    )

    return (
        matrix,
        items
    )


# ============================================================
# INDEX VALIDATION
# ============================================================

def validate_index(
    matrix,
    items,
    expected_segments: int,
):
    problems = []

    if matrix.shape[0] != expected_segments:
        problems.append(
            "Embedding row count does not "
            "match segment count."
        )

    if len(items) != expected_segments:
        problems.append(
            "Item count does not match "
            "segment count."
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        problems.append(
            "Embedding matrix contains "
            "NaN or Inf."
        )

    norms = np.linalg.norm(
        matrix,
        axis=1
    )

    if len(norms):
        max_norm_error = float(
            np.max(
                np.abs(
                    norms - 1.0
                )
            )
        )
    else:
        max_norm_error = 0.0

    if max_norm_error > 1e-3:
        problems.append(
            "Segment embeddings are not "
            "properly L2-normalized."
        )

    return {
        "valid":
            len(problems) == 0,

        "problems":
            problems,

        "max_norm_error":
            max_norm_error,
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

    rep_cfg = (
        get_representation_config(
            cfg["base"]
        )
    )

    segments_path = Path(
        "data/segments/"
        f"{cfg['video_id']}"
        "_segments.json"
    )

    segment_payload = load_json(
        segments_path
    )

    segments = (
        segment_payload[
            "segments"
        ]
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "R3 MULTI-FRAME SEGMENT REPRESENTATION"
    )

    print(
        "========================================"
    )

    print(
        f"[R3] Video ID    : "
        f"{cfg['video_id']}"
    )

    print(
        f"[R3] Segments    : "
        f"{len(segments)}"
    )

    print(
        f"[R3] Positions   : "
        f"{rep_cfg['representative_positions']}"
    )

    # --------------------------------------------------------
    # Extract representative frames.
    # --------------------------------------------------------

    samples, failed_reads = (
        extract_representative_frames(
            video_id=
                cfg["video_id"],

            video_path=
                cfg[
                    "video_path"
                ],

            segments=
                segments,

            positions=
                rep_cfg[
                    "representative_positions"
                ],

            jpeg_quality=
                rep_cfg[
                    "jpeg_quality"
                ],

            save_images=
                rep_cfg[
                    "save_representative_frames"
                ],
        )
    )

    print(
        f"[R3] Extracted   : "
        f"{len(samples)} "
        f"representative frames"
    )

    if failed_reads:
        print(
            f"[R3 WARN] Failed reads: "
            f"{len(failed_reads)}"
        )

    # --------------------------------------------------------
    # CLIP encode all representative frames.
    # --------------------------------------------------------

    model, processor, device = (
        load_clip(
            rep_cfg[
                "model_name"
            ]
        )
    )

    frame_embeddings = (
        encode_images(
            samples=
                samples,

            model=
                model,

            processor=
                processor,

            device=
                device,

            batch_size=
                rep_cfg[
                    "batch_size"
                ],
        )
    )

    # --------------------------------------------------------
    # Mean pooling per segment + L2 normalization.
    # --------------------------------------------------------

    segment_embeddings, items = (
        pool_segment_embeddings(
            segments=
                segments,

            samples=
                samples,

            frame_embeddings=
                frame_embeddings,
        )
    )

    validation = validate_index(
        matrix=
            segment_embeddings,

        items=
            items,

        expected_segments=
            len(segments),
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    index_dir = Path(
        "data/index"
    )

    index_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings_path = (
        index_dir
        /
        f"{cfg['video_id']}"
        f"_segment_embeddings.npy"
    )

    items_path = (
        index_dir
        /
        f"{cfg['video_id']}"
        f"_segment_items.json"
    )

    info_path = (
        index_dir
        /
        f"{cfg['video_id']}"
        f"_segment_index_info.json"
    )

    np.save(
        embeddings_path,
        segment_embeddings
    )

    items_path.write_text(
        json.dumps(
            items,
            indent=2
        ),
        encoding="utf-8",
    )

    info = {
        "video_id":
            cfg["video_id"],

        "video_path":
            str(
                Path(
                    cfg[
                        "video_path"
                    ]
                ).as_posix()
            ),

        "source_segments":
            segments_path.as_posix(),

        "method":
            "multi_frame_mean_pooling",

        "encoder":
            rep_cfg[
                "model_name"
            ],

        "representative_positions":
            rep_cfg[
                "representative_positions"
            ],

        "num_segments":
            len(
                segments
            ),

        "num_representative_frames":
            len(
                samples
            ),

        "embedding_shape":
            list(
                segment_embeddings.shape
            ),

        "embedding_dim":
            int(
                segment_embeddings.shape[1]
            ),

        "failed_frame_reads":
            failed_reads,

        "validation":
            validation,
    }

    info_path.write_text(
        json.dumps(
            info,
            indent=2
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console report.
    # --------------------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        "R3 RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"[R3 OK] Segment embeddings : "
        f"{segment_embeddings.shape}"
    )

    print(
        f"[R3 OK] Expected segments   : "
        f"{len(segments)}"
    )

    print(
        f"[R3 OK] Index valid         : "
        f"{validation['valid']}"
    )

    print(
        f"[R3 OK] Max norm error      : "
        f"{validation['max_norm_error']:.8f}"
    )

    if validation["problems"]:
        print(
            "[R3 WARN] Problems:"
        )

        for problem in (
            validation[
                "problems"
            ]
        ):
            print(
                f"  - {problem}"
            )

    print(
        f"[R3 OK] Embeddings          : "
        f"{embeddings_path.resolve()}"
    )

    print(
        f"[R3 OK] Items               : "
        f"{items_path.resolve()}"
    )

    print(
        f"[R3 OK] Index info          : "
        f"{info_path.resolve()}"
    )


if __name__ == "__main__":
    main()

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, AutoProcessor

from src.config_utils import get_video


DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_TOP_K = 5


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


def seconds_to_hhmmss(seconds: float):
    seconds = max(
        0,
        int(round(seconds))
    )

    hours = (
        seconds // 3600
    )

    minutes = (
        (seconds % 3600)
        // 60
    )

    secs = (
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def get_retrieval_config(base_cfg: dict):
    representation_cfg = (
        base_cfg.get(
            "representation",
            {}
        )
    )

    retrieval_cfg = (
        base_cfg.get(
            "retrieval",
            {}
        )
    )

    model_name = str(
        retrieval_cfg.get(
            "model_name",
            representation_cfg.get(
                "model_name",
                DEFAULT_MODEL
            )
        )
    )

    top_k = int(
        retrieval_cfg.get(
            "top_k_segments",
            DEFAULT_TOP_K
        )
    )

    return {
        "model_name": model_name,
        "top_k_segments": top_k,
    }


def load_segment_index(video_id: str):
    index_dir = Path(
        "data/index"
    )

    embeddings_path = (
        index_dir
        /
        f"{video_id}"
        f"_segment_embeddings.npy"
    )

    items_path = (
        index_dir
        /
        f"{video_id}"
        f"_segment_items.json"
    )

    info_path = (
        index_dir
        /
        f"{video_id}"
        f"_segment_index_info.json"
    )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Segment embeddings not found:\n"
            f"  {embeddings_path.resolve()}\n"
            f"Run R3 first."
        )

    if not items_path.exists():
        raise FileNotFoundError(
            f"Segment items not found:\n"
            f"  {items_path.resolve()}\n"
            f"Run R3 first."
        )

    embeddings = np.load(
        embeddings_path
    ).astype(
        np.float32
    )

    items = load_json(
        items_path
    )

    info = (
        load_json(
            info_path
        )
        if info_path.exists()
        else {}
    )

    if embeddings.ndim != 2:
        raise RuntimeError(
            "Segment embedding matrix "
            "must be 2-dimensional."
        )

    if embeddings.shape[0] != len(items):
        raise RuntimeError(
            "Embedding row count does not "
            "match segment item count."
        )

    return {
        "embeddings": embeddings,
        "items": items,
        "info": info,
        "embeddings_path": embeddings_path,
        "items_path": items_path,
    }


def load_clip(
    model_name: str
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[R4] Device     : {device}"
    )

    print(
        f"[R4] CLIP model : {model_name}"
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


def encode_text(
    query: str,
    model,
    processor,
    device
):
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    input_ids = (
        inputs[
            "input_ids"
        ]
        .to(device)
    )

    attention_mask = (
        inputs.get(
            "attention_mask"
        )
    )

    if attention_mask is not None:
        attention_mask = (
            attention_mask
            .to(device)
        )

    with torch.inference_mode():
        text_outputs = (
            model.text_model(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,

                return_dict=True
            )
        )

        pooled_output = (
            text_outputs
            .pooler_output
        )

        text_features = (
            model
            .text_projection(
                pooled_output
            )
        )

        text_features = (
            text_features
            /
            text_features.norm(
                dim=-1,
                keepdim=True
            )
        )

    return (
        text_features
        .cpu()
        .numpy()
        .astype(
            np.float32
        )[0]
    )


def retrieve_segments(
    query: str,
    video_id: str,
    top_k: int = DEFAULT_TOP_K,
):
    cfg = get_video(
        video_id
    )

    retrieval_cfg = (
        get_retrieval_config(
            cfg["base"]
        )
    )

    index = load_segment_index(
        video_id
    )

    embeddings = (
        index[
            "embeddings"
        ]
    )

    items = (
        index[
            "items"
        ]
    )

    model, processor, device = (
        load_clip(
            retrieval_cfg[
                "model_name"
            ]
        )
    )

    start_time = (
        time.perf_counter()
    )

    query_embedding = encode_text(
        query=query,
        model=model,
        processor=processor,
        device=device,
    )

    if (
        embeddings.shape[1]
        !=
        query_embedding.shape[0]
    ):
        raise RuntimeError(
            "Query embedding dimension "
            "does not match segment index."
        )

    scores = (
        embeddings
        @
        query_embedding
    )

    requested_top_k = int(
        top_k
    )

    effective_top_k = min(
        requested_top_k,
        len(items)
    )

    order = (
        np.argsort(
            -scores
        )[
            :effective_top_k
        ]
    )

    latency_ms = (
        time.perf_counter()
        -
        start_time
    ) * 1000.0

    results = []

    for rank, idx in enumerate(
        order,
        start=1
    ):
        idx = int(
            idx
        )

        item = dict(
            items[idx]
        )

        start_sec = float(
            item[
                "start_sec"
            ]
        )

        end_sec = float(
            item[
                "end_sec"
            ]
        )

        result = {
            "rank": rank,

            "segment_index":
                idx,

            "segment_id":
                int(
                    item[
                        "segment_id"
                    ]
                ),

            "score":
                float(
                    scores[idx]
                ),

            "start_sec":
                start_sec,

            "end_sec":
                end_sec,

            "start_time":
                seconds_to_hhmmss(
                    start_sec
                ),

            "end_time":
                seconds_to_hhmmss(
                    end_sec
                ),

            "duration_sec":
                float(
                    item[
                        "duration_sec"
                    ]
                ),

            "regime":
                item[
                    "regime"
                ],

            "policy_mode":
                item[
                    "policy_mode"
                ],

            "representative_frames":
                item[
                    "representative_frames"
                ],
        }

        results.append(
            result
        )

    payload = {
        "video_id":
            video_id,

        "video_path":
            cfg[
                "video_path"
            ],

        "query":
            query,

        "method":
            "clip_segment_retrieval",

        "encoder":
            retrieval_cfg[
                "model_name"
            ],

        "requested_top_k":
            requested_top_k,

        "effective_top_k":
            effective_top_k,

        "num_indexed_segments":
            len(
                items
            ),

        "embedding_dim":
            int(
                embeddings.shape[1]
            ),

        "latency_ms":
            latency_ms,

        "results":
            results,
    }

    return payload


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video-id",
        required=True
    )

    parser.add_argument(
        "--query",
        required=True
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None
    )

    parser.add_argument(
        "--output",
        default=None
    )

    args = parser.parse_args()

    cfg = get_video(
        args.video_id
    )

    retrieval_cfg = (
        get_retrieval_config(
            cfg["base"]
        )
    )

    top_k = (
        args.top_k
        if args.top_k is not None
        else retrieval_cfg[
            "top_k_segments"
        ]
    )

    payload = retrieve_segments(
        query=
            args.query,

        video_id=
            args.video_id,

        top_k=
            top_k,
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "R4 TEXT QUERY -> SEGMENT RETRIEVAL"
    )

    print(
        "========================================"
    )

    print(
        f"[R4] Video ID       : "
        f"{payload['video_id']}"
    )

    print(
        f"[R4] Query          : "
        f"{payload['query']}"
    )

    print(
        f"[R4] Indexed segments: "
        f"{payload['num_indexed_segments']}"
    )

    print(
        f"[R4] Top-K          : "
        f"{payload['effective_top_k']}"
    )

    print(
        "\nTop segments:"
    )

    for result in (
        payload[
            "results"
        ]
    ):
        print(
            f"  "
            f"#{result['rank']:02d} "
            f"| score="
            f"{result['score']:.4f} "
            f"| seg="
            f"{result['segment_id']:04d} "
            f"| "
            f"{result['start_time']}"
            f" -> "
            f"{result['end_time']} "
            f"| "
            f"{result['regime']:10s} "
            f"| "
            f"{result['duration_sec']:.2f}s"
        )

    print(
        f"\n[R4] Query latency  : "
        f"{payload['latency_ms']:.2f} ms"
    )

    output_path = (
        Path(
            args.output
        )
        if args.output
        else
        Path(
            "data/results/"
            f"{args.video_id}"
            "_last_query.json"
        )
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
        encoding="utf-8",
    )

    print(
        f"[R4 OK] Output       : "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()

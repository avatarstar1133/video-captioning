from pathlib import Path
import yaml

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"
}


def load_yaml(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_video(video_id: str,
              base_config_path: str = "configs/base.yaml",
              videos_config_path: str = "configs/videos.yaml"):
    base = load_yaml(base_config_path)
    videos = load_yaml(videos_config_path).get("videos", {})

    if video_id not in videos:
        available = ", ".join(sorted(videos)) or "(none)"
        raise KeyError(
            f"Unknown video_id '{video_id}'. Available: {available}"
        )

    path = Path(videos[video_id]["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Video not found for '{video_id}':\n  {path.resolve()}\n"
            f"Update configs/videos.yaml."
        )

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension: {path.suffix}"
        )

    return {
        "video_id": video_id,
        "video_path": str(path),
        "base": base,
    }


def list_videos(videos_config_path: str = "configs/videos.yaml"):
    return load_yaml(videos_config_path).get("videos", {})

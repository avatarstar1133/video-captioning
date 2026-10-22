import argparse
import subprocess
import sys

from src.config_utils import list_videos


def run(cmd):
    print("\n>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="HQVR R0 + R1: global video analysis and local temporal profiling."
    )
    parser.add_argument("--video-id")
    parser.add_argument("--list-videos", action="store_true")
    args = parser.parse_args()

    if args.list_videos:
        videos = list_videos()
        print("Registered videos:")
        for video_id, entry in videos.items():
            print(f"  {video_id:16s} path={entry['path']}")
        return

    if not args.video_id:
        parser.error(
            "--video-id is required unless --list-videos is used."
        )

    run([
        sys.executable,
        "-m",
        "src.analysis.video_analysis",
        "--video-id",
        args.video_id,
    ])

    run([
        sys.executable,
        "-m",
        "src.profiling.local_profiler",
        "--video-id",
        args.video_id,
    ])

    print(
        f"\n[DONE] R0 + R1 completed for video_id='{args.video_id}'."
    )


if __name__ == "__main__":
    main()

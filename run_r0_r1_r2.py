import argparse
import subprocess
import sys


def run(cmd):
    print("\n>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()

    run([sys.executable, "-m", "src.analysis.video_analysis", "--video-id", args.video_id])
    run([sys.executable, "-m", "src.profiling.local_profiler", "--video-id", args.video_id])
    run([sys.executable, "-m", "src.segmentation.adaptive_segmenter", "--video-id", args.video_id])

    print(f"\n[DONE] R0 + R1 + R2 completed for '{args.video_id}'.")


if __name__ == "__main__":
    main()

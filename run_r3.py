import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video-id",
        required=True
    )

    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.representation.build_segment_index",
        "--video-id",
        args.video_id,
    ]

    print(
        ">",
        " ".join(cmd)
    )

    subprocess.run(
        cmd,
        check=True
    )


if __name__ == "__main__":
    main()

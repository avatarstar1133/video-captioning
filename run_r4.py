import argparse
import subprocess
import sys


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
        default=5
    )

    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.retrieval.query_segments",
        "--video-id",
        args.video_id,
        "--query",
        args.query,
        "--top-k",
        str(args.top_k),
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

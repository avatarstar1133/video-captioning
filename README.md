# HQVR — R0 + R1

This build starts from the revised architecture:

R0 — Global Video Analysis
R1 — Local Temporal Profiling
R2 — Adaptive Normalized Segmentation (next stage)

The algorithm does NOT label the whole video as vlog/documentary/fast-cut/CCTV.
Those remain dataset domains only.

Instead, each selected video is analyzed as a sequence of overlapping temporal
windows and each local region is assigned a temporal regime:

- static
- low_cut
- medium_cut
- high_cut

This allows one video to contain several different temporal styles.

## Recommended folder layout

Keep heavy videos outside the repository:

video captioning/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   ├── video_003.mp4
│   └── video_004.mp4
└── HQVR_R0_R1_Local_Profiling/

Update configs/videos.yaml if paths or filenames differ.

## Install

```powershell
pip install -r requirements.txt
```

## List registered videos

```powershell
python run_r0_r1.py --list-videos
```

## Run one selected video

```powershell
python run_r0_r1.py --video-id video_001
```

## R0 output

```text
data/analysis/video_001_analysis.json
```

Contains:
- duration, FPS, resolution
- shot boundaries
- cuts per minute
- global shot duration statistics
- sparse frame-difference measurements

## R1 output

```text
data/profiles/video_001_local_profile.json
```

Default local profiling:
- 60-second windows
- 30-second stride
- overlapping windows
- local cut rate
- local median/mean shot duration
- local frame-difference statistics
- static-pair ratio
- local temporal regime

IMPORTANT:
R1 windows are analysis windows only.
They are NOT final retrieval segments.

R2 will use these local regimes to select/adapt a segmentation policy,
while keeping the final segment timeline continuous.

## Why no vlog/documentary/CCTV profile in videos.yaml?

Because those labels describe dataset/content domains, while the segmentation
algorithm should react to local temporal dynamics. A single YouTube video can
contain static sections, slow-cut sections, and fast montages.

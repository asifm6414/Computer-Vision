# Computer Vision: Object Detection, Tracking & Segmentation

A collection of Python computer-vision demos built with OpenCV and Ultralytics YOLO. The scripts process webcam or video input to detect objects, track people, count tracked objects, and visualize instance-segmentation results.

> **Project type:** Learning and portfolio project. Model performance depends on the input, hardware, and pretrained YOLO weights used.

## Features

- **Webcam detection** — run YOLO object detection on a live camera feed.
- **Video detection** — detect objects in a video file.
- **Human tracking and trails** — track people between frames and visualize their movement.
- **Object counting** — count unique tracked bottle detections in a video.
- **Segmentation and tracking** — show person detections, tracking IDs, and segmentation outlines.

## Technology

- Python
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`)
- NumPy (used by the segmentation demo)

## Repository structure

| Script | Description | Input |
| --- | --- | --- |
| `imagedetection.py` | Real-time object detection from a webcam | Default camera (index `0`) |
| `video_cv.py` | Object detection on video | `walking.mkv` |
| `Human_trail.py` | Person tracking with movement trails | `walking.mkv` |
| `Object_counting.py` | Tracking and unique bottle count (COCO class `39`) | `botal.mkv` |
| `segmentation.py` | Person tracking with segmentation outlines | `walking.mkv` |

## Getting started

### Requirements

- Python 3.9 or newer
- A webcam for `imagedetection.py`, or the video files listed above for video demos
- A supported environment for OpenCV display windows; GUI display may not be available in headless environments

### Install

```bash
git clone https://github.com/asifm6414/Computer-Vision.git
cd Computer-Vision
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python ultralytics numpy
```

Ultralytics downloads the requested pretrained model weights (for example, `yolov8n.pt`) when first needed. An internet connection may be required for the initial download.

## Run a demo

Run commands from the repository directory:

```bash
python imagedetection.py
python video_cv.py
python Human_trail.py
python Object_counting.py
python segmentation.py
```

Place the video files in the repository directory, or update the video paths in the corresponding scripts. Press **Q** in the OpenCV window to stop a running demo.

### Webcam output path

`imagedetection.py` currently saves every annotated webcam frame to `D:\mycv\webcam_output_<number>.jpg`. Create that folder or change `output_path` in the script to a directory on your system. Saving every frame can use substantial disk space during a long run.

## Notes and limitations

- The scripts use pretrained YOLO models and are intended as examples; they are not trained or validated for a specific deployment environment.
- `Object_counting.py` counts unique tracker IDs observed during a run. Tracking errors or a target leaving and re-entering the scene may affect the count.
- Video filenames and the webcam output directory are currently set directly in the scripts.
- Use recorded or live video only when you have appropriate permission and respect applicable privacy requirements.

## CV / portfolio summary

**Computer Vision — Object Detection and Tracking:** Built Python demos using OpenCV and Ultralytics YOLO for webcam and video object detection, person tracking with motion trails, instance segmentation, and unique object counting. Integrated model inference with real-time frame annotation and video display.

## License

No license is currently specified. Please contact the repository owner before reusing or distributing this project.

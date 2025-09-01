# Football Player Re-Identification System

This system uses computer vision and deep learning to track and re-identify football players in video footage, even when they exit and re-enter the camera's field of view.

## Features

- Player detection using YOLOv4
- Player re-identification using deep learning features
- Persistent player ID tracking across frames
- Handles players exiting and re-entering the field of view
- Visual display of player IDs and bounding boxes
- Video output option for saving results

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- scikit-learn
- Required model files (see Setup section)

## Setup

1. Install the required Python packages:

```
pip install opencv-python numpy scikit-learn
```

2. Download the required model files and place them in the `models` directory:

- [YOLOv4 weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [YOLOv4 config](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg)
- [COCO names](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names)
- [ReID model](https://drive.google.com/file/d/1_4tJMT_SG_xqfG8zzwC_OUtJG-iJYbAJ/view)

## Usage

Run the main script to launch the GUI:

```
python main.py
```

The GUI provides the following features:

1. **Video Selection**: Browse and select input video files
2. **Output Options**: Specify an output path to save the processed video
3. **Parameter Adjustment**: Adjust detection and re-identification thresholds using sliders
4. **Playback Controls**: Start, pause, resume, and stop video processing
5. **Status Information**: View current frame count and number of tracked players

During video playback:
- Player IDs will be displayed above each detected player
- Players who exit and re-enter will maintain the same ID
- The video display automatically scales to fit the window

## How It Works

1. **Player Detection**: The system uses YOLOv4 to detect people in each frame
2. **Feature Extraction**: For each detected player, a feature vector is extracted using a deep learning model
3. **Re-Identification**: When a new player is detected, their features are compared with previously seen players
4. **ID Assignment**: If the similarity exceeds a threshold, the player is assigned the same ID; otherwise, a new ID is created
5. **Tracking**: The system maintains a memory of players who have exited the frame and can re-identify them when they return

## Customization

You can adjust the following parameters in the `FootballPlayerReID` class initialization:

- `confidence_threshold`: Minimum confidence for player detection (default: 0.5)
- `nms_threshold`: Non-maximum suppression threshold (default: 0.4)
- `reid_threshold`: Similarity threshold for re-identification (default: 0.7)

The system also has a time-to-live parameter for players who haven't been seen for a while (default: 100 frames).
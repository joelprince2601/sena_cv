"""
Configuration settings for Football Player Re-ID Streamlit Application
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Model configurations
YOLO_MODELS = {
    "yolov8n": {
        "name": "YOLOv8 Nano",
        "weights": "yolov8n.pt",
        "description": "Fastest, lowest accuracy"
    },
    "yolov8s": {
        "name": "YOLOv8 Small", 
        "weights": "yolov8s.pt",
        "description": "Fast, good accuracy"
    },
    "yolov8m": {
        "name": "YOLOv8 Medium",
        "weights": "yolov8m.pt", 
        "description": "Balanced speed/accuracy"
    },
    "yolov8l": {
        "name": "YOLOv8 Large",
        "weights": "yolov8l.pt",
        "description": "Slow, high accuracy"
    },
    "yolov8x": {
        "name": "YOLOv8 Extra Large",
        "weights": "yolov8x.pt",
        "description": "Slowest, highest accuracy"
    }
}

# Tracker configurations
TRACKER_CONFIGS = {
    "custom": {
        "name": "Custom ReID",
        "description": "Enhanced tracker with deep re-identification features (Always Available)",
        "params": {
            "reid_threshold": 0.7,
            "max_age": 150,
            "feature_history_size": 5
        },
        "available": True
    },
    "ocsort": {
        "name": "OC-SORT",
        "description": "Observation-centric SORT with improved association (Requires OC_SORT)",
        "params": {
            "det_thresh": 0.5,
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3
        },
        "available": False  # Will be updated at runtime
    },
    "bytetrack": {
        "name": "ByteTrack",
        "description": "Fast and accurate multi-object tracking (Requires ByteTrack)",
        "params": {
            "track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "frame_rate": 30
        },
        "available": False  # Will be updated at runtime
    }
}

# Default parameters
DEFAULT_PARAMS = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "reid_threshold": 0.7,
    "max_age": 30,
    "min_hits": 3,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "iou_threshold": 0.3
}

# Video processing settings
VIDEO_SETTINGS = {
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv"],
    "max_file_size_mb": 500,
    "frame_skip": 1,  # Process every nth frame
    "display_fps": 30,
    "output_codec": "mp4v"
}

# UI settings
UI_SETTINGS = {
    "page_title": "Football Player Re-ID System",
    "page_icon": "⚽",
    "layout": "wide",
    "sidebar_width": 300,
    "video_display_width": 800
}

# Color scheme for player IDs
PLAYER_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green  
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (255, 192, 203), # Pink
    (0, 128, 0),    # Dark Green
    (128, 128, 0),  # Olive
    (0, 0, 128),    # Navy
    (128, 0, 0),    # Maroon
    (0, 128, 128),  # Teal
    (192, 192, 192), # Silver
    (255, 165, 0),  # Orange
    (75, 0, 130),   # Indigo
    (240, 230, 140), # Khaki
    (220, 20, 60),  # Crimson
    (32, 178, 170)  # Light Sea Green
]

def get_player_color(player_id):
    """Get color for a specific player ID"""
    return PLAYER_COLORS[player_id % len(PLAYER_COLORS)]

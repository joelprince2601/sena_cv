"""
Utility functions for Football Player Re-ID Streamlit Application
"""
import cv2
import numpy as np
import streamlit as st
import os
import requests
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import hashlib
import json
import tempfile
import time

from config import MODELS_DIR, YOLO_MODELS, VIDEO_SETTINGS

class ModelManager:
    """Manage YOLO model downloads and loading"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        
    def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available locally"""
        model_config = YOLO_MODELS.get(model_name)
        if not model_config:
            return False
            
        model_path = self.models_dir / model_config["weights"]
        return model_path.exists()
        
    def download_model(self, model_name: str, progress_callback=None) -> str:
        """Download YOLO model if not available"""
        model_config = YOLO_MODELS.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_path = self.models_dir / model_config["weights"]
        
        if model_path.exists():
            return str(model_path)
            
        # For YOLOv8 models, ultralytics will auto-download
        # Just return the model name and let ultralytics handle it
        return model_name
        
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        model_config = YOLO_MODELS.get(model_name, {})
        model_path = self.models_dir / model_config.get("weights", "")
        
        return {
            "name": model_config.get("name", model_name),
            "description": model_config.get("description", ""),
            "available": model_path.exists() if model_config else False,
            "path": str(model_path) if model_path.exists() else None,
            "size": model_path.stat().st_size if model_path.exists() else 0
        }

class FeatureExtractor:
    """Feature extraction utilities for re-identification"""
    
    @staticmethod
    def extract_color_histogram(roi: np.ndarray, bins: Tuple[int, int, int] = (30, 32, 32)) -> np.ndarray:
        """Extract color histogram features from ROI"""
        if roi.size == 0:
            return None
            
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        hist_h = cv2.calcHist([hsv_roi], [0], None, [bins[0]], [0, 180])
        hist_s = cv2.calcHist([hsv_roi], [1], None, [bins[1]], [0, 256])
        hist_v = cv2.calcHist([hsv_roi], [2], None, [bins[2]], [0, 256])
        
        # Normalize and concatenate
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        return np.concatenate([hist_h, hist_s, hist_v])
    
    @staticmethod
    def extract_texture_features(roi: np.ndarray) -> np.ndarray:
        """Extract texture features using LBP-like approach"""
        if roi.size == 0:
            return None
            
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Create histogram of gradient magnitudes
        hist, _ = np.histogram(magnitude.flatten(), bins=32, range=(0, 255))
        hist = hist.astype(np.float32)
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    @staticmethod
    def combine_features(color_features: np.ndarray, texture_features: np.ndarray) -> np.ndarray:
        """Combine color and texture features"""
        if color_features is None or texture_features is None:
            return color_features if color_features is not None else texture_features
            
        # Normalize both feature vectors
        color_norm = cv2.normalize(color_features, color_features).flatten()
        texture_norm = cv2.normalize(texture_features, texture_features).flatten()
        
        # Combine with weights (favor color features for sports)
        combined = np.concatenate([color_norm * 0.7, texture_norm * 0.3])
        
        return cv2.normalize(combined, combined).flatten()

class VideoUtils:
    """Video utility functions"""
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """Get video file information"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {}
            
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    @staticmethod
    def resize_frame_for_display(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
        """Resize frame for optimal display in Streamlit"""
        height, width = frame.shape[:2]
        
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
        return frame
    
    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """Save uploaded file and return path"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.frame_times = []
        self.detection_times = []
        self.tracking_times = []
        
    def start_frame_timer(self):
        """Start timing a frame"""
        self.frame_start = time.time()
        
    def end_frame_timer(self):
        """End timing a frame"""
        if hasattr(self, 'frame_start'):
            frame_time = time.time() - self.frame_start
            self.frame_times.append(frame_time)
            
            # Keep only last 100 measurements
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
                
    def get_fps(self) -> float:
        """Get current FPS"""
        if not self.frame_times:
            return 0
            
        avg_frame_time = np.mean(self.frame_times[-10:])  # Average of last 10 frames
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'avg_fps': self.get_fps(),
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
            'min_frame_time': np.min(self.frame_times) if self.frame_times else 0,
            'max_frame_time': np.max(self.frame_times) if self.frame_times else 0
        }

def validate_video_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded video file"""
    if uploaded_file is None:
        return False, "No file uploaded"
        
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > VIDEO_SETTINGS["max_file_size_mb"]:
        return False, f"File too large: {file_size_mb:.1f}MB (max: {VIDEO_SETTINGS['max_file_size_mb']}MB)"
        
    # Check file extension
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in VIDEO_SETTINGS["supported_formats"]:
        return False, f"Unsupported format: {file_ext}"
        
    return True, "Valid video file"

@st.cache_data
def load_demo_video_info() -> Dict[str, str]:
    """Load information about demo videos"""
    # This would contain info about any demo videos you want to include
    return {
        "demo1.mp4": "Football match highlights - 30 seconds",
        "demo2.mp4": "Training session - 45 seconds"
    }

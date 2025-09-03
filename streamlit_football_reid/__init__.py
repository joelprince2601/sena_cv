"""
Football Player Re-ID Streamlit Application
Modern web interface for football player tracking and re-identification
"""

__version__ = "1.0.0"
__author__ = "Football ReID Team"
__description__ = "Streamlit application for football player tracking and re-identification"

# Import main components
from .config import *
from .tracking_system import EnhancedFootballTracker
from .utils import ModelManager, FeatureExtractor, VideoUtils, PerformanceMonitor
from .video_processor import VideoProcessor, StreamlitVideoProcessor

__all__ = [
    "EnhancedFootballTracker",
    "ModelManager", 
    "FeatureExtractor",
    "VideoUtils",
    "PerformanceMonitor",
    "VideoProcessor",
    "StreamlitVideoProcessor"
]

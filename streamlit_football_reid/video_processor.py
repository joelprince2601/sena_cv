"""
Video processing utilities for Streamlit Football Re-ID Application
"""
import cv2
import numpy as np
import streamlit as st
import tempfile
import os
from pathlib import Path
import time
from typing import Generator, Tuple, List, Dict, Any

from tracking_system import EnhancedFootballTracker
from config import VIDEO_SETTINGS

class VideoProcessor:
    """Enhanced video processor with real-time capabilities"""
    
    def __init__(self, tracker: EnhancedFootballTracker):
        self.tracker = tracker
        self.frame_skip = VIDEO_SETTINGS["frame_skip"]
        
    def process_video_stream(self, video_path: str) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        Process video frame by frame and yield results
        
        Args:
            video_path: Path to video file
            
        Yields:
            Tuple of (annotated_frame, statistics)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Skip frames if configured
                if frame_count % self.frame_skip != 0:
                    continue
                
                # Detect players
                detections = self.tracker.detect_players(frame)
                
                # Update tracks
                tracks = self.tracker.update_tracks(frame, detections)
                
                # Draw results
                annotated_frame = self.tracker.draw_tracks(frame, tracks)
                
                # Get statistics
                stats = self.tracker.get_statistics()
                stats.update({
                    'current_frame': frame_count,
                    'total_frames': total_frames,
                    'fps': fps,
                    'progress': frame_count / total_frames,
                    'timestamp': frame_count / fps,
                    'active_tracks': len(tracks)
                })
                
                yield annotated_frame, stats
                
        finally:
            cap.release()
    
    def save_processed_video(self, video_path: str, output_path: str, 
                           progress_callback=None) -> str:
        """
        Save processed video with tracking annotations
        
        Args:
            video_path: Input video path
            output_path: Output video path
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to saved video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_SETTINGS["output_codec"])
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            for annotated_frame, stats in self.process_video_stream(video_path):
                frame_count += 1
                
                # Write frame
                out.write(annotated_frame)
                
                # Update progress
                if progress_callback:
                    progress_callback(stats['progress'])
                    
        finally:
            cap.release()
            out.release()
            
        return output_path

class StreamlitVideoProcessor:
    """Streamlit-specific video processor with UI integration"""
    
    def __init__(self):
        self.processor = None
        
    def setup_processor(self, config: Dict[str, Any]) -> None:
        """Setup video processor with given configuration"""
        tracker = EnhancedFootballTracker(
            yolo_model=config['model'],
            tracker_type=config['tracker'],
            confidence_threshold=config['confidence_threshold'],
            nms_threshold=config['nms_threshold'],
            reid_threshold=config['reid_threshold'],
            max_age=config.get('max_age', 30),
            min_hits=config.get('min_hits', 3)
        )
        
        self.processor = VideoProcessor(tracker)
        
    def process_uploaded_video(self, uploaded_file, config: Dict[str, Any]) -> List[Dict]:
        """
        Process uploaded video file in Streamlit
        
        Args:
            uploaded_file: Streamlit uploaded file object
            config: Processing configuration
            
        Returns:
            List of frame statistics
        """
        if self.processor is None:
            self.setup_processor(config)
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
            
        try:
            # Create UI elements
            col1, col2 = st.columns([3, 1])
            
            with col1:
                video_placeholder = st.empty()
                
            with col2:
                stats_container = st.container()
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_data = []
            
            # Process video
            for frame, stats in self.processor.process_video_stream(temp_video_path):
                # Update video display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # Update statistics
                with stats_container:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Active Players", stats['active_tracks'])
                        st.metric("Frame", f"{stats['current_frame']}/{stats['total_frames']}")
                    with col_b:
                        st.metric("Time", f"{stats['timestamp']:.1f}s")

                        # Show ID pool info for custom tracker
                        if 'id_pool_usage' in stats:
                            st.metric("ID Pool", stats['id_pool_usage'])
                        if 'gallery_players' in stats:
                            st.metric("Gallery", stats['gallery_players'])
                    
                # Update progress
                progress_bar.progress(stats['progress'])
                status_text.text(f"Processing: {stats['progress']*100:.1f}% complete")
                
                # Store data
                results_data.append(stats)
                
                # Control processing speed for display
                time.sleep(0.03)  # ~30 FPS display
                
            return results_data
            
        finally:
            # Clean up
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)

def create_download_link(video_path: str, filename: str) -> str:
    """Create download link for processed video"""
    with open(video_path, "rb") as file:
        video_bytes = file.read()
        
    st.download_button(
        label="📥 Download Processed Video",
        data=video_bytes,
        file_name=filename,
        mime="video/mp4"
    )

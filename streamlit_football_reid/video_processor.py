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
        Process uploaded video file in Streamlit with enhanced live analytics

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

        # Get video info for proper timing
        cap = cv2.VideoCapture(temp_video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Calculate frame timing for smooth playback
        max_display_fps = config.get('max_display_fps', 15)
        playback_speed = config.get('playback_speed', 1.0)
        target_fps = min(video_fps * playback_speed, max_display_fps)  # Respect user settings
        frame_delay = 1.0 / target_fps if target_fps > 0 else 0.067  # Default ~15 FPS

        try:
            # Create enhanced UI layout
            col1, col2 = st.columns([2.5, 1.5])

            with col1:
                st.markdown("### 🎬 Live Video Analysis")
                video_placeholder = st.empty()

                # Video controls (use config values as defaults)
                control_col1, control_col2, control_col3 = st.columns(3)
                with control_col1:
                    speed_options = [0.25, 0.5, 1.0, 1.5, 2.0]
                    default_speed_idx = speed_options.index(config.get('playback_speed', 1.0))
                    speed_multiplier = st.selectbox("Playback Speed", speed_options, index=default_speed_idx)
                with control_col2:
                    frame_skip = st.selectbox("Frame Skip", [1, 2, 3, 5], index=0)
                with control_col3:
                    show_debug = st.checkbox("Debug Info", value=config.get('enable_debug_overlay', False))

            with col2:
                st.markdown("### 📊 Live Statistics")
                stats_container = st.container()

                # Performance metrics
                st.markdown("### ⚡ Performance")
                perf_container = st.container()

            # Progress and status
            progress_col1, progress_col2 = st.columns([3, 1])
            with progress_col1:
                progress_bar = st.progress(0)
            with progress_col2:
                status_text = st.empty()

            # Enhanced analytics containers
            st.markdown("### 📈 Real-time Analytics")
            analytics_col1, analytics_col2 = st.columns(2)

            with analytics_col1:
                chart_placeholder = st.empty()
            with analytics_col2:
                heatmap_placeholder = st.empty()

            results_data = []
            frame_buffer = []
            processing_times = []
            player_positions = []

            # Adjust frame delay based on speed multiplier
            adjusted_delay = frame_delay / speed_multiplier

            # Process video with enhanced analytics
            frame_count = 0
            for frame, stats in self.processor.process_video_stream(temp_video_path):
                frame_count += 1

                # Apply frame skipping
                if frame_count % frame_skip != 0:
                    continue

                start_display_time = time.time()

                # Update video display with enhanced frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Add debug overlay if enabled
                if show_debug:
                    rgb_frame = self._add_debug_overlay(rgb_frame, stats, processing_times)

                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update enhanced statistics
                with stats_container:
                    self._update_live_stats(stats)

                # Update performance metrics
                with perf_container:
                    self._update_performance_metrics(stats, processing_times)

                # Update real-time analytics
                self._update_realtime_analytics(
                    stats, chart_placeholder, heatmap_placeholder,
                    player_positions, results_data
                )

                # Update progress with enhanced info
                progress_bar.progress(stats['progress'])
                status_text.text(
                    f"Frame {stats['current_frame']}/{stats['total_frames']} "
                    f"({stats['progress']*100:.1f}%) - "
                    f"{stats['active_tracks']} players"
                )

                # Store enhanced data
                enhanced_stats = stats.copy()
                enhanced_stats.update({
                    'processing_time': time.time() - start_display_time,
                    'display_fps': 1.0 / adjusted_delay,
                    'frame_skip': frame_skip,
                    'speed_multiplier': speed_multiplier
                })
                results_data.append(enhanced_stats)

                # Track processing performance
                processing_times.append(enhanced_stats['processing_time'])
                if len(processing_times) > 30:  # Keep last 30 measurements
                    processing_times.pop(0)

                # Collect player position data for heatmap
                if 'active_tracks' in stats and stats['active_tracks'] > 0:
                    player_positions.append({
                        'frame': stats['current_frame'],
                        'players': stats['active_tracks'],
                        'timestamp': stats['timestamp']
                    })

                # Smart timing control
                elapsed_time = time.time() - start_display_time
                sleep_time = max(0, adjusted_delay - elapsed_time)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            return results_data

        finally:
            # Clean up
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)

    def _add_debug_overlay(self, frame, stats, processing_times):
        """Add debug information overlay to frame"""
        import cv2

        # Convert to BGR for OpenCV operations
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add debug text
        debug_info = [
            f"Frame: {stats.get('current_frame', 0)}/{stats.get('total_frames', 0)}",
            f"Players: {stats.get('active_tracks', 0)}",
            f"FPS: {1.0/np.mean(processing_times[-5:]):.1f}" if processing_times else "FPS: --",
            f"Time: {stats.get('timestamp', 0):.1f}s"
        ]

        # Draw debug overlay
        y_offset = 30
        for i, text in enumerate(debug_info):
            cv2.putText(bgr_frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert back to RGB
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    def _update_live_stats(self, stats):
        """Update live statistics display"""
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Active Players", stats.get('active_tracks', 0))
            st.metric("Frame", f"{stats.get('current_frame', 0)}/{stats.get('total_frames', 0)}")

            # Show ID pool info for custom tracker
            if 'id_pool_usage' in stats:
                st.metric("ID Pool", stats['id_pool_usage'])

        with col_b:
            st.metric("Video Time", f"{stats.get('timestamp', 0):.1f}s")
            st.metric("Progress", f"{stats.get('progress', 0)*100:.1f}%")

            if 'gallery_players' in stats:
                st.metric("Gallery", stats['gallery_players'])

    def _update_performance_metrics(self, stats, processing_times):
        """Update performance metrics display"""
        if processing_times:
            avg_time = np.mean(processing_times[-10:])  # Last 10 frames
            current_fps = 1.0 / avg_time if avg_time > 0 else 0

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Processing FPS", f"{current_fps:.1f}")
            with col_p2:
                st.metric("Avg Frame Time", f"{avg_time*1000:.1f}ms")

    def _update_realtime_analytics(self, stats, chart_placeholder, heatmap_placeholder,
                                 player_positions, results_data):
        """Update real-time analytics charts"""
        if len(results_data) > 10:  # Only update after some data is collected
            try:
                import pandas as pd
                import plotly.express as px
                import plotly.graph_objects as go

                # Create DataFrame from recent data
                recent_data = results_data[-50:]  # Last 50 frames
                df = pd.DataFrame(recent_data)

                # Player count over time chart
                if 'timestamp' in df.columns and 'active_tracks' in df.columns:
                    fig = px.line(df, x='timestamp', y='active_tracks',
                                title='Players Over Time',
                                labels={'timestamp': 'Time (s)', 'active_tracks': 'Active Players'})
                    fig.update_layout(height=300, showlegend=False)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Performance heatmap (simplified)
                if len(player_positions) > 5:
                    pos_df = pd.DataFrame(player_positions[-20:])  # Last 20 data points

                    # Create a simple performance visualization
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=pos_df['timestamp'],
                        y=pos_df['players'],
                        mode='markers+lines',
                        name='Player Activity',
                        marker=dict(
                            size=pos_df['players'] * 3,
                            color=pos_df['players'],
                            colorscale='Viridis',
                            showscale=True
                        )
                    ))

                    fig2.update_layout(
                        title='Player Activity Heatmap',
                        xaxis_title='Time (s)',
                        yaxis_title='Number of Players',
                        height=300,
                        showlegend=False
                    )
                    heatmap_placeholder.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                # Silently handle chart update errors
                pass

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

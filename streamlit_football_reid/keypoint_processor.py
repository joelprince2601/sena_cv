"""
Keypoint Processing for Streamlit Application
Handles video processing with keypoint detection and tracking
"""
import cv2
import numpy as np
import streamlit as st
import tempfile
import os
from typing import Generator, Tuple, List, Dict, Any
import time

from keypoint_tracker import FootballPoseAnalyzer
from keypoint_visualizer import FootballPoseVisualizer

class StreamlitKeypointProcessor:
    """Streamlit-specific football pose processor"""

    def __init__(self):
        self.analyzer = None
        self.visualizer = FootballPoseVisualizer()
        
    def setup_processor(self, detection_type: str, confidence_threshold: float):
        """Setup football pose analyzer with given configuration"""
        self.analyzer = FootballPoseAnalyzer(
            detection_type=detection_type,
            confidence_threshold=confidence_threshold
        )
        # Reset visualizer colors for new session
        self.visualizer.reset_colors()
        
    def process_video_stream(self, video_path: str) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """Process video frame by frame with keypoint detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        processing_times = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                start_time = time.time()
                
                # Detect football player poses
                detections = self.analyzer.detect_football_players(frame)

                # Track football players
                tracked_results = self.analyzer.track_football_players(frame, detections)

                # Visualize football poses
                annotated_frame = self.visualizer.draw_football_poses(
                    frame, tracked_results,
                    show_connections=True,
                    show_labels=True,
                    show_actions=True
                )
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Calculate statistics
                stats = self._calculate_stats(
                    tracked_results, frame_count, total_frames, fps, processing_times
                )
                
                yield annotated_frame, stats
                
        finally:
            cap.release()
            
    def _calculate_stats(self, tracked_results, frame_count, total_frames, fps, processing_times):
        """Calculate football-specific processing statistics"""
        # Count football players and actions
        football_players = len(tracked_results)

        # Count actions
        action_counts = {'standing': 0, 'running': 0, 'kicking': 0, 'jumping': 0, 'crouching': 0}
        for result in tracked_results:
            action = result.get('action_analysis', {}).get('action', 'standing')
            if action in action_counts:
                action_counts[action] += 1

        # Calculate total keypoints
        total_keypoints = sum(len(r['landmarks']) for r in tracked_results)

        # Calculate average confidence and pose quality
        confidences = [r['confidence'] for r in tracked_results]
        pose_qualities = [r.get('pose_quality', {}).get('overall_quality', 0) for r in tracked_results]

        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_pose_quality = np.mean(pose_qualities) if pose_qualities else 0.0

        # Calculate processing FPS
        recent_times = processing_times[-10:]  # Last 10 frames
        avg_processing_time = np.mean(recent_times) if recent_times else 0.0
        processing_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0

        # Get analyzer statistics
        analyzer_stats = self.analyzer.get_football_statistics() if self.analyzer else {}

        return {
            'frame_count': frame_count,
            'total_frames': total_frames,
            'progress': frame_count / total_frames,
            'timestamp': frame_count / fps,
            'fps': fps,
            'processing_fps': processing_fps,
            'football_players': football_players,
            'action_counts': action_counts,
            'total_entities': football_players,
            'total_keypoints': total_keypoints,
            'avg_confidence': avg_confidence,
            'avg_pose_quality': avg_pose_quality,
            'active_tracks': analyzer_stats.get('active_players', 0),
            'most_common_action': max(action_counts, key=action_counts.get) if any(action_counts.values()) else 'standing'
        }
        
    def process_uploaded_video(self, uploaded_file, detection_type: str, 
                             confidence_threshold: float, 
                             show_connections: bool = True,
                             show_labels: bool = True,
                             show_3d: bool = False) -> List[Dict]:
        """Process uploaded video file in Streamlit"""
        
        # Setup processor
        self.setup_processor(detection_type, confidence_threshold)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
            
        try:
            # Create UI layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("## 🎬 Keypoint Detection & Tracking")
                video_placeholder = st.empty()
                
                # 3D visualization placeholder
                if show_3d:
                    st.markdown("### 📊 3D Keypoint Visualization")
                    viz_3d_placeholder = st.empty()
                
            with col2:
                st.markdown("## 📊 Statistics")
                stats_container = st.container()
                
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_data = []
            
            # Process video
            for frame, stats in self.process_video_stream(temp_video_path):
                # Update video display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # Update 3D visualization periodically
                if show_3d and stats['frame_count'] % 10 == 0:  # Every 10 frames
                    tracked_results = []  # Would need to pass this from processing
                    if tracked_results:
                        fig_3d = self.visualizer.create_3d_visualization(tracked_results)
                        viz_3d_placeholder.plotly_chart(fig_3d, use_container_width=True)
                
                # Update statistics
                with stats_container:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Football Players", stats['football_players'])
                        st.metric("Most Common Action", stats['most_common_action'].title())
                        st.metric("Keypoints", stats['total_keypoints'])
                    with col_b:
                        st.metric("Pose Quality", f"{stats['avg_pose_quality']:.2f}")
                        st.metric("Confidence", f"{stats['avg_confidence']:.2f}")
                        st.metric("Active Tracks", stats['active_tracks'])

                    # Action breakdown
                    st.markdown("**Actions:**")
                    action_counts = stats['action_counts']
                    for action, count in action_counts.items():
                        if count > 0:
                            st.text(f"{action.title()}: {count}")

                    # Processing info
                    st.metric("Processing FPS", f"{stats['processing_fps']:.1f}")
                    st.metric("Frame", f"{stats['frame_count']}/{stats['total_frames']}")
                    st.metric("Time", f"{stats['timestamp']:.1f}s")
                
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
                
    def get_detection_types(self):
        """Get available football analysis types"""
        return {
            "football_pose": {
                "name": "Football Pose Analysis",
                "description": "Full body pose analysis for football players (33 keypoints + actions)",
                "keypoints": 33,
                "actions": ["standing", "running", "kicking", "jumping", "crouching"]
            },
            "advanced_football": {
                "name": "Advanced Football Analysis",
                "description": "Detailed pose analysis with action recognition and biomechanics",
                "keypoints": 33,
                "actions": ["standing", "running", "kicking", "jumping", "crouching", "diving"]
            }
        }
        
    def create_keypoint_analytics(self, results_data):
        """Create analytics dashboard for keypoint data"""
        if not results_data:
            return None
            
        import pandas as pd
        import plotly.express as px
        
        df = pd.DataFrame(results_data)
        
        # Create visualizations
        figs = {}
        
        # Entity count over time
        figs['entities_over_time'] = px.line(
            df, x='timestamp', y='total_entities',
            title='Detected Entities Over Time',
            labels={'timestamp': 'Time (seconds)', 'total_entities': 'Number of Entities'}
        )
        
        # Keypoint types distribution
        keypoint_data = []
        for _, row in df.iterrows():
            keypoint_data.extend([
                {'time': row['timestamp'], 'type': 'Pose', 'count': row['pose_count']},
                {'time': row['timestamp'], 'type': 'Hands', 'count': row['hand_count']},
                {'time': row['timestamp'], 'type': 'Faces', 'count': row['face_count']}
            ])
        
        keypoint_df = pd.DataFrame(keypoint_data)
        figs['keypoint_distribution'] = px.line(
            keypoint_df, x='time', y='count', color='type',
            title='Keypoint Types Over Time',
            labels={'time': 'Time (seconds)', 'count': 'Count'}
        )
        
        # Confidence over time
        figs['confidence_over_time'] = px.line(
            df, x='timestamp', y='avg_confidence',
            title='Average Confidence Over Time',
            labels={'timestamp': 'Time (seconds)', 'avg_confidence': 'Average Confidence'}
        )
        
        # Processing performance
        figs['processing_fps'] = px.line(
            df, x='timestamp', y='processing_fps',
            title='Processing Performance (FPS)',
            labels={'timestamp': 'Time (seconds)', 'processing_fps': 'Processing FPS'}
        )
        
        return figs

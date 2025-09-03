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
        action_counts = {'standing': 0, 'running': 0, 'kicking': 0, 'jumping': 0, 'moving': 0}
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
            
        # Get video info for proper timing
        cap = cv2.VideoCapture(temp_video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Calculate frame timing for smooth playback
        target_fps = min(video_fps, 12)  # Cap at 12 FPS for pose analysis
        frame_delay = 1.0 / target_fps if target_fps > 0 else 0.083  # Default ~12 FPS

        try:
            # Create enhanced UI layout
            col1, col2 = st.columns([2.5, 1.5])

            with col1:
                st.markdown("### 🤸 Live Football Pose Analysis")
                video_placeholder = st.empty()

                # Enhanced controls
                control_col1, control_col2, control_col3 = st.columns(3)
                with control_col1:
                    speed_multiplier = st.selectbox("Analysis Speed", [0.5, 1.0, 1.5, 2.0], index=1, key="pose_speed")
                with control_col2:
                    frame_skip = st.selectbox("Frame Skip", [0.5, 1.0, 1.5, 2.0], index=0, key="pose_skip")
                with control_col3:
                    show_pose_debug = st.checkbox("Pose Debug", value=False)

                # 3D visualization placeholder
                if show_3d:
                    st.markdown("### 📊 3D Pose Visualization")
                    viz_3d_placeholder = st.empty()

            with col2:
                st.markdown("### 📊 Live Football Stats")
                stats_container = st.container()

                # Action analysis
                st.markdown("### ⚽ Action Analysis")
                action_container = st.container()

            # Progress and status
            progress_col1, progress_col2 = st.columns([3, 1])
            with progress_col1:
                progress_bar = st.progress(0)
            with progress_col2:
                status_text = st.empty()

            # Enhanced pose analytics
            st.markdown("### 📈 Real-time Pose Analytics")
            pose_analytics_col1, pose_analytics_col2 = st.columns(2)

            with pose_analytics_col1:
                action_chart_placeholder = st.empty()
            with pose_analytics_col2:
                quality_chart_placeholder = st.empty()

            results_data = []
            action_history = []
            quality_history = []

            # Adjust frame delay based on speed multiplier
            adjusted_delay = frame_delay / speed_multiplier

            # Process video with enhanced pose analytics
            frame_count = 0
            for frame, stats in self.process_video_stream(temp_video_path):
                frame_count += 1

                # Apply frame skipping
                if frame_count % frame_skip != 0:
                    continue

                start_display_time = time.time()

                # Update video display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Add pose debug overlay if enabled
                if show_pose_debug:
                    rgb_frame = self._add_pose_debug_overlay(rgb_frame, stats)

                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update 3D visualization periodically
                if show_3d and stats['frame_count'] % 15 == 0:  # Every 15 frames for performance
                    try:
                        # Create simplified 3D visualization
                        self._update_3d_pose_viz(viz_3d_placeholder, stats)
                    except:
                        pass  # Skip 3D updates if they fail

                # Update enhanced statistics
                with stats_container:
                    self._update_pose_stats(stats)

                # Update action analysis
                with action_container:
                    self._update_action_analysis(stats, action_history)

                # Update real-time pose analytics
                self._update_pose_analytics(
                    stats, action_chart_placeholder, quality_chart_placeholder,
                    action_history, quality_history, results_data
                )

                # Update progress with enhanced info
                progress_bar.progress(stats['progress'])
                status_text.text(
                    f"Frame {stats['frame_count']}/{stats['total_frames']} "
                    f"({stats['progress']*100:.1f}%) - "
                    f"{stats['football_players']} players - "
                    f"Action: {stats['most_common_action'].title()}"
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

                # Track action and quality history
                action_history.append({
                    'timestamp': stats['timestamp'],
                    'action': stats['most_common_action'],
                    'players': stats['football_players'],
                    'quality': stats['avg_pose_quality']
                })

                quality_history.append({
                    'timestamp': stats['timestamp'],
                    'quality': stats['avg_pose_quality'],
                    'confidence': stats['avg_confidence']
                })

                # Keep history manageable
                if len(action_history) > 100:
                    action_history.pop(0)
                if len(quality_history) > 100:
                    quality_history.pop(0)

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

    def _add_pose_debug_overlay(self, frame, stats):
        """Add pose debug information overlay to frame"""
        import cv2

        # Convert to BGR for OpenCV operations
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add pose debug text
        debug_info = [
            f"Players: {stats.get('football_players', 0)}",
            f"Action: {stats.get('most_common_action', 'standing').title()}",
            f"Quality: {stats.get('avg_pose_quality', 0):.2f}",
            f"Confidence: {stats.get('avg_confidence', 0):.2f}",
            f"Keypoints: {stats.get('total_keypoints', 0)}"
        ]

        # Draw debug overlay with background
        overlay = bgr_frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, bgr_frame, 0.3, 0, bgr_frame)

        y_offset = 25
        for i, text in enumerate(debug_info):
            cv2.putText(bgr_frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Convert back to RGB
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    def _update_pose_stats(self, stats):
        """Update pose statistics display"""
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Football Players", stats.get('football_players', 0))
            st.metric("Total Keypoints", stats.get('total_keypoints', 0))
            st.metric("Active Tracks", stats.get('active_tracks', 0))

        with col_b:
            st.metric("Pose Quality", f"{stats.get('avg_pose_quality', 0):.2f}")
            st.metric("Confidence", f"{stats.get('avg_confidence', 0):.2f}")
            st.metric("Processing FPS", f"{stats.get('processing_fps', 0):.1f}")

    def _update_action_analysis(self, stats, action_history):
        """Update action analysis display"""
        action_counts = stats.get('action_counts', {})
        most_common = stats.get('most_common_action', 'standing')

        st.markdown(f"**Current Dominant Action: {most_common.title()}**")

        # Show action breakdown
        for action, count in action_counts.items():
            if count > 0:
                percentage = (count / max(sum(action_counts.values()), 1)) * 100
                st.progress(percentage / 100, text=f"{action.title()}: {count} ({percentage:.1f}%)")

        # Show action trend if we have history
        if len(action_history) > 5:
            recent_actions = [item['action'] for item in action_history[-10:]]
            action_trend = max(set(recent_actions), key=recent_actions.count)
            st.info(f"🔄 Recent Trend: {action_trend.title()}")

    def _update_3d_pose_viz(self, placeholder, stats):
        """Update 3D pose visualization (simplified)"""
        try:
            import plotly.graph_objects as go

            # Create a simple 3D scatter plot representing pose quality
            fig = go.Figure()

            # Add dummy 3D points for demonstration
            fig.add_trace(go.Scatter3d(
                x=[0, 1, 2],
                y=[0, 1, 2],
                z=[0, 1, 2],
                mode='markers',
                marker=dict(
                    size=10,
                    color=[stats.get('avg_pose_quality', 0.5)] * 3,
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Pose Quality'
            ))

            fig.update_layout(
                title='3D Pose Quality Visualization',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                height=400
            )

            placeholder.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            placeholder.text("3D visualization temporarily unavailable")

    def _update_pose_analytics(self, stats, action_chart_placeholder, quality_chart_placeholder,
                             action_history, quality_history, results_data):
        """Update real-time pose analytics charts"""
        if len(results_data) > 10:  # Only update after some data is collected
            try:
                import pandas as pd
                import plotly.express as px
                import plotly.graph_objects as go

                # Action distribution over time
                if len(action_history) > 5:
                    action_df = pd.DataFrame(action_history[-30:])  # Last 30 data points

                    # Create action timeline
                    fig1 = px.scatter(action_df, x='timestamp', y='action',
                                    size='players', color='quality',
                                    title='Football Actions Timeline',
                                    labels={'timestamp': 'Time (s)', 'action': 'Action'},
                                    color_continuous_scale='Viridis')
                    fig1.update_layout(height=300, showlegend=False)
                    action_chart_placeholder.plotly_chart(fig1, use_container_width=True)

                # Quality and confidence over time
                if len(quality_history) > 5:
                    quality_df = pd.DataFrame(quality_history[-30:])  # Last 30 data points

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=quality_df['timestamp'],
                        y=quality_df['quality'],
                        mode='lines+markers',
                        name='Pose Quality',
                        line=dict(color='green')
                    ))
                    fig2.add_trace(go.Scatter(
                        x=quality_df['timestamp'],
                        y=quality_df['confidence'],
                        mode='lines+markers',
                        name='Confidence',
                        line=dict(color='blue')
                    ))

                    fig2.update_layout(
                        title='Pose Quality & Confidence',
                        xaxis_title='Time (s)',
                        yaxis_title='Score',
                        height=300,
                        yaxis=dict(range=[0, 1])
                    )
                    quality_chart_placeholder.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                # Silently handle chart update errors
                pass

    def get_detection_types(self):
        """Get available football analysis types"""
        return {
            "football_pose": {
                "name": "Football Pose Analysis",
                "description": "Full body pose analysis for football players (33 keypoints + actions)",
                "keypoints": 33,
                "actions": ["standing", "running", "kicking", "jumping", "moving"]
            },
            "advanced_football": {
                "name": "Advanced Football Analysis",
                "description": "Detailed pose analysis with action recognition and biomechanics",
                "keypoints": 33,
                "actions": ["standing", "running", "kicking", "jumping", "moving", "diving"]
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

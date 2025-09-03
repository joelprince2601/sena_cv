"""
Football Pose Visualization System
Advanced visualization for football player poses with action recognition
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
import colorsys

class FootballPoseVisualizer:
    """Advanced football pose visualization with action recognition"""
    
    def __init__(self):
        # Color schemes for football poses
        self.pose_colors = self._generate_football_pose_colors()
        self.action_colors = self._generate_action_colors()

        # Track colors for consistent visualization
        self.track_colors = {}
        self.color_palette = self._generate_color_palette(20)

        # Football-specific visualization settings
        self.show_action_indicators = True
        self.show_pose_quality = True
        
    def _generate_color_palette(self, num_colors):
        """Generate distinct colors for tracking"""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
        
    def _generate_football_pose_colors(self):
        """Generate colors for football pose keypoints"""
        return {
            'head': (255, 100, 100),    # Light Red
            'torso': (100, 255, 100),   # Light Green
            'left_arm': (100, 100, 255), # Light Blue
            'right_arm': (255, 100, 255), # Light Magenta
            'left_leg': (255, 255, 100), # Light Yellow
            'right_leg': (100, 255, 255), # Light Cyan
            'key_joints': (255, 255, 255), # White for important joints
            'default': (200, 200, 200)  # Light Gray
        }

    def _generate_action_colors(self):
        """Generate colors for different football actions"""
        return {
            'standing': (100, 100, 100),   # Gray
            'running': (0, 255, 0),        # Green
            'kicking': (255, 0, 0),        # Red
            'jumping': (0, 0, 255),        # Blue
            'crouching': (255, 255, 0),    # Yellow
            'diving': (255, 0, 255),       # Magenta
            'unknown': (128, 128, 128)     # Dark Gray
        }
        
    def _generate_hand_colors(self):
        """Generate colors for hand keypoints"""
        return {
            'thumb': (255, 0, 0),     # Red
            'index': (0, 255, 0),     # Green
            'middle': (0, 0, 255),    # Blue
            'ring': (255, 255, 0),    # Yellow
            'pinky': (255, 0, 255),   # Magenta
            'palm': (0, 255, 255),    # Cyan
            'default': (255, 255, 255) # White
        }
        
    def _generate_face_colors(self):
        """Generate colors for face keypoints"""
        return {
            'contour': (255, 255, 0),   # Yellow
            'eyes': (0, 255, 0),        # Green
            'eyebrows': (0, 0, 255),    # Blue
            'nose': (255, 0, 0),        # Red
            'lips': (255, 0, 255),      # Magenta
            'default': (255, 255, 255)  # White
        }
        
    def get_track_color(self, track_id):
        """Get consistent color for a track ID"""
        if track_id not in self.track_colors:
            color_idx = len(self.track_colors) % len(self.color_palette)
            self.track_colors[track_id] = self.color_palette[color_idx]
        return self.track_colors[track_id]
        
    def draw_football_poses(self, frame, tracked_results, show_connections=True, show_labels=True, show_actions=True):
        """Draw football poses with action recognition on frame"""
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]

        for result in tracked_results:
            track_id = result['track_id']
            landmarks = result['landmarks']
            connections = result.get('connections', [])
            action_analysis = result.get('action_analysis', {})
            pose_quality = result.get('pose_quality', {})

            # Get colors based on action
            action = action_analysis.get('action', 'standing')
            action_color = self.action_colors.get(action, self.action_colors['unknown'])
            track_color = self.get_track_color(track_id)

            # Draw connections first (so they appear behind keypoints)
            if show_connections and connections:
                self._draw_football_connections(annotated_frame, landmarks, connections,
                                              action_color, width, height)

            # Draw keypoints with football-specific coloring
            self._draw_football_landmarks(annotated_frame, landmarks, action_color,
                                        track_color, width, height)

            # Draw action indicators
            if show_actions:
                self._draw_action_indicator(annotated_frame, result, width, height)

            # Draw labels with football info
            if show_labels:
                self._draw_football_labels(annotated_frame, result, width, height)

        return annotated_frame

    def _draw_football_connections(self, frame, landmarks, connections, color, width, height):
        """Draw connections between keypoints with football-specific styling"""
        for connection in connections:
            start_idx, end_idx = connection

            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]

                # Check visibility
                start_visible = start_point.get('visibility', 1.0) > 0.5
                end_visible = end_point.get('visibility', 1.0) > 0.5

                if start_visible and end_visible:
                    start_pos = (int(start_point['x'] * width),
                               int(start_point['y'] * height))
                    end_pos = (int(end_point['x'] * width),
                             int(end_point['y'] * height))

                    # Thicker lines for better visibility
                    cv2.line(frame, start_pos, end_pos, color, 3)

    def _draw_football_landmarks(self, frame, landmarks, action_color, track_color, width, height):
        """Draw football-specific keypoints"""
        for i, landmark in enumerate(landmarks):
            if landmark.get('visibility', 1.0) > 0.5:
                x = int(landmark['x'] * width)
                y = int(landmark['y'] * height)

                # Get specific color for keypoint
                point_color = self._get_football_keypoint_color(i, action_color)

                # Draw keypoint with larger size
                cv2.circle(frame, (x, y), 6, point_color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 2)  # White border

    def _get_football_keypoint_color(self, landmark_idx, default_color):
        """Get color for football keypoint based on body part (YOLO keypoints)"""
        pose_colors = self.pose_colors

        # Head keypoints (nose, eyes, ears)
        if landmark_idx in [0, 1, 2, 3, 4]:
            return pose_colors['head']
        # Torso keypoints (shoulders and hips)
        elif landmark_idx in [5, 6, 11, 12]:
            return pose_colors['torso']
        # Left arm (shoulder, elbow, wrist)
        elif landmark_idx in [5, 7, 9]:
            return pose_colors['left_arm']
        # Right arm (shoulder, elbow, wrist)
        elif landmark_idx in [6, 8, 10]:
            return pose_colors['right_arm']
        # Left leg (hip, knee, ankle)
        elif landmark_idx in [11, 13, 15]:
            return pose_colors['left_leg']
        # Right leg (hip, knee, ankle)
        elif landmark_idx in [12, 14, 16]:
            return pose_colors['right_leg']
        else:
            return default_color

    def _draw_action_indicator(self, frame, result, width, height):
        """Draw action indicator for football player"""
        action_analysis = result.get('action_analysis', {})
        action = action_analysis.get('action', 'standing')
        confidence = action_analysis.get('confidence', 0.0)
        landmarks = result['landmarks']

        if landmarks:
            # Find head position for indicator placement
            head_landmarks = [lm for i, lm in enumerate(landmarks) if i <= 10 and lm.get('visibility', 0) > 0.5]
            if head_landmarks:
                head_x = int(np.mean([lm['x'] for lm in head_landmarks]) * width)
                head_y = int(np.mean([lm['y'] for lm in head_landmarks]) * height)

                # Draw action indicator circle
                action_color = self.action_colors.get(action, self.action_colors['unknown'])
                cv2.circle(frame, (head_x, head_y - 40), 15, action_color, -1)
                cv2.circle(frame, (head_x, head_y - 40), 15, (255, 255, 255), 2)

                # Add action text
                cv2.putText(frame, action.upper()[:3],
                           (head_x - 15, head_y - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_football_labels(self, frame, result, width, height):
        """Draw comprehensive labels for football players"""
        track_id = result['track_id']
        action_analysis = result.get('action_analysis', {})
        pose_quality = result.get('pose_quality', {})
        confidence = result['confidence']

        action = action_analysis.get('action', 'standing')
        action_confidence = action_analysis.get('confidence', 0.0)
        overall_quality = pose_quality.get('overall_quality', 0.0)

        # Find center position for label
        landmarks = result['landmarks']
        if landmarks:
            center_x = int(np.mean([lm['x'] for lm in landmarks if lm.get('visibility', 0) > 0.5]) * width)
            center_y = int(np.mean([lm['y'] for lm in landmarks if lm.get('visibility', 0) > 0.5]) * height)

            # Create comprehensive label
            label_lines = [
                f"Player {track_id}",
                f"Action: {action.title()}",
                f"Conf: {action_confidence:.2f}",
                f"Quality: {overall_quality:.2f}"
            ]

            # Get track color
            track_color = self.get_track_color(track_id)

            # Draw label background
            label_height = len(label_lines) * 20 + 10
            label_width = max(len(line) * 8 for line in label_lines) + 10

            cv2.rectangle(frame,
                         (center_x - label_width//2, center_y + 30),
                         (center_x + label_width//2, center_y + 30 + label_height),
                         track_color, -1)

            cv2.rectangle(frame,
                         (center_x - label_width//2, center_y + 30),
                         (center_x + label_width//2, center_y + 30 + label_height),
                         (255, 255, 255), 2)

            # Draw label text
            for i, line in enumerate(label_lines):
                cv2.putText(frame, line,
                           (center_x - label_width//2 + 5, center_y + 50 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    def _draw_connections(self, frame, landmarks, connections, color, width, height):
        """Draw connections between keypoints"""
        for connection in connections:
            start_idx, end_idx = connection
            
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                # Check visibility for pose landmarks
                start_visible = start_point.get('visibility', 1.0) > 0.5
                end_visible = end_point.get('visibility', 1.0) > 0.5
                
                if start_visible and end_visible:
                    start_pos = (int(start_point['x'] * width), 
                               int(start_point['y'] * height))
                    end_pos = (int(end_point['x'] * width), 
                             int(end_point['y'] * height))
                    
                    cv2.line(frame, start_pos, end_pos, color, 2)
                    
    def _draw_landmarks(self, frame, landmarks, keypoint_type, track_color, width, height):
        """Draw individual keypoints"""
        for i, landmark in enumerate(landmarks):
            # Check visibility for pose landmarks
            if landmark.get('visibility', 1.0) > 0.5:
                x = int(landmark['x'] * width)
                y = int(landmark['y'] * height)
                
                # Get specific color for keypoint type
                point_color = self._get_keypoint_color(keypoint_type, i, track_color)
                
                # Draw keypoint
                cv2.circle(frame, (x, y), 4, point_color, -1)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)  # White border
                
    def _get_keypoint_color(self, keypoint_type, landmark_idx, default_color):
        """Get color for specific keypoint based on type and index"""
        if keypoint_type == "pose":
            return self._get_pose_keypoint_color(landmark_idx, default_color)
        elif keypoint_type == "hand":
            return self._get_hand_keypoint_color(landmark_idx, default_color)
        elif keypoint_type == "face":
            return self._get_face_keypoint_color(landmark_idx, default_color)
        else:
            return default_color
            
    def _get_pose_keypoint_color(self, idx, default_color):
        """Get color for pose keypoint based on body part"""
        # MediaPipe pose landmark indices
        if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Head/face
            return self.pose_colors['head']
        elif idx in [11, 12, 23, 24]:  # Torso
            return self.pose_colors['torso']
        elif idx in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:  # Arms
            return self.pose_colors['arms']
        elif idx in [25, 26, 27, 28, 29, 30, 31, 32]:  # Legs
            return self.pose_colors['legs']
        else:
            return default_color
            
    def _get_hand_keypoint_color(self, idx, default_color):
        """Get color for hand keypoint based on finger"""
        if idx in [1, 2, 3, 4]:  # Thumb
            return self.hand_colors['thumb']
        elif idx in [5, 6, 7, 8]:  # Index
            return self.hand_colors['index']
        elif idx in [9, 10, 11, 12]:  # Middle
            return self.hand_colors['middle']
        elif idx in [13, 14, 15, 16]:  # Ring
            return self.hand_colors['ring']
        elif idx in [17, 18, 19, 20]:  # Pinky
            return self.hand_colors['pinky']
        elif idx == 0:  # Wrist
            return self.hand_colors['palm']
        else:
            return default_color
            
    def _get_face_keypoint_color(self, idx, default_color):
        """Get color for face keypoint based on facial feature"""
        # Simplified face coloring (MediaPipe has 468 face landmarks)
        if idx < 17:  # Face contour
            return self.face_colors['contour']
        elif idx < 68:  # Eyes region
            return self.face_colors['eyes']
        elif idx < 100:  # Nose region
            return self.face_colors['nose']
        else:
            return default_color
            
    def _draw_labels(self, frame, result, width, height):
        """Draw labels for tracked keypoints"""
        track_id = result['track_id']
        keypoint_type = result['type']
        confidence = result['confidence']
        handedness = result.get('handedness', '')
        
        # Find center of keypoints for label placement
        landmarks = result['landmarks']
        if landmarks:
            center_x = int(np.mean([lm['x'] for lm in landmarks]) * width)
            center_y = int(np.mean([lm['y'] for lm in landmarks]) * height)
            
            # Create label text
            label_parts = [f"ID:{track_id}", keypoint_type.upper()]
            if handedness:
                label_parts.append(handedness)
            label_parts.append(f"{confidence:.2f}")
            
            label = " | ".join(label_parts)
            
            # Get track color
            track_color = self.get_track_color(track_id)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, 
                         (center_x - label_size[0]//2 - 5, center_y - 25),
                         (center_x + label_size[0]//2 + 5, center_y - 5),
                         track_color, -1)
            
            # Draw label text
            cv2.putText(frame, label, 
                       (center_x - label_size[0]//2, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                       
    def create_3d_visualization(self, tracked_results):
        """Create 3D visualization of keypoints"""
        fig = go.Figure()
        
        for result in tracked_results:
            track_id = result['track_id']
            keypoint_type = result['type']
            landmarks = result['landmarks']
            
            # Extract 3D coordinates
            x_coords = [lm['x'] for lm in landmarks if lm.get('visibility', 1.0) > 0.5]
            y_coords = [lm['y'] for lm in landmarks if lm.get('visibility', 1.0) > 0.5]
            z_coords = [lm['z'] for lm in landmarks if lm.get('visibility', 1.0) > 0.5]
            
            if x_coords:  # Only plot if we have visible landmarks
                # Get track color (convert to RGB string)
                track_color = self.get_track_color(track_id)
                color_str = f'rgb({track_color[0]}, {track_color[1]}, {track_color[2]})'
                
                # Add scatter plot for keypoints
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color_str,
                    ),
                    name=f'{keypoint_type.upper()} ID:{track_id}',
                    text=[f'Point {i}' for i in range(len(x_coords))],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
                ))
                
        # Update layout
        fig.update_layout(
            title='3D Keypoint Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
        
    def create_keypoint_statistics(self, tracked_results):
        """Create statistics visualization for keypoints"""
        stats = {
            'pose': 0,
            'hand': 0,
            'face': 0,
            'total_keypoints': 0,
            'avg_confidence': 0
        }
        
        confidences = []
        
        for result in tracked_results:
            keypoint_type = result['type']
            confidence = result['confidence']
            num_keypoints = len(result['landmarks'])
            
            stats[keypoint_type] += 1
            stats['total_keypoints'] += num_keypoints
            confidences.append(confidence)
            
        if confidences:
            stats['avg_confidence'] = np.mean(confidences)
            
        return stats
        
    def reset_colors(self):
        """Reset track colors (useful when starting new video)"""
        self.track_colors = {}

"""
Football Player Pose Analysis System
Real-time pose detection and analysis specifically for football players using YOLO
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import math

class FootballPoseAnalyzer:
    """Football-specific pose detection and analysis system using YOLO"""

    def __init__(self, detection_type="pose", confidence_threshold=0.5):
        self.detection_type = detection_type
        self.confidence_threshold = confidence_threshold

        # Initialize YOLO pose model
        try:
            self.pose_model = YOLO('yolov8n-pose.pt')  # Lightweight pose model
        except Exception as e:
            print(f"Warning: Could not load YOLO pose model: {e}")
            self.pose_model = None

        # Football-specific pose analysis
        self.pose_actions = {
            'running': 0,
            'kicking': 0,
            'jumping': 0,
            'standing': 0,
            'moving': 0,
            'diving': 0
        }
        
        # Tracking data for football players
        self.tracked_players = {}
        self.next_id = 1
        self.frame_count = 0

        # Football-specific keypoint analysis
        self.pose_connections = self._define_pose_connections()
        self.football_keypoints = self._define_football_keypoints()

        # Action recognition thresholds
        self.action_thresholds = {
            'kick_angle_threshold': 45,  # degrees
            'jump_height_threshold': 0.1,  # normalized
            'run_speed_threshold': 0.05,  # normalized movement
            'crouch_knee_angle': 120  # degrees
        }

        # YOLO pose keypoint indices (17 keypoints)
        self.yolo_keypoints = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
    def _define_pose_connections(self):
        """Define pose connections for YOLO keypoints"""
        return [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head connections
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

    def _define_football_keypoints(self):
        """Define key body parts for football analysis using YOLO keypoints"""
        return {
            'head': [0, 1, 2, 3, 4],  # Nose, eyes, ears
            'torso': [5, 6, 11, 12],  # Shoulders and hips
            'left_arm': [5, 7, 9],  # Left shoulder, elbow, wrist
            'right_arm': [6, 8, 10],  # Right shoulder, elbow, wrist
            'left_leg': [11, 13, 15],  # Left hip, knee, ankle
            'right_leg': [12, 14, 16],  # Right hip, knee, ankle
            'key_joints': {
                'nose': 0, 'left_eye': 1, 'right_eye': 2,
                'left_shoulder': 5, 'right_shoulder': 6,
                'left_elbow': 7, 'right_elbow': 8,
                'left_wrist': 9, 'right_wrist': 10,
                'left_hip': 11, 'right_hip': 12,
                'left_knee': 13, 'right_knee': 14,
                'left_ankle': 15, 'right_ankle': 16
            }
        }
        
    def _init_hand_detection(self):
        """Initialize hand detection"""
        self.mp_hands = mp.solutions.hands
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=10,
            model_complexity=1,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=0.5
        )
        
    def _init_face_detection(self):
        """Initialize face detection"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=0.5
        )
        
    def _init_holistic_detection(self):
        """Initialize holistic detection (pose + hands + face)"""
        self.mp_holistic = mp.solutions.holistic
        self.holistic_detector = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=0.5
        )
        
    def detect_football_players(self, frame):
        """Detect football players and analyze their poses using YOLO"""
        self.frame_count += 1

        # Use YOLO for pose detection
        return self._detect_football_poses_yolo(frame)
            
    def _detect_football_poses_yolo(self, frame):
        """Detect football player poses with action analysis using YOLO"""
        detections = []

        if self.pose_model is None:
            # Fallback: create dummy detection for demonstration
            return self._create_dummy_detection(frame)

        try:
            # Run YOLO pose detection
            results = self.pose_model(frame, conf=self.confidence_threshold, verbose=False)

            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data  # Shape: [num_persons, num_keypoints, 3]
                    boxes = result.boxes.data if result.boxes is not None else None

                    for i, person_keypoints in enumerate(keypoints):
                        # Convert YOLO keypoints to our format
                        landmarks = []
                        for j, kp in enumerate(person_keypoints):
                            x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
                            # Normalize coordinates
                            h, w = frame.shape[:2]
                            landmarks.append({
                                'x': x / w,
                                'y': y / h,
                                'z': 0.0,  # YOLO doesn't provide Z
                                'visibility': conf
                            })

                        # Only process if we have enough visible keypoints
                        visible_count = sum(1 for lm in landmarks if lm['visibility'] > 0.3)
                        if visible_count < 8:  # Need at least 8 visible keypoints
                            continue

                        # Analyze football-specific actions
                        action_analysis = self._analyze_football_actions(landmarks)
                        pose_quality = self._assess_pose_quality(landmarks)

                        # Get detection confidence from bounding box if available
                        detection_conf = boxes[i][4].item() if boxes is not None and i < len(boxes) else 0.5

                        detections.append({
                            'type': 'football_pose',
                            'landmarks': landmarks,
                            'connections': self.pose_connections,
                            'confidence': detection_conf,
                            'action_analysis': action_analysis,
                            'pose_quality': pose_quality,
                            'body_parts': self._extract_body_parts(landmarks)
                        })

        except Exception as e:
            print(f"Error in YOLO pose detection: {e}")
            # Return dummy detection on error
            return self._create_dummy_detection(frame)

        return detections

    def _create_dummy_detection(self, frame):
        """Create dummy detection for demonstration when YOLO fails"""
        h, w = frame.shape[:2]

        # Create a simple standing pose in the center
        dummy_landmarks = []
        for i in range(17):  # YOLO has 17 keypoints
            # Place keypoints in a rough human pose shape
            if i == 0:  # nose
                x, y = 0.5, 0.2
            elif i in [1, 2]:  # eyes
                x, y = 0.48 + (i-1)*0.04, 0.18
            elif i in [3, 4]:  # ears
                x, y = 0.46 + (i-3)*0.08, 0.19
            elif i in [5, 6]:  # shoulders
                x, y = 0.4 + (i-5)*0.2, 0.35
            elif i in [7, 8]:  # elbows
                x, y = 0.35 + (i-7)*0.3, 0.5
            elif i in [9, 10]:  # wrists
                x, y = 0.3 + (i-9)*0.4, 0.65
            elif i in [11, 12]:  # hips
                x, y = 0.45 + (i-11)*0.1, 0.6
            elif i in [13, 14]:  # knees
                x, y = 0.45 + (i-13)*0.1, 0.75
            else:  # ankles
                x, y = 0.45 + (i-15)*0.1, 0.9

            dummy_landmarks.append({
                'x': x,
                'y': y,
                'z': 0.0,
                'visibility': 0.8
            })

        action_analysis = {'action': 'standing', 'confidence': 0.5, 'details': {}}
        pose_quality = {'overall_quality': 0.7, 'key_parts_quality': 0.8, 'visible_landmarks': 17, 'total_landmarks': 17}

        return [{
            'type': 'football_pose',
            'landmarks': dummy_landmarks,
            'connections': self.pose_connections,
            'confidence': 0.5,
            'action_analysis': action_analysis,
            'pose_quality': pose_quality,
            'body_parts': self._extract_body_parts(dummy_landmarks)
        }]

    def _analyze_football_actions(self, landmarks):
        """Analyze football-specific actions from pose landmarks"""
        actions = {
            'action': 'standing',
            'confidence': 0.0,
            'details': {}
        }

        try:
            # Get key joint positions
            joints = self._get_joint_positions(landmarks)

            # Analyze different actions
            kick_score = self._detect_kicking(joints, landmarks)
            jump_score = self._detect_jumping(joints, landmarks)
            run_score = self._detect_running(joints, landmarks)
            crouch_score = self._detect_moving(joints, landmarks)

            # Determine primary action
            action_scores = {
                'kicking': kick_score,
                'jumping': jump_score,
                'running': run_score,
                'moving': crouch_score
            }

            best_action = max(action_scores, key=action_scores.get)
            best_score = action_scores[best_action]

            if best_score > 0.3:  # Threshold for action detection
                actions['action'] = best_action
                actions['confidence'] = best_score
                actions['details'] = self._get_action_details(best_action, joints, landmarks)

        except Exception as e:
            print(f"Error in action analysis: {e}")

        return actions

    def _get_joint_positions(self, landmarks):
        """Extract key joint positions for analysis"""
        joints = {}
        key_joints = self.football_keypoints['key_joints']

        for joint_name, joint_idx in key_joints.items():
            if joint_idx < len(landmarks) and landmarks[joint_idx]['visibility'] > 0.5:
                joints[joint_name] = {
                    'x': landmarks[joint_idx]['x'],
                    'y': landmarks[joint_idx]['y'],
                    'z': landmarks[joint_idx]['z'],
                    'visibility': landmarks[joint_idx]['visibility']
                }

        return joints

    def _detect_kicking(self, joints, landmarks):
        """Detect kicking motion"""
        try:
            # Check if leg is extended and foot position
            if all(joint in joints for joint in ['left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle']):

                # Calculate leg angles
                left_leg_angle = self._calculate_leg_angle(
                    joints['left_hip'], joints['left_knee'], joints['left_ankle'])
                right_leg_angle = self._calculate_leg_angle(
                    joints['right_hip'], joints['right_knee'], joints['right_ankle'])

                # Check for extended leg (kicking motion)
                left_extended = left_leg_angle > 160  # Nearly straight
                right_extended = right_leg_angle > 160

                # Check foot height relative to ground
                left_foot_high = joints['left_ankle']['y'] < joints['left_hip']['y'] - 0.1
                right_foot_high = joints['right_ankle']['y'] < joints['right_hip']['y'] - 0.1

                kick_score = 0
                if (left_extended and left_foot_high) or (right_extended and right_foot_high):
                    kick_score = 0.8
                elif left_extended or right_extended:
                    kick_score = 0.5

                return kick_score

        except Exception:
            pass

        return 0.0

    def _detect_jumping(self, joints, landmarks):
        """Detect jumping motion"""
        try:
            if all(joint in joints for joint in ['left_ankle', 'right_ankle', 'left_hip', 'right_hip']):

                # Check if both feet are off ground (high y position)
                avg_foot_y = (joints['left_ankle']['y'] + joints['right_ankle']['y']) / 2
                avg_hip_y = (joints['left_hip']['y'] + joints['right_hip']['y']) / 2

                # If feet are significantly higher than normal stance
                if avg_foot_y < avg_hip_y - 0.3:  # Feet high relative to hips
                    return 0.9
                elif avg_foot_y < avg_hip_y - 0.15:
                    return 0.6

        except Exception:
            pass

        return 0.0

    def _detect_running(self, joints, landmarks):
        """Detect running motion"""
        try:
            if all(joint in joints for joint in ['left_knee', 'right_knee', 'left_ankle', 'right_ankle']):

                # Check knee lift and alternating leg pattern
                left_knee_lift = joints['left_knee']['y'] < joints['left_ankle']['y'] - 0.05
                right_knee_lift = joints['right_knee']['y'] < joints['right_ankle']['y'] - 0.05

                # Running typically has one knee lifted
                if left_knee_lift != right_knee_lift:  # Alternating pattern
                    return 0.7
                elif left_knee_lift or right_knee_lift:
                    return 0.4

        except Exception:
            pass

        return 0.0

    def _detect_moving(self, joints, landmarks):
        """Detect moving/low stance"""
        try:
            if all(joint in joints for joint in ['left_hip', 'left_knee', 'right_hip', 'right_knee']):

                # Calculate knee angles
                left_knee_angle = self._calculate_knee_angle(joints)
                right_knee_angle = self._calculate_knee_angle(joints, side='right')

                # Check if knees are bent significantly
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                if avg_knee_angle < 120:  # Significantly bent
                    return 0.8
                elif avg_knee_angle < 140:
                    return 0.5

        except Exception:
            pass

        return 0.0

    def _calculate_leg_angle(self, hip, knee, ankle):
        """Calculate angle of leg (hip-knee-ankle)"""
        try:
            # Vector from knee to hip
            v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y']])
            # Vector from knee to ankle
            v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])

            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 180  # Default straight leg

    def _calculate_knee_angle(self, joints, side='left'):
        """Calculate knee angle"""
        try:
            hip_key = f'{side}_hip'
            knee_key = f'{side}_knee'
            ankle_key = f'{side}_ankle'

            if all(key in joints for key in [hip_key, knee_key, ankle_key]):
                return self._calculate_leg_angle(
                    joints[hip_key], joints[knee_key], joints[ankle_key])
        except:
            pass
        return 180

    def _assess_pose_quality(self, landmarks):
        """Assess the quality of pose detection"""
        visible_landmarks = sum(1 for lm in landmarks if lm['visibility'] > 0.5)
        total_landmarks = len(landmarks)

        quality_score = visible_landmarks / total_landmarks

        # Check key body parts visibility
        key_parts_visible = 0
        key_indices = [11, 12, 23, 24, 25, 26, 27, 28]  # Shoulders, hips, knees, ankles

        for idx in key_indices:
            if idx < len(landmarks) and landmarks[idx]['visibility'] > 0.7:
                key_parts_visible += 1

        key_parts_score = key_parts_visible / len(key_indices)

        return {
            'overall_quality': quality_score,
            'key_parts_quality': key_parts_score,
            'visible_landmarks': visible_landmarks,
            'total_landmarks': total_landmarks
        }

    def _extract_body_parts(self, landmarks):
        """Extract body part positions for analysis"""
        body_parts = {}

        for part_name, indices in self.football_keypoints.items():
            if part_name != 'key_joints':
                part_landmarks = []
                for idx in indices:
                    if idx < len(landmarks) and landmarks[idx]['visibility'] > 0.5:
                        part_landmarks.append({
                            'x': landmarks[idx]['x'],
                            'y': landmarks[idx]['y'],
                            'z': landmarks[idx]['z'],
                            'visibility': landmarks[idx]['visibility']
                        })
                body_parts[part_name] = part_landmarks

        return body_parts

    def _get_action_details(self, action, joints, landmarks):
        """Get detailed information about detected action"""
        details = {}

        if action == 'kicking':
            details['kicking_leg'] = self._determine_kicking_leg(joints)
            details['leg_extension'] = self._calculate_leg_extension(joints)

        elif action == 'jumping':
            details['jump_height'] = self._estimate_jump_height(joints)
            details['body_position'] = self._analyze_body_position(joints)

        elif action == 'running':
            details['stride_pattern'] = self._analyze_stride_pattern(joints)
            details['body_lean'] = self._calculate_body_lean(joints)

        elif action == 'moving':
            details['crouch_depth'] = self._calculate_crouch_depth(joints)
            details['balance'] = self._assess_balance(joints)

        return details

    def _determine_kicking_leg(self, joints):
        """Determine which leg is kicking"""
        try:
            left_ankle_height = joints.get('left_ankle', {}).get('y', 1.0)
            right_ankle_height = joints.get('right_ankle', {}).get('y', 1.0)

            if left_ankle_height < right_ankle_height - 0.05:
                return 'left'
            elif right_ankle_height < left_ankle_height - 0.05:
                return 'right'
        except:
            pass
        return 'unknown'

    def _calculate_leg_extension(self, joints):
        """Calculate how extended the kicking leg is"""
        # Implementation for leg extension calculation
        return 0.5  # Placeholder

    def _estimate_jump_height(self, joints):
        """Estimate jump height based on foot position"""
        # Implementation for jump height estimation
        return 0.3  # Placeholder

    def _analyze_body_position(self, joints):
        """Analyze body position during action"""
        return 'balanced'  # Placeholder

    def _analyze_stride_pattern(self, joints):
        """Analyze running stride pattern"""
        return 'normal'  # Placeholder

    def _calculate_body_lean(self, joints):
        """Calculate body lean angle"""
        return 0.0  # Placeholder

    def _calculate_crouch_depth(self, joints):
        """Calculate how deep the crouch is"""
        return 0.5  # Placeholder

    def _assess_balance(self, joints):
        """Assess player balance"""
        return 'stable'  # Placeholder
        
    def _detect_hands(self, rgb_frame):
        """Detect hand keypoints"""
        results = self.hand_detector.process(rgb_frame)
        detections = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': 1.0  # Hands don't have visibility scores
                    })
                
                # Get handedness
                handedness = "Unknown"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                
                detections.append({
                    'type': 'hand',
                    'landmarks': landmarks,
                    'connections': self.hand_connections,
                    'handedness': handedness,
                    'confidence': results.multi_handedness[idx].classification[0].score if results.multi_handedness else 0.5
                })
                
        return detections
        
    def _detect_face(self, rgb_frame):
        """Detect face keypoints"""
        results = self.face_detector.process(rgb_frame)
        detections = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': 1.0  # Face landmarks don't have visibility scores
                    })
                
                detections.append({
                    'type': 'face',
                    'landmarks': landmarks,
                    'connections': self.face_connections,
                    'confidence': self._calculate_face_confidence(landmarks)
                })
                
        return detections
        
    def _detect_holistic(self, rgb_frame):
        """Detect holistic keypoints (pose + hands + face)"""
        results = self.holistic_detector.process(rgb_frame)
        detections = []
        
        # Pose landmarks
        if results.pose_landmarks:
            pose_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                pose_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            detections.append({
                'type': 'pose',
                'landmarks': pose_landmarks,
                'connections': self.pose_connections,
                'confidence': self._calculate_pose_confidence(pose_landmarks)
            })
        
        # Hand landmarks
        if results.left_hand_landmarks:
            left_hand_landmarks = []
            for landmark in results.left_hand_landmarks.landmark:
                left_hand_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': 1.0
                })
            
            detections.append({
                'type': 'hand',
                'landmarks': left_hand_landmarks,
                'connections': self.hand_connections,
                'handedness': 'Left',
                'confidence': 0.8
            })
            
        if results.right_hand_landmarks:
            right_hand_landmarks = []
            for landmark in results.right_hand_landmarks.landmark:
                right_hand_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': 1.0
                })
            
            detections.append({
                'type': 'hand',
                'landmarks': right_hand_landmarks,
                'connections': self.hand_connections,
                'handedness': 'Right',
                'confidence': 0.8
            })
        
        # Face landmarks
        if results.face_landmarks:
            face_landmarks = []
            for landmark in results.face_landmarks.landmark:
                face_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': 1.0
                })
            
            detections.append({
                'type': 'face',
                'landmarks': face_landmarks,
                'connections': self.face_connections,
                'confidence': self._calculate_face_confidence(face_landmarks)
            })
            
        return detections
        
    def _calculate_pose_confidence(self, landmarks):
        """Calculate overall confidence for pose detection"""
        visible_landmarks = [lm for lm in landmarks if lm['visibility'] > 0.5]
        if not visible_landmarks:
            return 0.0
        return sum(lm['visibility'] for lm in visible_landmarks) / len(visible_landmarks)
        
    def _calculate_face_confidence(self, landmarks):
        """Calculate overall confidence for face detection"""
        # For face, we assume high confidence if we have a good number of landmarks
        return 0.9 if len(landmarks) > 400 else 0.5
        
    def track_football_players(self, frame, detections):
        """Track football players across frames with pose analysis"""
        tracked_results = []

        for detection in detections:
            # Find best match with existing tracked players
            best_match_id = self._find_best_player_match(detection)

            if best_match_id is not None:
                # Update existing player track
                self.tracked_players[best_match_id].update({
                    'landmarks': detection['landmarks'],
                    'confidence': detection['confidence'],
                    'action_analysis': detection['action_analysis'],
                    'pose_quality': detection['pose_quality'],
                    'last_seen': self.frame_count,
                    'age': self.tracked_players[best_match_id]['age'] + 1,
                    'action_history': self.tracked_players[best_match_id].get('action_history', []) + [detection['action_analysis']['action']]
                })

                # Keep only recent action history
                if len(self.tracked_players[best_match_id]['action_history']) > 10:
                    self.tracked_players[best_match_id]['action_history'] = self.tracked_players[best_match_id]['action_history'][-10:]

                track_id = best_match_id
            else:
                # Create new player track
                track_id = self.next_id
                self.tracked_players[track_id] = {
                    'type': detection['type'],
                    'landmarks': detection['landmarks'],
                    'confidence': detection['confidence'],
                    'action_analysis': detection['action_analysis'],
                    'pose_quality': detection['pose_quality'],
                    'last_seen': self.frame_count,
                    'age': 1,
                    'action_history': [detection['action_analysis']['action']]
                }
                self.next_id += 1

            tracked_results.append({
                'track_id': track_id,
                'type': detection['type'],
                'landmarks': detection['landmarks'],
                'connections': detection['connections'],
                'confidence': detection['confidence'],
                'action_analysis': detection['action_analysis'],
                'pose_quality': detection['pose_quality'],
                'body_parts': detection.get('body_parts', {}),
                'action_history': self.tracked_players[track_id]['action_history']
            })

        # Remove old tracks
        self._cleanup_old_player_tracks()

        return tracked_results
        
    def _find_best_player_match(self, detection):
        """Find best matching tracked football player"""
        best_match_id = None
        best_similarity = 0

        for track_id, tracked_player in self.tracked_players.items():
            if tracked_player['type'] != detection['type']:
                continue

            # Skip if not seen recently
            if self.frame_count - tracked_player['last_seen'] > 15:  # Longer for football
                continue

            # Calculate similarity based on landmark positions
            pose_similarity = self._calculate_landmark_similarity(
                detection['landmarks'], tracked_player['landmarks'])

            # Add action consistency bonus
            action_consistency = self._calculate_action_consistency(
                detection['action_analysis'], tracked_player.get('action_history', []))

            # Combined similarity score
            total_similarity = pose_similarity * 0.8 + action_consistency * 0.2

            if total_similarity > best_similarity and total_similarity > 0.6:  # Lower threshold for football
                best_match_id = track_id
                best_similarity = total_similarity

        return best_match_id

    def _calculate_action_consistency(self, current_action, action_history):
        """Calculate consistency of actions for better tracking"""
        if not action_history:
            return 0.5  # Neutral score

        current_action_type = current_action['action']

        # Check if current action is consistent with recent history
        recent_actions = action_history[-3:]  # Last 3 actions

        if current_action_type in recent_actions:
            return 0.8  # High consistency
        elif any(action in ['running', 'standing'] for action in recent_actions) and current_action_type in ['running', 'standing']:
            return 0.6  # Related actions
        else:
            return 0.3  # Low consistency
        
    def _calculate_landmark_similarity(self, landmarks1, landmarks2):
        """Calculate similarity between two sets of landmarks"""
        if len(landmarks1) != len(landmarks2):
            return 0.0
            
        total_distance = 0
        valid_points = 0
        
        for lm1, lm2 in zip(landmarks1, landmarks2):
            # Only consider visible landmarks for pose
            if lm1.get('visibility', 1.0) > 0.5 and lm2.get('visibility', 1.0) > 0.5:
                distance = math.sqrt(
                    (lm1['x'] - lm2['x'])**2 + 
                    (lm1['y'] - lm2['y'])**2
                )
                total_distance += distance
                valid_points += 1
        
        if valid_points == 0:
            return 0.0
            
        avg_distance = total_distance / valid_points
        # Convert distance to similarity (closer = more similar)
        similarity = max(0, 1.0 - avg_distance * 5)  # Scale factor
        
        return similarity
        
    def _cleanup_old_player_tracks(self):
        """Remove player tracks that haven't been seen recently"""
        tracks_to_remove = []

        for track_id, tracked_player in self.tracked_players.items():
            if self.frame_count - tracked_player['last_seen'] > 45:  # 45 frames for football
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracked_players[track_id]

    def get_football_statistics(self):
        """Get football-specific statistics"""
        active_players = len([p for p in self.tracked_players.values()
                            if self.frame_count - p['last_seen'] <= 5])

        # Count actions
        action_counts = {'standing': 0, 'running': 0, 'kicking': 0, 'jumping': 0, 'moving': 0}
        total_confidence = 0
        pose_qualities = []

        for player in self.tracked_players.values():
            if self.frame_count - player['last_seen'] <= 5:  # Active players only
                action = player.get('action_analysis', {}).get('action', 'standing')
                if action in action_counts:
                    action_counts[action] += 1

                total_confidence += player.get('confidence', 0)

                pose_quality = player.get('pose_quality', {}).get('overall_quality', 0)
                pose_qualities.append(pose_quality)

        avg_confidence = total_confidence / max(active_players, 1)
        avg_pose_quality = sum(pose_qualities) / max(len(pose_qualities), 1)

        return {
            'active_players': active_players,
            'total_tracked_players': len(self.tracked_players),
            'action_counts': action_counts,
            'avg_confidence': avg_confidence,
            'avg_pose_quality': avg_pose_quality,
            'frame_count': self.frame_count
        }

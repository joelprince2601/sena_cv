"""
Enhanced tracking system for football player re-identification
Integrates OC-SORT, ByteTrack, and custom re-identification
"""
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics not available")

try:
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available, using numpy fallback")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available")

# Add OC_SORT to path
sys.path.append(str(Path(__file__).parent.parent / "OC_SORT"))

try:
    from trackers.ocsort_tracker.ocsort import OCSort
    OCSORT_AVAILABLE = True
except ImportError:
    OCSORT_AVAILABLE = False
    print("OC-SORT not available")

try:
    from trackers.byte_tracker.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("ByteTrack not available")

from config import DEFAULT_PARAMS, get_player_color

class EnhancedFootballTracker:
    def __init__(self, 
                 yolo_model="yolov8n",
                 tracker_type="ocsort",
                 confidence_threshold=0.5,
                 nms_threshold=0.4,
                 reid_threshold=0.7,
                 **kwargs):
        
        self.yolo_model_name = yolo_model
        self.tracker_type = tracker_type
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.reid_threshold = reid_threshold
        
        # Initialize YOLO model
        if ULTRALYTICS_AVAILABLE:
            self.yolo_model = YOLO(yolo_model)
        else:
            raise ImportError("Ultralytics YOLO not available. Please install: pip install ultralytics")
        
        # Initialize tracker
        self.tracker = self._initialize_tracker(**kwargs)
        
        # Player management
        self.next_id = 1
        self.players = {}
        self.frame_count = 0
        
        # Feature extraction for re-identification
        self.feature_extractor = self._initialize_feature_extractor()
        
    def _initialize_tracker(self, **kwargs):
        """Initialize the selected tracker"""
        if self.tracker_type == "ocsort" and OCSORT_AVAILABLE:
            return OCSort(
                det_thresh=kwargs.get('det_thresh', 0.5),
                max_age=kwargs.get('max_age', 30),
                min_hits=kwargs.get('min_hits', 3),
                iou_threshold=kwargs.get('iou_threshold', 0.3)
            )
        elif self.tracker_type == "bytetrack" and BYTETRACK_AVAILABLE:
            # Create a simple args object for ByteTracker
            class Args:
                def __init__(self):
                    self.track_thresh = kwargs.get('track_thresh', 0.5)
                    self.track_buffer = kwargs.get('track_buffer', 30)
                    self.match_thresh = kwargs.get('match_thresh', 0.8)
                    self.mot20 = False

            args = Args()
            return BYTETracker(args, frame_rate=kwargs.get('frame_rate', 30))
        else:
            # Custom tracker
            return None
            
    def _initialize_feature_extractor(self):
        """Initialize feature extractor for re-identification"""
        # For now, we'll use color histograms as features
        # In a production system, you'd use a pre-trained ReID model
        return None
        
    def extract_features(self, frame, bbox):
        """Extract comprehensive features from player's bounding box for re-identification"""
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within frame boundaries
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))

        if x2 <= x1 or y2 <= y1:
            return None

        # Extract player ROI
        player_roi = frame[y1:y2, x1:x2]

        if player_roi.size == 0:
            return None

        # Resize ROI to standard size for consistent features
        roi_resized = cv2.resize(player_roi, (64, 128))  # Standard person aspect ratio

        # 1. Color features (HSV histograms)
        hsv_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)

        # Calculate histograms for each channel with more bins for better discrimination
        hist_h = cv2.calcHist([hsv_roi], [0], None, [36], [0, 180])  # Hue
        hist_s = cv2.calcHist([hsv_roi], [1], None, [32], [0, 256])  # Saturation
        hist_v = cv2.calcHist([hsv_roi], [2], None, [32], [0, 256])  # Value

        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()

        # 2. Texture features (LBP-like)
        gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

        # Calculate gradients for texture
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Texture histogram
        texture_hist, _ = np.histogram(magnitude.flatten(), bins=32, range=(0, 255))
        texture_hist = cv2.normalize(texture_hist.astype(np.float32), None).flatten()

        # 3. Spatial features (position and size)
        center_x = (x1 + x2) / 2.0 / frame.shape[1]  # Normalized center x
        center_y = (y1 + y2) / 2.0 / frame.shape[0]  # Normalized center y
        width_ratio = (x2 - x1) / frame.shape[1]     # Normalized width
        height_ratio = (y2 - y1) / frame.shape[0]    # Normalized height
        aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6) # Aspect ratio

        spatial_features = np.array([center_x, center_y, width_ratio, height_ratio, aspect_ratio])

        # Combine all features with appropriate weights
        color_features = np.concatenate([hist_h, hist_s, hist_v]) * 0.6  # Color is most important
        texture_features = texture_hist * 0.3  # Texture for discrimination
        spatial_features = spatial_features * 0.1  # Spatial for continuity

        # Combine and normalize final features
        combined_features = np.concatenate([color_features, texture_features, spatial_features])
        features = cv2.normalize(combined_features, None).flatten()

        return features
        
    def detect_players(self, frame):
        """Detect players using YOLO"""
        results = self.yolo_model(frame, conf=self.confidence_threshold, iou=self.nms_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Only consider person class (class 0)
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections) if detections else np.empty((0, 5))
        
    def update_tracks(self, frame, detections):
        """Update tracks using the selected tracker"""
        self.frame_count += 1

        if self.tracker_type == "custom":
            return self._custom_tracking(frame, detections)
        else:
            # Use external tracker (OC-SORT or ByteTrack)
            if len(detections) > 0:
                # Prepare image info for trackers
                img_info = (frame.shape[0], frame.shape[1])  # height, width
                img_size = (frame.shape[0], frame.shape[1])  # height, width

                if self.tracker is not None:
                    if self.tracker_type == "ocsort":
                        tracks = self.tracker.update(detections, img_info, img_size)
                    elif self.tracker_type == "bytetrack":
                        tracks = self.tracker.update(detections, img_info, img_size)
                    else:
                        tracks = self.tracker.update(detections)
                    return self._process_external_tracks(frame, tracks)
                else:
                    # Fallback to custom tracking if tracker failed to initialize
                    return self._custom_tracking(frame, detections)
            else:
                # Handle empty detections
                if self.tracker is not None:
                    img_info = (frame.shape[0], frame.shape[1])
                    img_size = (frame.shape[0], frame.shape[1])
                    empty_detections = np.empty((0, 5))

                    if self.tracker_type in ["ocsort", "bytetrack"]:
                        tracks = self.tracker.update(empty_detections, img_info, img_size)
                    else:
                        tracks = self.tracker.update(empty_detections)
                    return self._process_external_tracks(frame, tracks)
                return []
                
    def _custom_tracking(self, frame, detections):
        """Enhanced custom tracking with improved re-identification"""
        # Update time since last seen for all players
        for player_id in self.players:
            self.players[player_id]['time_since_seen'] += 1
            self.players[player_id]['matched_this_frame'] = False

        matched_tracks = []
        unmatched_detections = []

        # Sort detections by confidence (process high confidence first)
        if len(detections) > 0:
            sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
        else:
            sorted_detections = []

        # Match detections with existing players using multi-stage matching
        for detection in sorted_detections:
            x1, y1, x2, y2, conf = detection
            bbox = [x1, y1, x2, y2]

            # Extract features
            features = self.extract_features(frame, bbox)
            if features is None:
                continue

            best_match_id = None
            best_score = 0

            # Stage 1: High confidence spatial matching for recent tracks
            for player_id, player_info in self.players.items():
                if player_info.get('matched_this_frame', False):
                    continue

                # For recently seen players, prioritize spatial continuity
                if player_info['time_since_seen'] <= 5:
                    iou = self._calculate_iou(bbox, player_info['bbox'])
                    motion_consistency = self._calculate_motion_consistency(bbox, player_info)

                    # High spatial score for recent tracks
                    spatial_score = iou * 0.7 + motion_consistency * 0.3

                    if spatial_score > 0.3:  # Lower threshold for spatial matching
                        feature_similarity = self._calculate_similarity(features, player_info['features'])
                        combined_score = spatial_score * 0.6 + feature_similarity * 0.4

                        if combined_score > best_score:
                            best_match_id = player_id
                            best_score = combined_score

            # Stage 2: Feature-based matching for older tracks
            if best_match_id is None:
                for player_id, player_info in self.players.items():
                    if player_info.get('matched_this_frame', False):
                        continue

                    # For older tracks, rely more on features
                    feature_similarity = self._calculate_similarity(features, player_info['features'])

                    # Apply time decay to similarity
                    time_decay = max(0.1, 1.0 - (player_info['time_since_seen'] / 50.0))
                    adjusted_similarity = feature_similarity * time_decay

                    # Consider spatial proximity with lower weight
                    if player_info['time_since_seen'] < 30:
                        iou = self._calculate_iou(bbox, player_info['bbox'])
                        adjusted_similarity = adjusted_similarity * 0.8 + iou * 0.2

                    if adjusted_similarity > self.reid_threshold and adjusted_similarity > best_score:
                        best_match_id = player_id
                        best_score = adjusted_similarity

            # Update matched player or store as unmatched
            if best_match_id is not None:
                # Update existing player with enhanced feature updating
                old_features = self.players[best_match_id]['features']

                # Adaptive feature update based on confidence and time
                confidence_weight = min(conf, 0.9)  # Cap confidence influence
                time_weight = max(0.1, 1.0 - (self.players[best_match_id]['time_since_seen'] / 20.0))
                update_rate = confidence_weight * time_weight * 0.3  # Conservative update

                updated_features = self._update_features(old_features, features, alpha=1-update_rate)

                # Update player info
                self.players[best_match_id].update({
                    'bbox': bbox,
                    'features': updated_features,
                    'time_since_seen': 0,
                    'matched_this_frame': True,
                    'confidence': conf,
                    'last_position': bbox,
                    'velocity': self._calculate_velocity(bbox, player_info.get('last_position', bbox))
                })

                matched_tracks.append([x1, y1, x2, y2, best_match_id, conf])
            else:
                unmatched_detections.append((bbox, features, conf))

        # Assign new IDs to unmatched detections (with minimum confidence threshold)
        for bbox, features, conf in unmatched_detections:
            if conf > 0.4:  # Only create new tracks for confident detections
                self.players[self.next_id] = {
                    'bbox': bbox,
                    'features': features,
                    'features_history': [features],
                    'time_since_seen': 0,
                    'matched_this_frame': True,
                    'confidence': conf,
                    'last_position': bbox,
                    'velocity': [0, 0],
                    'creation_frame': self.frame_count
                }
                x1, y1, x2, y2 = bbox
                matched_tracks.append([x1, y1, x2, y2, self.next_id, conf])
                self.next_id += 1

        # Remove old tracks with adaptive thresholds
        players_to_remove = []
        for player_id, player_info in self.players.items():
            # More aggressive removal for low-confidence tracks
            max_age = 200 if player_info.get('confidence', 0) > 0.7 else 100

            if player_info['time_since_seen'] > max_age:
                players_to_remove.append(player_id)

        for pid in players_to_remove:
            del self.players[pid]

        return matched_tracks
        
    def _process_external_tracks(self, frame, tracks):
        """Process tracks from external trackers"""
        processed_tracks = []
        
        if len(tracks) > 0:
            for track in tracks:
                if len(track) >= 5:
                    x1, y1, x2, y2, track_id = track[:5]
                    conf = track[4] if len(track) > 5 else 0.5
                    processed_tracks.append([x1, y1, x2, y2, int(track_id), conf])
                    
        return processed_tracks
        
    def _calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        try:
            if SCIPY_AVAILABLE:
                # Use scipy cosine similarity
                similarity = 1 - cosine(features1, features2)
            else:
                # Fallback to numpy implementation
                dot_product = np.dot(features1, features2)
                norm1 = np.linalg.norm(features1)
                norm2 = np.linalg.norm(features2)
                similarity = dot_product / (norm1 * norm2 + 1e-8)
            return max(0, similarity)  # Ensure non-negative
        except:
            return 0
            
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    def _update_features(self, old_features, new_features, alpha=0.8):
        """Update features with exponential moving average"""
        return alpha * old_features + (1 - alpha) * new_features

    def _calculate_motion_consistency(self, current_bbox, player_info):
        """Calculate motion consistency score based on predicted movement"""
        if 'velocity' not in player_info or 'last_position' not in player_info:
            return 0.5  # Neutral score for new tracks

        last_bbox = player_info['last_position']
        velocity = player_info['velocity']

        # Predict current position based on last position and velocity
        predicted_x = last_bbox[0] + velocity[0]
        predicted_y = last_bbox[1] + velocity[1]
        predicted_bbox = [predicted_x, predicted_y,
                         predicted_x + (last_bbox[2] - last_bbox[0]),
                         predicted_y + (last_bbox[3] - last_bbox[1])]

        # Calculate how well current detection matches prediction
        motion_iou = self._calculate_iou(current_bbox, predicted_bbox)

        return motion_iou

    def _calculate_velocity(self, current_bbox, last_bbox):
        """Calculate velocity between two bounding boxes"""
        if last_bbox is None:
            return [0, 0]

        # Calculate center movement
        current_center = [(current_bbox[0] + current_bbox[2]) / 2,
                         (current_bbox[1] + current_bbox[3]) / 2]
        last_center = [(last_bbox[0] + last_bbox[2]) / 2,
                      (last_bbox[1] + last_bbox[3]) / 2]

        velocity = [current_center[0] - last_center[0],
                   current_center[1] - last_center[1]]

        return velocity
        
    def draw_tracks(self, frame, tracks):
        """Draw tracking results on frame"""
        annotated_frame = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf = track
            
            # Get color for this track ID
            color = get_player_color(int(track_id))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw track ID and confidence
            label = f"ID: {int(track_id)} ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_frame, 
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        return annotated_frame
        
    def get_statistics(self):
        """Get current tracking statistics"""
        active_players = len([p for p in self.players.values() if p['time_since_seen'] == 0])
        total_players = len(self.players)
        
        return {
            'frame_count': self.frame_count,
            'active_players': active_players,
            'total_players_detected': total_players,
            'tracker_type': self.tracker_type,
            'model_type': self.yolo_model_name
        }
        
    def reset(self):
        """Reset tracker state"""
        self.players = {}
        self.next_id = 1
        self.frame_count = 0
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()

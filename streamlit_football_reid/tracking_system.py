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

# Import simplified trackers (always available)
try:
    from simple_trackers import SimpleBYTETracker, SimpleOCSORT
    SIMPLE_TRACKERS_AVAILABLE = True
    print("✅ Simplified trackers available")
except ImportError:
    SIMPLE_TRACKERS_AVAILABLE = False
    print("⚠️ Simplified trackers not available")

# Try to import external trackers (optional)
OCSORT_AVAILABLE = False
BYTETRACK_AVAILABLE = False

# Add OC_SORT to path if available
ocsort_path = Path(__file__).parent.parent / "OC_SORT"
if ocsort_path.exists():
    sys.path.append(str(ocsort_path))

    try:
        from trackers.ocsort_tracker.ocsort import OCSort
        OCSORT_AVAILABLE = True
        print("✅ External OC-SORT available")
    except ImportError:
        print("⚠️ External OC-SORT not available")

    try:
        from trackers.byte_tracker.byte_tracker import BYTETracker
        BYTETRACK_AVAILABLE = True
        print("✅ External ByteTrack available")
    except ImportError:
        print("⚠️ External ByteTrack not available")

# Ensure trackers are always available using simplified versions
if not OCSORT_AVAILABLE and SIMPLE_TRACKERS_AVAILABLE:
    OCSORT_AVAILABLE = True
    print("✅ Using simplified OC-SORT")

if not BYTETRACK_AVAILABLE and SIMPLE_TRACKERS_AVAILABLE:
    BYTETRACK_AVAILABLE = True
    print("✅ Using simplified ByteTrack")

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

        # Player management with fixed ID pool
        self.MAX_PLAYERS = 12  # Maximum number of unique player IDs
        self.next_id = 1
        self.players = {}  # Active players: {id: player_info}
        self.player_gallery = {}  # Long-term player memory: {id: consolidated_features}
        self.available_ids = list(range(1, self.MAX_PLAYERS + 1))  # Pool of available IDs
        self.frame_count = 0
        
        # Feature extraction for re-identification
        self.feature_extractor = self._initialize_feature_extractor()
        
    def _initialize_tracker(self, **kwargs):
        """Initialize the selected tracker"""
        if self.tracker_type == "ocsort" and OCSORT_AVAILABLE:
            try:
                # Try external OC-SORT first, then simplified version
                if 'OCSort' in globals():
                    return OCSort(
                        det_thresh=kwargs.get('det_thresh', 0.5),
                        max_age=kwargs.get('max_age', 30),
                        min_hits=kwargs.get('min_hits', 3),
                        iou_threshold=kwargs.get('iou_threshold', 0.3)
                    )
                else:
                    # Use simplified OC-SORT
                    return SimpleOCSORT(
                        det_thresh=kwargs.get('det_thresh', 0.5),
                        max_age=kwargs.get('max_age', 30),
                        min_hits=kwargs.get('min_hits', 3),
                        iou_threshold=kwargs.get('iou_threshold', 0.3)
                    )
            except Exception as e:
                print(f"⚠️ Failed to initialize OC-SORT: {e}")
                print("🔄 Falling back to simplified OC-SORT")
                try:
                    return SimpleOCSORT(
                        det_thresh=kwargs.get('det_thresh', 0.5),
                        max_age=kwargs.get('max_age', 30),
                        min_hits=kwargs.get('min_hits', 3),
                        iou_threshold=kwargs.get('iou_threshold', 0.3)
                    )
                except:
                    print("🔄 Falling back to custom tracker")
                    self.tracker_type = "custom"
                    return None

        elif self.tracker_type == "bytetrack" and BYTETRACK_AVAILABLE:
            try:
                # Try external ByteTrack first, then simplified version
                if 'BYTETracker' in globals():
                    # Create a simple args object for external ByteTracker
                    class Args:
                        def __init__(self):
                            self.track_thresh = kwargs.get('track_thresh', 0.5)
                            self.track_buffer = kwargs.get('track_buffer', 30)
                            self.match_thresh = kwargs.get('match_thresh', 0.8)
                            self.mot20 = False

                    args = Args()
                    return BYTETracker(args, frame_rate=kwargs.get('frame_rate', 30))
                else:
                    # Use simplified ByteTrack
                    return SimpleBYTETracker(
                        track_thresh=kwargs.get('track_thresh', 0.5),
                        track_buffer=kwargs.get('track_buffer', 30),
                        match_thresh=kwargs.get('match_thresh', 0.8)
                    )
            except Exception as e:
                print(f"⚠️ Failed to initialize ByteTrack: {e}")
                print("🔄 Falling back to simplified ByteTrack")
                try:
                    return SimpleBYTETracker(
                        track_thresh=kwargs.get('track_thresh', 0.5),
                        track_buffer=kwargs.get('track_buffer', 30),
                        match_thresh=kwargs.get('match_thresh', 0.8)
                    )
                except:
                    print("🔄 Falling back to custom tracker")
                    self.tracker_type = "custom"
                    return None
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
        """Fixed ID pool tracking with maximum 12 persistent player IDs"""
        # Update time since last seen for all active players
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

        # Match detections with existing players using comprehensive matching
        for detection in sorted_detections:
            x1, y1, x2, y2, conf = detection
            bbox = [x1, y1, x2, y2]

            # Extract features
            features = self.extract_features(frame, bbox)
            if features is None:
                continue

            best_match_id = None
            best_score = 0

            # Stage 1: Match with active players (recently seen)
            for player_id, player_info in self.players.items():
                if player_info.get('matched_this_frame', False):
                    continue

                # Calculate comprehensive similarity
                similarity_score = self._calculate_comprehensive_similarity(
                    bbox, features, player_info, conf)

                if similarity_score > best_score and similarity_score > 0.3:
                    best_match_id = player_id
                    best_score = similarity_score

            # Stage 2: If no active match, check player gallery (long-term memory)
            if best_match_id is None:
                gallery_match_id, gallery_score = self._match_with_gallery(features, bbox)

                if gallery_match_id is not None and gallery_score > 0.6:
                    # Reactivate player from gallery
                    best_match_id = gallery_match_id
                    best_score = gallery_score

                    # Move from gallery back to active players
                    self.players[gallery_match_id] = {
                        'bbox': bbox,
                        'features': self.player_gallery[gallery_match_id]['features'],
                        'time_since_seen': 0,
                        'matched_this_frame': True,
                        'confidence': conf,
                        'last_position': bbox,
                        'velocity': [0, 0],
                        'reactivation_count': self.player_gallery[gallery_match_id].get('reactivation_count', 0) + 1,
                        'total_appearances': self.player_gallery[gallery_match_id].get('total_appearances', 1) + 1
                    }

                    print(f"🔄 Reactivated player ID {gallery_match_id} (appearances: {self.players[gallery_match_id]['total_appearances']})")

            # Update matched player or store as unmatched
            if best_match_id is not None:
                self._update_matched_player(best_match_id, bbox, features, conf)
                matched_tracks.append([x1, y1, x2, y2, best_match_id, conf])
            else:
                unmatched_detections.append((bbox, features, conf))

        # Handle unmatched detections - assign new IDs only if we have available slots
        for bbox, features, conf in unmatched_detections:
            if conf > 0.5:  # Higher threshold for new players
                assigned_id = self._assign_new_player_id(bbox, features, conf)
                if assigned_id is not None:
                    x1, y1, x2, y2 = bbox
                    matched_tracks.append([x1, y1, x2, y2, assigned_id, conf])
                    print(f"🆕 Assigned new player ID {assigned_id} (total active: {len(self.players)})")

        # Move inactive players to gallery (long-term memory)
        self._manage_player_lifecycle()

        return matched_tracks

    def _calculate_comprehensive_similarity(self, bbox, features, player_info, conf):
        """Calculate comprehensive similarity score for matching"""
        time_since_seen = player_info['time_since_seen']

        # Feature similarity (primary)
        feature_sim = self._calculate_similarity(features, player_info['features'])

        # Spatial continuity (important for recent tracks)
        spatial_score = 0
        if time_since_seen <= 10:
            spatial_iou = self._calculate_iou(bbox, player_info['bbox'])
            motion_consistency = self._calculate_motion_consistency(bbox, player_info)
            spatial_score = spatial_iou * 0.6 + motion_consistency * 0.4

        # Time decay factor
        time_factor = max(0.2, 1.0 - (time_since_seen / 100.0))

        # Confidence boost for high-confidence detections
        conf_boost = min(conf, 0.9) * 0.1

        # Combine scores based on track age
        if time_since_seen <= 5:
            # Recent tracks: prioritize spatial continuity
            combined_score = (spatial_score * 0.6 + feature_sim * 0.4) * time_factor + conf_boost
        elif time_since_seen <= 20:
            # Medium age: balance spatial and features
            combined_score = (spatial_score * 0.4 + feature_sim * 0.6) * time_factor + conf_boost
        else:
            # Old tracks: rely on features
            combined_score = feature_sim * time_factor + conf_boost

        return combined_score

    def _match_with_gallery(self, features, bbox):
        """Match detection with players in long-term gallery"""
        best_match_id = None
        best_score = 0

        for player_id, gallery_info in self.player_gallery.items():
            # Skip if player is currently active
            if player_id in self.players:
                continue

            # Calculate feature similarity with gallery features
            feature_sim = self._calculate_similarity(features, gallery_info['features'])

            # Boost score for players with more appearances (more reliable)
            appearance_boost = min(gallery_info.get('total_appearances', 1) / 10.0, 0.2)

            # Time decay since last seen
            frames_since_gallery = self.frame_count - gallery_info.get('last_seen_frame', 0)
            time_decay = max(0.3, 1.0 - (frames_since_gallery / 1000.0))

            final_score = (feature_sim + appearance_boost) * time_decay

            if final_score > best_score:
                best_match_id = player_id
                best_score = final_score

        return best_match_id, best_score

    def _assign_new_player_id(self, bbox, features, conf):
        """Assign a new player ID from available pool"""
        if not self.available_ids:
            # No available IDs - try to reclaim from gallery
            reclaimed_id = self._reclaim_id_from_gallery()
            if reclaimed_id is not None:
                self.available_ids.append(reclaimed_id)
            else:
                print(f"⚠️ Maximum players ({self.MAX_PLAYERS}) reached - cannot assign new ID")
                return None

        # Get the next available ID
        new_id = self.available_ids.pop(0)

        # Create new player
        self.players[new_id] = {
            'bbox': bbox,
            'features': features,
            'time_since_seen': 0,
            'matched_this_frame': True,
            'confidence': conf,
            'last_position': bbox,
            'velocity': [0, 0],
            'creation_frame': self.frame_count,
            'total_appearances': 1
        }

        return new_id

    def _reclaim_id_from_gallery(self):
        """Reclaim an ID from the gallery (remove least reliable player)"""
        if not self.player_gallery:
            return None

        # Find player with lowest reliability score
        worst_player_id = None
        worst_score = float('inf')

        for player_id, gallery_info in self.player_gallery.items():
            # Calculate reliability score (lower is worse)
            appearances = gallery_info.get('total_appearances', 1)
            frames_since_seen = self.frame_count - gallery_info.get('last_seen_frame', 0)

            reliability_score = appearances - (frames_since_seen / 100.0)

            if reliability_score < worst_score:
                worst_score = reliability_score
                worst_player_id = player_id

        if worst_player_id is not None:
            del self.player_gallery[worst_player_id]
            print(f"🗑️ Reclaimed ID {worst_player_id} from gallery")
            return worst_player_id

        return None
        
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

    def _predict_next_position(self, player_info):
        """Predict next position based on velocity and position history"""
        if 'velocity' not in player_info or 'last_position' not in player_info:
            return player_info['bbox']

        last_bbox = player_info['last_position']
        velocity = player_info['velocity']

        # Apply velocity with damping for multiple frames
        frames_ahead = player_info['time_since_seen']
        damping = max(0.5, 1.0 - frames_ahead * 0.1)  # Reduce prediction confidence over time

        predicted_x = last_bbox[0] + velocity[0] * frames_ahead * damping
        predicted_y = last_bbox[1] + velocity[1] * frames_ahead * damping

        # Keep same size
        width = last_bbox[2] - last_bbox[0]
        height = last_bbox[3] - last_bbox[1]

        return [predicted_x, predicted_y, predicted_x + width, predicted_y + height]

    def _update_matched_player(self, player_id, bbox, features, conf):
        """Update matched player with new detection"""
        player_info = self.players[player_id]

        # Calculate velocity
        velocity = self._calculate_velocity(bbox, player_info.get('last_position', bbox))

        # Conservative feature update to maintain identity
        old_features = player_info['features']
        time_weight = max(0.1, 1.0 - (player_info['time_since_seen'] / 30.0))
        conf_weight = min(conf, 0.9)
        update_rate = time_weight * conf_weight * 0.2  # Very conservative

        updated_features = self._update_features(old_features, features, alpha=1-update_rate)

        # Update player info
        self.players[player_id].update({
            'bbox': bbox,
            'features': updated_features,
            'time_since_seen': 0,
            'matched_this_frame': True,
            'confidence': conf,
            'last_position': player_info['bbox'],  # Store previous position
            'velocity': velocity,
            'total_appearances': player_info.get('total_appearances', 1) + 1
        })

    def _manage_player_lifecycle(self):
        """Manage player lifecycle - move inactive players to gallery"""
        players_to_gallery = []

        for player_id, player_info in self.players.items():
            # Move to gallery if not seen for a while but don't delete
            if player_info['time_since_seen'] > 60:  # 60 frames without detection
                players_to_gallery.append(player_id)

        # Move players to gallery
        for player_id in players_to_gallery:
            player_info = self.players[player_id]

            # Store in gallery with consolidated features
            self.player_gallery[player_id] = {
                'features': player_info['features'],
                'last_seen_frame': self.frame_count - player_info['time_since_seen'],
                'total_appearances': player_info.get('total_appearances', 1),
                'confidence_history': player_info.get('confidence', 0.5),
                'last_bbox': player_info['bbox']
            }

            # Remove from active players and return ID to available pool
            del self.players[player_id]
            if player_id not in self.available_ids:
                self.available_ids.append(player_id)
                self.available_ids.sort()  # Keep IDs sorted

            print(f"📚 Moved player ID {player_id} to gallery (appearances: {self.player_gallery[player_id]['total_appearances']})")

        # Limit gallery size to prevent memory issues
        if len(self.player_gallery) > self.MAX_PLAYERS * 2:
            # Remove oldest entries
            sorted_gallery = sorted(self.player_gallery.items(),
                                  key=lambda x: x[1]['last_seen_frame'])

            for player_id, _ in sorted_gallery[:len(self.player_gallery) - self.MAX_PLAYERS * 2]:
                del self.player_gallery[player_id]
                print(f"🗑️ Removed old player ID {player_id} from gallery")
        
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
        total_active = len(self.players)
        gallery_players = len(self.player_gallery)
        available_ids = len(self.available_ids)

        return {
            'frame_count': self.frame_count,
            'active_players': active_players,
            'total_players_detected': total_active,
            'gallery_players': gallery_players,
            'available_ids': available_ids,
            'max_players': self.MAX_PLAYERS,
            'tracker_type': self.tracker_type,
            'model_type': self.yolo_model_name,
            'id_pool_usage': f"{self.MAX_PLAYERS - available_ids}/{self.MAX_PLAYERS}"
        }
        
    def reset(self):
        """Reset tracker state"""
        self.players = {}
        self.player_gallery = {}
        self.available_ids = list(range(1, self.MAX_PLAYERS + 1))
        self.next_id = 1
        self.frame_count = 0
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        print(f"🔄 Tracker reset - {self.MAX_PLAYERS} player IDs available")

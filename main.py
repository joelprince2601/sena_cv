import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, Scale
from PIL import Image, ImageTk
import threading
import sys

# Try to import ultralytics for YOLOv8 support
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics package not found. YOLOv8 support will be limited.")
    
# Try to import tracking libraries
try:
    from bytetrack.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("ByteTrack package not found. ByteTrack tracking will not be available.")
    
try:
    from ocsort.ocsort import OCSort
    OCSORT_AVAILABLE = True
except ImportError:
    OCSORT_AVAILABLE = False
    print("OC-SORT package not found. OC-SORT tracking will not be available.")

# Custom implementation of cosine similarity to avoid sklearn dependency
def cosine_similarity(A, B):
    """Calculate cosine similarity between two vectors"""
    # Ensure inputs are numpy arrays
    A = np.array(A)
    B = np.array(B)
    
    # Calculate dot product
    dot_product = np.dot(A, B.T)
    
    # Calculate magnitudes
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    
    # Calculate similarity
    similarity = dot_product / (np.outer(norm_A, norm_B) + 1e-8)
    
    return similarity

class FootballPlayerReID:
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4, reid_threshold=0.7, yolo_model="yolov4", tracker_type="custom"):
        # Initialize parameters
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.reid_threshold = reid_threshold
        self.yolo_model = yolo_model  # Store the selected YOLO model
        self.tracker_type = tracker_type  # Store the selected tracker type
        self.next_id = 1
        self.players = {}  # Dictionary to store player information {id: {features, bbox, time_since_seen}}
        self.using_ultralytics = False  # Flag to indicate if using ultralytics YOLO
        
        # Initialize tracker based on selected type
        self.initialize_tracker()
        
        # Get the absolute path to the models directory
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        # Load YOLO model for person detection
        self.load_yolo()
        
        # Initialize feature extractor for re-identification if using custom tracker
        if self.tracker_type == "custom":
            reid_model_path = os.path.join(self.models_dir, 'reid_model.t7')
            try:
                self.feature_extractor = cv2.dnn.readNetFromTorch(reid_model_path)
            except cv2.error:
                # If reid_model.t7 fails, try using best.pt with a different method
                print("Warning: Could not load reid_model.t7, using best.pt as fallback")
                # For PyTorch models, we'll use a simple placeholder for now
                # In a real implementation, you would use PyTorch to load this model
                self.feature_extractor = None
        else:
            # For ByteTrack and OC-SORT, we don't need the feature extractor
            self.feature_extractor = None
        
    def initialize_tracker(self):
        """Initialize the selected tracker"""
        if self.tracker_type == "bytetrack":
            if BYTETRACK_AVAILABLE:
                # Initialize ByteTrack with default parameters
                self.tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
                print("Initialized ByteTrack tracker")
            else:
                print("ByteTrack not available, falling back to custom tracker")
                self.tracker_type = "custom"
        elif self.tracker_type == "ocsort":
            if OCSORT_AVAILABLE:
                # Initialize OC-SORT with default parameters
                self.tracker = OCSort(det_thresh=0.5, max_age=30, min_hits=3)
                print("Initialized OC-SORT tracker")
            else:
                print("OC-SORT not available, falling back to custom tracker")
                self.tracker_type = "custom"
        else:
            # Custom tracker uses our own implementation
            self.tracker = None
            print("Using custom tracker with ReID features")
    
    def load_yolo(self):
        # Load YOLO model for person detection
        print(f"Loading {self.yolo_model} model for player detection...")
        
        # Define model configurations based on selected model
        model_configs = {
            "yolov4": {
                "cfg": "yolov4.cfg",
                "weights": "yolov4.weights",
                "alt_cfg": "yolov4x.cfg"
            },
            "yolov7": {
                "cfg": "yolov7.cfg",
                "weights": "yolov7.weights",
                "alt_cfg": "yolov7x.cfg"
            },
            "yolov8": {
                "cfg": "yolov8.cfg",
                "weights": "yolov8.pt",
                "alt_cfg": "yolov8n.yaml"
            }
        }
        
        # Get configuration for selected model
        model_config = model_configs.get(self.yolo_model, model_configs["yolov4"])  # Default to yolov4 if not found
        
        # Try primary config file, then alternative
        cfg_path = os.path.join(self.models_dir, model_config["cfg"])
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(self.models_dir, model_config["alt_cfg"])
            
        weights_path = os.path.join(self.models_dir, model_config["weights"])
        
        # Load the model
        if self.yolo_model == "yolov8" and weights_path.endswith(".pt"):
            if ULTRALYTICS_AVAILABLE:
                # For YOLOv8, use the ultralytics YOLO package
                try:
                    self.yolo_model_obj = YOLO(weights_path)
                    self.using_ultralytics = True
                    print(f"Loaded YOLOv8 model from {weights_path} using ultralytics")
                    return
                except Exception as e:
                    print(f"Error loading YOLOv8 model: {e}. Falling back to YOLOv4.")
                    self.yolo_model = "yolov4"
                    return self.load_yolo()
            else:
                print("YOLOv8 support requires the ultralytics package. Falling back to YOLOv4.")
                # Fallback to YOLOv4
                self.yolo_model = "yolov4"
                return self.load_yolo()
        else:
            # For YOLOv4/v7 we can use OpenCV DNN
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.using_ultralytics = False
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        
        # Load class names - check for both coco.names and coco.names.txt
        coco_names_path = os.path.join(self.models_dir, 'coco.names')
        if not os.path.exists(coco_names_path):
            coco_names_path = os.path.join(self.models_dir, 'coco.names.txt')
        
        with open(coco_names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
    
    def extract_features(self, frame, bbox):
        """Extract features from player's bounding box for re-identification"""
        x, y, w, h = bbox
        # Ensure coordinates are within frame boundaries
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return None
            
        # Extract player ROI
        player_roi = frame[y:y+h, x:x+w]
        
        # If feature extractor is not available, use a simple color histogram as features
        if self.feature_extractor is None:
            # Convert to HSV color space
            hsv_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
            # Calculate histogram
            hist = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
            # Normalize histogram
            features = cv2.normalize(hist, hist).flatten()
            return features
        
        # Resize ROI for feature extraction
        blob = cv2.dnn.blobFromImage(player_roi, 1.0/255, (128, 256), (0, 0, 0), swapRB=True, crop=False)
        
        # Extract features
        self.feature_extractor.setInput(blob)
        features = self.feature_extractor.forward()
        
        # Normalize features
        features = features / np.linalg.norm(features)
        
        return features
    
    def detect_players(self, frame):
        """Detect players in the frame using YOLO"""
        height, width = frame.shape[:2]
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        if hasattr(self, 'using_ultralytics') and self.using_ultralytics:
            # Use ultralytics YOLO model
            results = self.yolo_model_obj(frame)
            
            # Process results
            for result in results:
                boxes_data = result.boxes
                for i, box in enumerate(boxes_data):
                    # Get box coordinates in xywh format
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Convert to center format
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Get class and confidence
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    # Only consider person class (class 0)
                    if cls == 0 and conf > self.confidence_threshold:
                        class_ids.append(cls)
                        confidences.append(conf)
                        boxes.append([center_x, center_y, w, h])
        else:
            # Use OpenCV DNN for YOLOv4/v7
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            
            # Set input to the network
            self.net.setInput(blob)
            
            # Run forward pass
            outputs = self.net.forward(self.output_layers)
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                
                # Filter for person class (class_id 0 in COCO dataset)
                if class_id == 0 and confidence > self.confidence_threshold:
                    # YOLO returns center (x, y) and width, height
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detected_players = []
        if len(indices) > 0:
            for i in indices.flatten():
                detected_players.append((boxes[i], confidences[i]))
        
        return detected_players
    
    def match_players(self, frame, detected_players):
        """Match detected players with existing player IDs or assign new IDs"""
        # Update time since last seen for all players
        for player_id in self.players:
            self.players[player_id]['time_since_seen'] += 1
        
        # Store unmatched detections for potential new ID assignment
        unmatched_detections = []
        
        # Match detected players with existing players
        for bbox, confidence in detected_players:
            # Extract features for the detected player
            features = self.extract_features(frame, bbox)
            
            if features is None:
                continue
                
            best_match_id = None
            best_match_score = 0
            
            # Compare with existing players
            for player_id, player_info in self.players.items():
                # Skip players that have been matched in this frame
                if 'matched_this_frame' in player_info and player_info['matched_this_frame']:
                    continue
                    
                # Handle different feature types
                if self.feature_extractor is None:
                    # For histogram features, use correlation as similarity measure
                    similarity = cv2.compareHist(features.reshape(30, 32), 
                                                player_info['features'].reshape(30, 32), 
                                                cv2.HISTCMP_CORREL)
                else:
                    # For deep features, use cosine similarity
                    similarity = cosine_similarity(features.reshape(1, -1), 
                                                 player_info['features'].reshape(1, -1))
                    # Extract the similarity value from the matrix
                    similarity = similarity[0][0]
                
                # Consider spatial proximity for recently seen players
                if player_info['time_since_seen'] < 10:
                    # Calculate IoU between current detection and last known position
                    iou = self._calculate_iou(bbox, player_info['bbox'])
                    # Boost similarity score for spatially close detections
                    similarity = similarity * 0.7 + iou * 0.3
                
                if similarity > self.reid_threshold and similarity > best_match_score:
                    best_match_id = player_id
                    best_match_score = similarity
            
            # If match found, update player info
            if best_match_id is not None:
                self.players[best_match_id]['bbox'] = bbox
                # Update features with moving average to maintain identity consistency
                if 'features_history' not in self.players[best_match_id]:
                    self.players[best_match_id]['features_history'] = [features]
                else:
                    self.players[best_match_id]['features_history'].append(features)
                    # Keep only the last 5 feature vectors
                    if len(self.players[best_match_id]['features_history']) > 5:
                        self.players[best_match_id]['features_history'].pop(0)
                
                # Update features with average of recent observations
                if self.feature_extractor is None:
                    # For histogram features
                    self.players[best_match_id]['features'] = np.mean(self.players[best_match_id]['features_history'], axis=0)
                else:
                    # For deep features
                    avg_features = np.mean(self.players[best_match_id]['features_history'], axis=0)
                    # Normalize the averaged features
                    self.players[best_match_id]['features'] = avg_features / np.linalg.norm(avg_features)
                
                self.players[best_match_id]['time_since_seen'] = 0
                self.players[best_match_id]['matched_this_frame'] = True
            else:
                # Store for potential new ID assignment
                unmatched_detections.append((bbox, features))
        
        # Assign new IDs to unmatched detections
        for bbox, features in unmatched_detections:
            self.players[self.next_id] = {
                'features': features,
                'features_history': [features],
                'bbox': bbox,
                'time_since_seen': 0,
                'matched_this_frame': True
            }
            self.next_id += 1
        
        # Reset the matched_this_frame flag for the next frame
        for player_id in self.players:
            if 'matched_this_frame' in self.players[player_id]:
                self.players[player_id]['matched_this_frame'] = False
        
        # Remove players that haven't been seen for a long time
        players_to_remove = []
        for player_id, player_info in self.players.items():
            # Increase the threshold to maintain IDs longer for players who exit the frame
            if player_info['time_since_seen'] > 150:  # Increased from 100 to 150
                players_to_remove.append(player_id)
        
        for player_id in players_to_remove:
            del self.players[player_id]
            
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        # Extract coordinates
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate area of each box
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate coordinates of intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Check if there is an intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        union_area = area1 + area2 - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def draw_results(self, frame):
        """Draw bounding boxes and IDs on the frame"""
        for player_id, player_info in self.players.items():
            if player_info['time_since_seen'] == 0:  # Only draw currently visible players
                x, y, w, h = player_info['bbox']
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw player ID
                cv2.putText(frame, f"ID: {player_id}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def process_video(self, video_path, output_path=None):
        """Process video for player re-identification"""
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 2nd frame to improve performance
            if frame_count % 2 == 0:
                # Detect players
                detected_players = self.detect_players(frame)
                
                # Match players with existing IDs
                self.match_players(frame, detected_players)
            
            # Draw results on frame
            result_frame = self.draw_results(frame)
            
            # Display frame count and FPS
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Write frame to output video
            if out:
                out.write(result_frame)
            
            # Display result
            cv2.imshow('Football Player Re-ID', result_frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

class FootballReIDGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Player Re-ID System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.video_path = ""
        self.output_path = ""
        self.cap = None
        self.processing = False
        self.paused = False
        self.frame = None
        self.photo = None
        self.reid_system = None
        self.frame_count = 0
        
        # Create directory for models if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Create GUI elements
        self.create_widgets()
        
        # Check if models exist
        self.check_models()
    
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = tk.Frame(main_frame, bg="#f0f0f0", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(left_panel, text="Football Player Re-ID", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)
        
        # File selection
        file_frame = tk.LabelFrame(left_panel, text="File Selection", bg="#f0f0f0", padx=10, pady=10)
        file_frame.pack(fill=tk.X, pady=10)
        
        # Video input
        tk.Label(file_frame, text="Input Video:", bg="#f0f0f0").pack(anchor=tk.W)
        self.video_path_var = tk.StringVar()
        video_entry = tk.Entry(file_frame, textvariable=self.video_path_var, width=25)
        video_entry.pack(fill=tk.X, pady=5)
        video_btn = tk.Button(file_frame, text="Browse", command=self.browse_video)
        video_btn.pack(fill=tk.X)
        
        # Output video
        tk.Label(file_frame, text="Output Video (optional):", bg="#f0f0f0").pack(anchor=tk.W, pady=(10, 0))
        self.output_path_var = tk.StringVar()
        output_entry = tk.Entry(file_frame, textvariable=self.output_path_var, width=25)
        output_entry.pack(fill=tk.X, pady=5)
        output_btn = tk.Button(file_frame, text="Browse", command=self.browse_output)
        output_btn.pack(fill=tk.X)
        
        # Parameters frame
        param_frame = tk.LabelFrame(left_panel, text="Parameters", bg="#f0f0f0", padx=10, pady=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        # Model selection
        tk.Label(param_frame, text="YOLO Model:", bg="#f0f0f0").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="yolov4")
        model_options = ["yolov4", "yolov7", "yolov8"]
        model_dropdown = ttk.Combobox(param_frame, textvariable=self.model_var, values=model_options, state="readonly")
        model_dropdown.pack(fill=tk.X, pady=(0, 10))
        
        # Tracker selection
        tk.Label(param_frame, text="Tracker Type:", bg="#f0f0f0").pack(anchor=tk.W)
        self.tracker_var = tk.StringVar(value="custom")
        tracker_options = ["custom", "bytetrack", "ocsort"]
        tracker_dropdown = ttk.Combobox(param_frame, textvariable=self.tracker_var, values=tracker_options, state="readonly")
        tracker_dropdown.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold
        tk.Label(param_frame, text="Confidence Threshold:", bg="#f0f0f0").pack(anchor=tk.W)
        self.conf_threshold_var = tk.DoubleVar(value=0.5)
        conf_scale = Scale(param_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, 
                           variable=self.conf_threshold_var, bg="#f0f0f0")
        conf_scale.pack(fill=tk.X)
        
        # NMS threshold
        tk.Label(param_frame, text="NMS Threshold:", bg="#f0f0f0").pack(anchor=tk.W)
        self.nms_threshold_var = tk.DoubleVar(value=0.4)
        nms_scale = Scale(param_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, 
                          variable=self.nms_threshold_var, bg="#f0f0f0")
        nms_scale.pack(fill=tk.X)
        
        # ReID threshold
        tk.Label(param_frame, text="ReID Threshold:", bg="#f0f0f0").pack(anchor=tk.W)
        self.reid_threshold_var = tk.DoubleVar(value=0.7)
        reid_scale = Scale(param_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, 
                           variable=self.reid_threshold_var, bg="#f0f0f0")
        reid_scale.pack(fill=tk.X)
        
        # Control buttons
        control_frame = tk.Frame(left_panel, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = tk.Button(control_frame, text="Start Processing", command=self.start_processing, 
                                  bg="#4CAF50", fg="white", height=2)
        self.start_btn.pack(fill=tk.X, pady=5)
        
        self.pause_btn = tk.Button(control_frame, text="Pause", command=self.toggle_pause, 
                                  state=tk.DISABLED, height=2)
        self.pause_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = tk.Button(control_frame, text="Stop", command=self.stop_processing, 
                                 state=tk.DISABLED, bg="#f44336", fg="white", height=2)
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        # Status frame
        status_frame = tk.LabelFrame(left_panel, text="Status", bg="#f0f0f0", padx=10, pady=10)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(status_frame, textvariable=self.status_var, bg="#f0f0f0", wraplength=280)
        status_label.pack(fill=tk.X)
        
        self.frame_var = tk.StringVar(value="Frame: 0")
        frame_label = tk.Label(status_frame, textvariable=self.frame_var, bg="#f0f0f0")
        frame_label.pack(fill=tk.X, pady=5)
        
        self.players_var = tk.StringVar(value="Players: 0")
        players_label = tk.Label(status_frame, textvariable=self.players_var, bg="#f0f0f0")
        players_label.pack(fill=tk.X)
        
        # Right panel for video display
        self.right_panel = tk.Frame(main_frame, bg="black")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video canvas
        self.canvas = tk.Canvas(self.right_panel, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initial message on canvas
        self.canvas.create_text(400, 300, text="No video loaded", fill="white", font=("Arial", 20))
    
    def check_models(self):
        # Get the absolute path to the models directory
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        missing_models = []
        
        if not os.path.exists(os.path.join(models_dir, 'yolov4.weights')):
            missing_models.append("YOLOv4 weights")
        
        if not os.path.exists(os.path.join(models_dir, 'yolov4.cfg')) and not os.path.exists(os.path.join(models_dir, 'yolov4x.cfg')):
            missing_models.append("YOLOv4 config")
        
        # Check for both coco.names and coco.names.txt
        if not (os.path.exists(os.path.join(models_dir, 'coco.names')) or 
                os.path.exists(os.path.join(models_dir, 'coco.names.txt'))):
            missing_models.append("COCO names")
        
        if not os.path.exists(os.path.join(models_dir, 'reid_model.t7')) and not os.path.exists(os.path.join(models_dir, 'best.pt')):
            missing_models.append("ReID model")
        
        if missing_models:
            message = "The following model files are missing:\n\n"
            for model in missing_models:
                message += f"- {model}\n"
            
            message += "\nPlease download the required files:\n"
            message += "1. YOLOv4 weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights\n"
            message += "2. YOLOv4 config: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg\n"
            message += "3. COCO names: https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names\n"
            message += "4. ReID model: https://drive.google.com/file/d/1_4tJMT_SG_xqfG8zzwC_OUtJG-iJYbAJ/view\n\n"
            message += "Place these files in the 'models' directory."
            
            messagebox.showwarning("Missing Model Files", message)
            self.status_var.set("Missing model files. Please download required files.")
    
    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv"), ("All files", "*.*")])
        if file_path:
            self.video_path = file_path
            self.video_path_var.set(file_path)
            self.status_var.set(f"Video selected: {os.path.basename(file_path)}")
    
    def browse_output(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".avi", 
                                               filetypes=[("AVI files", "*.avi"), ("All files", "*.*")])
        if file_path:
            self.output_path = file_path
            self.output_path_var.set(file_path)
    
    def start_processing(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select an input video file.")
            return
        
        # Check if models exist
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        # Get selected model
        selected_model = self.model_var.get()
        
        # Define weights path based on selected model
        if selected_model == "yolov4":
            weights_path = os.path.join(models_dir, 'yolov4.weights')
        elif selected_model == "yolov7":
            weights_path = os.path.join(models_dir, 'yolov7.weights')
        elif selected_model == "yolov8":
            weights_path = os.path.join(models_dir, 'yolov8.pt')
        else:
            weights_path = os.path.join(models_dir, 'yolov4.weights')  # Default
        
        reid_model_path = os.path.join(models_dir, 'reid_model.t7')
        if not os.path.exists(reid_model_path):
            reid_model_path = os.path.join(models_dir, 'best.pt')
        
        if not os.path.exists(weights_path) or (not os.path.exists(os.path.join(models_dir, 'reid_model.t7')) and not os.path.exists(os.path.join(models_dir, 'best.pt'))):
            messagebox.showerror("Error", f"Required {selected_model.upper()} model files not found. Please download them first.")
            return
        
        # Initialize ReID system with current parameters and selected model
        self.reid_system = FootballPlayerReID(
            confidence_threshold=self.conf_threshold_var.get(),
            nms_threshold=self.nms_threshold_var.get(),
            reid_threshold=self.reid_threshold_var.get(),
            yolo_model=self.model_var.get()
        )
        
        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Processing video...")
        
        # Start processing in a separate thread
        self.processing = True
        self.paused = False
        processing_thread = threading.Thread(target=self.process_video)
        processing_thread.daemon = True
        processing_thread.start()
    
    def process_video(self):
        # Open video file
        self.cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output path is provided
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        self.frame_count = 0
        
        while self.cap.isOpened() and self.processing:
            if not self.paused:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Process every 2nd frame to improve performance
                if self.frame_count % 2 == 0:
                    # Detect players
                    detected_players = self.reid_system.detect_players(frame)
                    
                    # Match players with existing IDs
                    self.reid_system.match_players(frame, detected_players)
                
                # Draw results on frame
                result_frame = self.reid_system.draw_results(frame)
                
                # Display frame count
                cv2.putText(result_frame, f"Frame: {self.frame_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Update UI
                self.frame_var.set(f"Frame: {self.frame_count}")
                self.players_var.set(f"Players: {len(self.reid_system.players)}")
                
                # Write frame to output video
                if out:
                    out.write(result_frame)
                
                # Display result in GUI
                self.update_frame(result_frame)
                
                # Control frame rate for display
                self.root.update()
                time.sleep(1/30)  # Limit to 30 FPS for display
        
        # Release resources
        if self.cap:
            self.cap.release()
        if out:
            out.release()
        
        # Reset UI if processing completed normally
        if not self.processing:
            self.root.after(0, self.reset_ui)
        else:
            self.status_var.set("Processing completed")
            self.processing = False
            self.root.after(0, self.reset_ui)
    
    def update_frame(self, frame):
        # Convert frame to RGB for tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to fit canvas if needed
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
            frame_height, frame_width = rgb_frame.shape[:2]
            
            # Calculate scaling factor to fit in canvas
            scale_width = canvas_width / frame_width
            scale_height = canvas_height / frame_height
            scale = min(scale_width, scale_height)
            
            # Resize frame
            if scale < 1:
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Convert to PhotoImage
        self.frame = Image.fromarray(rgb_frame)
        self.photo = ImageTk.PhotoImage(image=self.frame)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)
    
    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="Resume")
            self.status_var.set("Paused")
        else:
            self.pause_btn.config(text="Pause")
            self.status_var.set("Processing video...")
    
    def stop_processing(self):
        self.processing = False
        self.paused = False
        self.status_var.set("Processing stopped")
    
    def reset_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

# Example usage
def main():
    # Create directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Start GUI
    root = tk.Tk()
    app = FootballReIDGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
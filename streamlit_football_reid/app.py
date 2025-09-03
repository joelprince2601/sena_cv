"""
Streamlit Football Player Re-ID Application
Modern web interface for football player tracking and re-identification
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tracking_system import EnhancedFootballTracker
from config import YOLO_MODELS, TRACKER_CONFIGS, DEFAULT_PARAMS, UI_SETTINGS, VIDEO_SETTINGS
from video_processor import StreamlitVideoProcessor
from utils import ModelManager, validate_video_file, PerformanceMonitor
from keypoint_processor import StreamlitKeypointProcessor

# Page configuration
st.set_page_config(
    page_title=UI_SETTINGS["page_title"],
    page_icon=UI_SETTINGS["page_icon"],
    layout=UI_SETTINGS["layout"],
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = StreamlitVideoProcessor()
    if 'keypoint_processor' not in st.session_state:
        st.session_state.keypoint_processor = StreamlitKeypointProcessor()
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'video_uploaded' not in st.session_state:
        st.session_state.video_uploaded = False
    if 'results_data' not in st.session_state:
        st.session_state.results_data = []
    if 'keypoint_results_data' not in st.session_state:
        st.session_state.keypoint_results_data = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Player Tracking"

def create_sidebar():
    """Create sidebar with controls"""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Model selection
    st.sidebar.markdown("### 🤖 YOLO Model")
    selected_model = st.sidebar.selectbox(
        "Choose YOLO model:",
        options=list(YOLO_MODELS.keys()),
        format_func=lambda x: f"{YOLO_MODELS[x]['name']} - {YOLO_MODELS[x]['description']}",
        index=0
    )
    
    # Tracker selection - all trackers are now always available
    st.sidebar.markdown("### 🎯 Tracker Type")

    selected_tracker = st.sidebar.selectbox(
        "Choose tracking algorithm:",
        options=list(TRACKER_CONFIGS.keys()),
        format_func=lambda x: f"{TRACKER_CONFIGS[x]['name']} - {TRACKER_CONFIGS[x]['description']}",
        index=0
    )
    
    # Parameters
    st.sidebar.markdown("### 📊 Detection Parameters")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_PARAMS["confidence_threshold"],
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    nms_threshold = st.sidebar.slider(
        "NMS Threshold", 
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_PARAMS["nms_threshold"],
        step=0.05,
        help="Non-maximum suppression threshold"
    )
    
    reid_threshold = st.sidebar.slider(
        "Re-ID Threshold",
        min_value=0.1, 
        max_value=1.0,
        value=DEFAULT_PARAMS["reid_threshold"],
        step=0.05,
        help="Re-identification similarity threshold"
    )
    
    # Advanced parameters
    with st.sidebar.expander("🔧 Advanced Parameters"):
        max_age = st.slider("Max Age", 10, 200, DEFAULT_PARAMS["max_age"])
        min_hits = st.slider("Min Hits", 1, 10, DEFAULT_PARAMS["min_hits"])
        
    return {
        'model': selected_model,
        'tracker': selected_tracker,
        'confidence_threshold': confidence_threshold,
        'nms_threshold': nms_threshold,
        'reid_threshold': reid_threshold,
        'max_age': max_age,
        'min_hits': min_hits
    }

def create_main_interface():
    """Create main interface for player tracking"""
    # File upload
    st.markdown("## 📹 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=VIDEO_SETTINGS["supported_formats"],
        help=f"Supported formats: {', '.join(VIDEO_SETTINGS['supported_formats'])}"
    )

    return uploaded_file

def create_keypoint_interface():
    """Create interface for keypoint detection"""
    # File upload
    st.markdown("## 📹 Upload Video for Keypoint Detection")
    uploaded_file = st.file_uploader(
        "Choose a video file for keypoint analysis",
        type=VIDEO_SETTINGS["supported_formats"],
        help=f"Supported formats: {', '.join(VIDEO_SETTINGS['supported_formats'])}",
        key="keypoint_uploader"
    )

    return uploaded_file

def create_keypoint_sidebar():
    """Create sidebar for keypoint detection configuration"""
    st.sidebar.markdown("## ⚙️ Keypoint Configuration")

    # Detection type selection
    st.sidebar.markdown("### 🎯 Detection Type")
    detection_types = st.session_state.keypoint_processor.get_detection_types()

    selected_detection = st.sidebar.selectbox(
        "Choose detection type:",
        options=list(detection_types.keys()),
        format_func=lambda x: f"{detection_types[x]['name']}",
        index=0
    )

    # Show description
    st.sidebar.info(detection_types[selected_detection]['description'])

    # Parameters
    st.sidebar.markdown("### 📊 Detection Parameters")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for keypoint detection"
    )

    # Visualization options
    st.sidebar.markdown("### 🎨 Visualization Options")

    show_connections = st.sidebar.checkbox("Show Connections", value=True)
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_3d = st.sidebar.checkbox("Show 3D Visualization", value=False)

    return {
        'detection_type': selected_detection,
        'confidence_threshold': confidence_threshold,
        'show_connections': show_connections,
        'show_labels': show_labels,
        'show_3d': show_3d
    }

def process_video(video_file, config):
    """Process uploaded video using the enhanced video processor"""
    try:
        # Validate video file
        is_valid, message = validate_video_file(video_file)
        if not is_valid:
            st.error(f"❌ {message}")
            return []

        # Check model availability
        model_manager = st.session_state.model_manager
        model_info = model_manager.get_model_info(config['model'])

        if not model_info['available']:
            with st.spinner(f"📥 Downloading {model_info['name']} model..."):
                model_manager.download_model(config['model'])

        # Process video using the enhanced processor
        st.markdown("## 🎬 Video Processing")

        # Setup processor
        video_processor = st.session_state.video_processor
        video_processor.setup_processor(config)

        # Process the video
        results_data = video_processor.process_uploaded_video(video_file, config)

        # Store results
        st.session_state.results_data = results_data

        return results_data

    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        return []

def process_keypoint_video(video_file, config):
    """Process uploaded video for keypoint detection"""
    try:
        # Validate video file
        is_valid, message = validate_video_file(video_file)
        if not is_valid:
            st.error(f"❌ {message}")
            return []

        # Process video using the keypoint processor
        st.markdown("## 🎬 Keypoint Detection & Tracking")

        # Process the video
        keypoint_processor = st.session_state.keypoint_processor
        results_data = keypoint_processor.process_uploaded_video(
            video_file,
            config['detection_type'],
            config['confidence_threshold'],
            config['show_connections'],
            config['show_labels'],
            config['show_3d']
        )

        # Store results
        st.session_state.keypoint_results_data = results_data

        return results_data

    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        return []

def show_analytics():
    """Show analytics dashboard"""
    if not st.session_state.results_data:
        st.info("📊 Process a video to see analytics")
        return
        
    st.markdown("## 📈 Analytics Dashboard")
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.results_data)
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max Players", df['active_players'].max())
    with col2:
        st.metric("Avg Players", f"{df['active_players'].mean():.1f}")
    with col3:
        st.metric("Total Unique", df['total_players'].max())
    with col4:
        st.metric("Video Duration", f"{df['timestamp'].max():.1f}s")
    
    # Player count over time
    fig_players = px.line(df, x='timestamp', y='active_players',
                         title='Active Players Over Time',
                         labels={'timestamp': 'Time (seconds)', 'active_players': 'Active Players'})
    st.plotly_chart(fig_players, use_container_width=True)
    
    # Player detection histogram
    fig_hist = px.histogram(df, x='active_players', nbins=20,
                           title='Distribution of Active Players',
                           labels={'active_players': 'Number of Active Players', 'count': 'Frequency'})
    st.plotly_chart(fig_hist, use_container_width=True)

def show_keypoint_analytics():
    """Show football pose analytics dashboard"""
    if not st.session_state.keypoint_results_data:
        st.info("📊 Process a video to see football pose analytics")
        return

    st.markdown("## 📈 Football Pose Analytics Dashboard")

    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.keypoint_results_data)

    # Create metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Max Players", df['football_players'].max())
    with col2:
        st.metric("Avg Players", f"{df['football_players'].mean():.1f}")
    with col3:
        st.metric("Avg Pose Quality", f"{df['avg_pose_quality'].mean():.2f}")
    with col4:
        st.metric("Avg Confidence", f"{df['avg_confidence'].mean():.2f}")

    # Players count over time
    fig_players = px.line(df, x='timestamp', y='football_players',
                         title='Football Players Detected Over Time',
                         labels={'timestamp': 'Time (seconds)', 'football_players': 'Number of Players'})
    st.plotly_chart(fig_players, use_container_width=True)

    # Action analysis over time
    if 'action_counts' in df.columns:
        # Extract action data
        action_data = []
        for _, row in df.iterrows():
            action_counts = row.get('action_counts', {})
            for action, count in action_counts.items():
                action_data.append({
                    'timestamp': row['timestamp'],
                    'action': action.title(),
                    'count': count
                })

        if action_data:
            action_df = pd.DataFrame(action_data)
            fig_actions = px.line(action_df, x='timestamp', y='count', color='action',
                                 title='Football Actions Over Time',
                                 labels={'timestamp': 'Time (seconds)', 'count': 'Number of Players'})
            st.plotly_chart(fig_actions, use_container_width=True)

    # Processing performance
    fig_perf = px.line(df, x='timestamp', y='processing_fps',
                      title='Processing Performance (FPS)',
                      labels={'timestamp': 'Time (seconds)', 'processing_fps': 'Processing FPS'})
    st.plotly_chart(fig_perf, use_container_width=True)

    # Pose quality over time
    fig_quality = px.line(df, x='timestamp', y='avg_pose_quality',
                         title='Average Pose Quality Over Time',
                         labels={'timestamp': 'Time (seconds)', 'avg_pose_quality': 'Pose Quality'})
    st.plotly_chart(fig_quality, use_container_width=True)

def show_model_status():
    """Show model availability status"""
    st.sidebar.markdown("### 📦 Model Status")

    model_manager = st.session_state.model_manager

    for model_name, model_config in YOLO_MODELS.items():
        model_info = model_manager.get_model_info(model_name)

        if model_info['available']:
            st.sidebar.success(f"✅ {model_config['name']}")
        else:
            st.sidebar.warning(f"⚠️ {model_config['name']} (will auto-download)")

def main():
    """Main application"""
    initialize_session_state()

    # Page navigation
    st.sidebar.markdown("# 🏠 Navigation")
    page = st.sidebar.selectbox(
        "Choose Application Mode:",
        ["Player Tracking", "Keypoint Detection"],
        index=0 if st.session_state.current_page == "Player Tracking" else 1
    )
    st.session_state.current_page = page

    if page == "Player Tracking":
        show_player_tracking_page()
    else:
        show_keypoint_detection_page()

def show_player_tracking_page():
    """Show player tracking page"""
    st.markdown('<h1 class="main-header">⚽ Football Player Re-ID System</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    config = create_sidebar()

    # Show model status
    show_model_status()

    # Main interface
    uploaded_file = create_main_interface()

    # Processing section
    if uploaded_file is not None:
        st.session_state.video_uploaded = True

        # Validate file
        is_valid, validation_message = validate_video_file(uploaded_file)

        if is_valid:
            # Show video info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "Status": "✅ Valid video file"
            }

            st.markdown("### 📋 Video Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Filename:** {file_details['Filename']}")
            with col2:
                st.write(f"**File size:** {file_details['File size']}")
            with col3:
                st.write(f"**Status:** {file_details['Status']}")

            # Configuration summary
            with st.expander("🔧 Processing Configuration"):
                st.write(f"**Model:** {YOLO_MODELS[config['model']]['name']}")
                st.write(f"**Tracker:** {TRACKER_CONFIGS[config['tracker']]['name']}")
                st.write(f"**Confidence:** {config['confidence_threshold']}")
                st.write(f"**Re-ID Threshold:** {config['reid_threshold']}")

            # Process button
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                with st.spinner("🔄 Initializing processing..."):
                    try:
                        results = process_video(uploaded_file, config)
                        if results:
                            st.balloons()
                            st.success("🎉 Processing completed successfully!")
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
                        st.exception(e)
        else:
            st.error(f"❌ {validation_message}")

    # Analytics section
    if st.session_state.results_data:
        st.markdown("---")
        show_analytics()

    # Performance metrics
    if hasattr(st.session_state, 'performance_monitor'):
        perf_stats = st.session_state.performance_monitor.get_performance_stats()
        if perf_stats['avg_fps'] > 0:
            st.sidebar.markdown("### ⚡ Performance")
            st.sidebar.metric("Processing FPS", f"{perf_stats['avg_fps']:.1f}")

def show_keypoint_detection_page():
    """Show keypoint detection and tracking page"""
    st.markdown('<h1 class="main-header">🤸 Keypoint Detection & Tracking</h1>', unsafe_allow_html=True)

    # Keypoint detection configuration
    keypoint_config = create_keypoint_sidebar()

    # Main interface for keypoint detection
    uploaded_file = create_keypoint_interface()

    # Processing section
    if uploaded_file is not None:
        # Validate file
        is_valid, validation_message = validate_video_file(uploaded_file)

        if is_valid:
            # Show video info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "Status": "✅ Valid video file"
            }

            st.markdown("### 📋 Video Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Filename:** {file_details['Filename']}")
            with col2:
                st.write(f"**File size:** {file_details['File size']}")
            with col3:
                st.write(f"**Status:** {file_details['Status']}")

            # Configuration summary
            with st.expander("🔧 Keypoint Configuration"):
                detection_types = st.session_state.keypoint_processor.get_detection_types()
                selected_type = detection_types[keypoint_config['detection_type']]
                st.write(f"**Detection Type:** {selected_type['name']}")
                st.write(f"**Description:** {selected_type['description']}")
                st.write(f"**Keypoints:** {selected_type['keypoints']}")
                st.write(f"**Confidence:** {keypoint_config['confidence_threshold']}")

            # Process button
            if st.button("🚀 Start Keypoint Detection", type="primary", use_container_width=True):
                with st.spinner("🔄 Initializing keypoint detection..."):
                    try:
                        results = process_keypoint_video(uploaded_file, keypoint_config)
                        if results:
                            st.balloons()
                            st.success("🎉 Keypoint detection completed successfully!")
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
                        st.exception(e)
        else:
            st.error(f"❌ {validation_message}")

    # Keypoint analytics section
    if st.session_state.keypoint_results_data:
        st.markdown("---")
        show_keypoint_analytics()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Keypoint Detection & Tracking System | Built with MediaPipe & Streamlit</p>
        <p><small>Upload a video to start detecting and tracking keypoints</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

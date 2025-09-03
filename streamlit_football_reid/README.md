# ⚽ Football Player Re-ID System

A modern web-based application for football player tracking and re-identification using state-of-the-art computer vision techniques.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🚀 Features

- **YOLOv8 Detection**: Multiple model variants for different speed/accuracy needs
- **Enhanced Re-ID**: Custom tracking with deep re-identification features
- **Real-time Processing**: Live video processing with progress tracking
- **Interactive Interface**: Modern Streamlit UI with drag-and-drop upload
- **Analytics Dashboard**: Comprehensive statistics and visualizations
- **Cloud Deployment**: Optimized for Streamlit Cloud deployment

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd streamlit_football_reid
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

### 🌐 Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Deploy your app**:

   - Repository: `your-username/your-repo-name`
   - Branch: `main`
   - Main file path: `streamlit_football_reid/app.py`

4. **The app will automatically install dependencies** from `requirements.txt` and `packages.txt`

### 📁 Required Files for Deployment

- `requirements.txt` - Python dependencies
- `packages.txt` - System dependencies
- `.streamlit/config.toml` - Streamlit configuration

## Usage

1. **Upload Video**: Drag and drop a football video file (MP4, AVI, MOV, MKV, WMV)
2. **Configure Parameters**:
   - Choose YOLO model (YOLOv8n for speed, YOLOv8x for accuracy)
   - Select tracker (OC-SORT recommended for accuracy, ByteTrack for speed)
   - Adjust detection and re-identification thresholds
3. **Process Video**: Click "Start Processing" to begin tracking
4. **View Results**: Watch real-time processing and view analytics

## Configuration Options

### YOLO Models

- **YOLOv8n**: Fastest, lowest accuracy
- **YOLOv8s**: Fast, good accuracy
- **YOLOv8m**: Balanced speed/accuracy
- **YOLOv8l**: Slow, high accuracy
- **YOLOv8x**: Slowest, highest accuracy

### Tracking Algorithms

- **OC-SORT**: Observation-centric SORT with improved association
- **ByteTrack**: Fast and accurate multi-object tracking
- **Custom ReID**: Custom tracker with deep re-identification features

### Parameters

- **Confidence Threshold**: Minimum confidence for player detections (0.1-1.0)
- **NMS Threshold**: Non-maximum suppression threshold (0.1-1.0)
- **Re-ID Threshold**: Re-identification similarity threshold (0.1-1.0)
- **Max Age**: Maximum frames to keep a track without detection
- **Min Hits**: Minimum detections before confirming a track

## File Structure

```
streamlit_football_reid/
├── app.py                 # Main Streamlit application
├── tracking_system.py     # Enhanced tracking system
├── video_processor.py     # Video processing utilities
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── run_app.py            # Startup script
├── requirements.txt      # Python dependencies
├── models/               # Model files directory
├── uploads/              # Temporary uploads
└── outputs/              # Processed videos
```

## Dependencies

- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **Ultralytics**: YOLOv8 implementation
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **Plotly**: Interactive charts
- **OC-SORT**: Advanced tracking algorithm
- **ByteTrack**: Multi-object tracking

## Performance Tips

1. **Model Selection**: Use YOLOv8n for real-time processing, YOLOv8l/x for best accuracy
2. **Video Resolution**: Lower resolution videos process faster
3. **Frame Skipping**: Adjust frame skip in config for faster processing
4. **Tracker Choice**: ByteTrack is faster, OC-SORT is more accurate

## Troubleshooting

### Common Issues

1. **Model Download Errors**: Models will auto-download on first use
2. **Memory Issues**: Use smaller YOLO models or reduce video resolution
3. **Slow Processing**: Adjust frame skip or use faster model variants
4. **Import Errors**: Ensure all requirements are installed

### Performance Optimization

- Use GPU acceleration if available (CUDA)
- Process videos at lower resolution for speed
- Adjust confidence thresholds to reduce false positives
- Use appropriate tracker for your use case

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!

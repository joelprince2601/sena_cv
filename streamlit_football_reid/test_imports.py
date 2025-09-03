"""
Test script to verify all imports work correctly
"""
import sys
import traceback

def test_imports():
    """Test all module imports"""
    print("Testing Football ReID Streamlit Application imports...")
    print("=" * 50)
    
    # Test basic imports
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        
    # Test optional imports
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"⚠️ Ultralytics not available: {e}")
        
    try:
        from scipy.spatial.distance import cosine
        print("✅ SciPy imported successfully")
    except ImportError as e:
        print(f"⚠️ SciPy not available: {e}")
        
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"⚠️ Scikit-learn not available: {e}")
    
    # Test our modules
    print("\nTesting custom modules...")
    print("-" * 30)
    
    try:
        import config
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Config module failed: {e}")
        traceback.print_exc()
        
    try:
        import utils
        print("✅ Utils module imported successfully")
    except Exception as e:
        print(f"❌ Utils module failed: {e}")
        traceback.print_exc()
        
    try:
        import tracking_system
        print("✅ Tracking system imported successfully")
    except Exception as e:
        print(f"❌ Tracking system failed: {e}")
        traceback.print_exc()
        
    try:
        import video_processor
        print("✅ Video processor imported successfully")
    except Exception as e:
        print(f"❌ Video processor failed: {e}")
        traceback.print_exc()
        
    try:
        import app
        print("✅ Main app imported successfully")
    except Exception as e:
        print(f"❌ Main app failed: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Import testing completed!")

if __name__ == "__main__":
    test_imports()

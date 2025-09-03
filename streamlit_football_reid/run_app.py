"""
Startup script for Football Player Re-ID Streamlit Application
"""
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    else:
        print("Requirements file not found. Installing basic packages...")
        packages = [
            "streamlit>=1.28.0",
            "opencv-python>=4.8.0", 
            "numpy>=1.24.0",
            "ultralytics>=8.0.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0",
            "plotly>=5.15.0"
        ]
        
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_environment():
    """Setup environment and paths"""
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Add OC_SORT to path if available
    ocsort_path = current_dir.parent / "OC_SORT"
    if ocsort_path.exists() and str(ocsort_path) not in sys.path:
        sys.path.insert(0, str(ocsort_path))
        print(f"Added OC_SORT path: {ocsort_path}")
    
    # Create necessary directories
    (current_dir / "models").mkdir(exist_ok=True)
    (current_dir / "uploads").mkdir(exist_ok=True)
    (current_dir / "outputs").mkdir(exist_ok=True)

def run_streamlit():
    """Run the Streamlit application"""
    app_file = Path(__file__).parent / "app.py"
    
    if not app_file.exists():
        print(f"Error: {app_file} not found!")
        return
    
    print("Starting Streamlit application...")
    print("The application will open in your default web browser.")
    print("Press Ctrl+C to stop the application.")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_file),
        "--server.address", "localhost",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ])

def main():
    """Main startup function"""
    print("=" * 50)
    print("Football Player Re-ID Streamlit Application")
    print("=" * 50)
    
    try:
        # Setup environment
        setup_environment()
        
        # Install requirements
        install_requirements()
        
        # Run application
        run_streamlit()
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

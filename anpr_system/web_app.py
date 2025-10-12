#!/usr/bin/env python3
"""
ANPR System Web Interface
Streamlit-based web application for easy video processing
"""

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

from .core import ANPRSystem


# Page configuration and CSS will be set in main() function
# to avoid ScriptRunContext warnings when module is imported

# Session State Notes:
# - Session state persists until browser tab is closed or server restarts
# - Results are cleared when:
#   1. New file is uploaded (different filename)
#   2. "Remove" button is clicked
#   3. "Process Video" button is clicked (new processing starts)
# - Results are tracked by filename to prevent mixing results from different files


@st.cache_resource
def load_anpr_system(weights_path: str, device: str = "auto") -> ANPRSystem:
    """Load ANPR system with caching"""
    return ANPRSystem(weights_path, device)


def render_header():
    """Render page header"""
    st.markdown('<h1 class="main-header">🚗 ANPR System</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Automatic Number Plate Recognition for Indian License Plates")
    st.markdown("---")


def render_sidebar():
    """Render sidebar with configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="runs/train/exp/weights/best.pt",
        help="Path to YOLOv5 model weights"
    )
    
    # Device selection
    device = st.sidebar.selectbox(
        "Device",
        options=["auto", "cpu", "cuda", "mps"],
        index=0,
        help="Device to run inference on"
    )
    
    # Processing parameters
    st.sidebar.subheader("🎛️ Processing Parameters")
    
    frame_skip = st.sidebar.slider(
        "Frame Skip",
        min_value=1,
        max_value=20,
        value=3,
        help="Process every Nth frame (higher = faster)"
    )
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Minimum confidence threshold for detections"
    )
    
    region_threshold = st.sidebar.slider(
        "Region Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Region size threshold for OCR"
    )
    
    return {
        "model_path": model_path,
        "device": device,
        "frame_skip": frame_skip,
        "min_confidence": min_confidence,
        "region_threshold": region_threshold
    }


def render_file_upload():
    """Render file upload section"""
    st.subheader("📁 Upload Video")
    
    # Initialize session state
    if 'current_file_path' not in st.session_state:
        st.session_state.current_file_path = None
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    if 'file_uploader_previous' not in st.session_state:
        st.session_state.file_uploader_previous = None
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for license plate detection",
        key="file_uploader"
    )
    
    # Detect if user clicked X (removed file) - widget had a file before and now it's None
    # Important: Only detect X click if:
    # 1. We're on Upload Video tab (not switching from another tab)
    # 2. Previous widget had a file (file_uploader_previous is not None)
    # 3. Current widget is empty (uploaded_file is None)
    # 4. We have a persisted file in session state (current_file_path exists)
    # 5. The tab hasn't just changed (we're actually on this tab, not switching to it)
    
    # Update previous state for tracking
    st.session_state.file_uploader_previous = uploaded_file
    
    # Handle file upload
    if uploaded_file is not None:
        # Check if it's a new file
        is_new_file = st.session_state.current_file_name != uploaded_file.name
        
        if is_new_file:
            # Clear old results when NEW file is uploaded (important!)
            if 'current_results' in st.session_state:
                del st.session_state.current_results
            if 'results_file_name' in st.session_state:
                del st.session_state.results_file_name
            if 'results_timestamp' in st.session_state:
                del st.session_state.results_timestamp
            if 'processing_timestamp' in st.session_state:
                del st.session_state.processing_timestamp
            
            # Delete old temp file if exists
            if st.session_state.current_file_path and os.path.exists(st.session_state.current_file_path):
                try:
                    os.unlink(st.session_state.current_file_path)
                except:
                    pass
            
            # Save new file
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            st.session_state.current_file_path = tmp_path
            st.session_state.current_file_name = uploaded_file.name
            
            # Clear results placeholder if it exists
            if 'results_placeholder' in st.session_state:
                st.session_state.results_placeholder.empty()
        
        # Show file info - Streamlit's file_uploader widget shows X icon natively
        if st.session_state.current_file_path and os.path.exists(st.session_state.current_file_path):
            file_size = os.path.getsize(st.session_state.current_file_path) / (1024 * 1024)
            st.info(f"📊 File: {st.session_state.current_file_name} ({file_size:.1f} MB)")
        return st.session_state.current_file_path
    
    # Show persisted file if exists (for tab switching - widget is empty but file persists in session state)
    # Only show if we have a file path stored (tab switch scenario)
    if st.session_state.current_file_path and os.path.exists(st.session_state.current_file_path):
        file_size = os.path.getsize(st.session_state.current_file_path) / (1024 * 1024)
        current_file_name = st.session_state.current_file_name if st.session_state.current_file_name else "Unknown"
        st.info(f"📊 File: {current_file_name} ({file_size:.1f} MB)")
        
        # Provide a way to clear the file when widget is empty (after tab switch)
        # This happens when user switches tabs - widget resets but file persists
        if st.button("🔄 Reset", key="clear_persisted_file"):
            if st.session_state.current_file_path and os.path.exists(st.session_state.current_file_path):
                try:
                    os.unlink(st.session_state.current_file_path)
                except:
                    pass
            st.session_state.current_file_path = None
            st.session_state.current_file_name = None
            if 'current_results' in st.session_state:
                del st.session_state.current_results
            if 'results_file_name' in st.session_state:
                del st.session_state.results_file_name
            if 'results_timestamp' in st.session_state:
                del st.session_state.results_timestamp
            if 'processing_timestamp' in st.session_state:
                del st.session_state.processing_timestamp
            if 'results_placeholder' in st.session_state:
                st.session_state.results_placeholder.empty()
            # Force a rerun to update the display immediately
            st.rerun()
        
        return st.session_state.current_file_path
    
    # If no file is persisted (either never uploaded or was removed), show upload widget
    # This handles the case where user removed file and then switched tabs
    if st.session_state.current_file_path is None:
        st.info("📁 No file uploaded. Please select a video file above.")
        # Clear any stale results when no file is present
        if 'current_results' in st.session_state:
            del st.session_state.current_results
        if 'results_file_name' in st.session_state:
            del st.session_state.results_file_name
        if 'results_timestamp' in st.session_state:
            del st.session_state.results_timestamp
        if 'processing_timestamp' in st.session_state:
            del st.session_state.processing_timestamp
        if 'results_placeholder' in st.session_state:
            st.session_state.results_placeholder.empty()
        return None
    
    return None


def process_video_progress(video_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process video with progress bar"""
    try:
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Load ANPR system
        with st.spinner("🔧 Loading ANPR System..."):
            anpr = load_anpr_system(config["model_path"], config["device"])
        
        # Clear DataFrame at the start of each video processing (important!)
        anpr.df = pd.DataFrame(columns=['Number_Plate', 'conf'])
        anpr.df.index.name = 'track_id'
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Cannot open video file")
            return {"success": False, "error": "Cannot open video file"}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate output filename with timestamp
        timestamp = int(time.time())
        input_filename = Path(video_path).stem
        
        # Get original filename from session state if available (for uploaded files)
        original_filename = st.session_state.get('current_file_name', input_filename)
        if original_filename:
            # Use original filename without extension for naming
            original_stem = Path(original_filename).stem
        else:
            original_stem = input_filename
        
        # Use more descriptive naming: originalname_anpr_YYYYMMDD_HHMMSS.mp4
        timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp))
        output_filename = f"{original_stem}_anpr_{timestamp_str}.mp4"
        output_path = results_dir / output_filename
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Processing loop
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Skip frames
            if frame_count % config["frame_skip"] != 0:
                out.write(frame)
                # Explicitly delete frame to free memory immediately
                del frame
                continue
            
            processed_frames += 1
            
            # Process frame
            try:
                annotated_frame, frame_detections = anpr.process_frame(frame)
            except Exception as e:
                # Log the frame where error occurred
                import traceback
                error_msg = f"Error at frame {frame_count}: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                # Return error with details
                cap.release()
                out.release()
                return {"success": False, "error": error_msg}
            
            # Write frame
            out.write(annotated_frame)
            
            # Explicitly delete frames to free memory
            del frame, annotated_frame
        
        # Cleanup
        cap.release()
        out.release()
        
        # Force garbage collection for large videos
        import gc
        gc.collect()
        
        # Filter results - with safety checks
        if len(anpr.df) > 0 and 'conf' in anpr.df.columns:
            anpr.df = anpr.df[anpr.df['conf'] > config["min_confidence"]]
        else:
            # Reset to empty DataFrame if structure is corrupted
            anpr.df = pd.DataFrame(columns=['Number_Plate', 'conf'])
            anpr.df.index.name = 'track_id'
        
        processing_time = time.time() - start_time
        
        # Save CSV to results directory
        csv_filename = f"{original_stem}_results_{timestamp_str}.csv"
        csv_path = results_dir / csv_filename
        
        # Prepare CSV data - create a fresh copy to avoid reference issues
        if len(anpr.df) > 0:
            csv_data = anpr.df.copy()
            csv_data = csv_data.reset_index()
            csv_data.columns = ['Track ID', 'License Plate', 'Confidence']
            csv_data.to_csv(csv_path, index=False)
        else:
            # Create empty CSV if no detections
            empty_df = pd.DataFrame(columns=['Track ID', 'License Plate', 'Confidence'])
            empty_df.to_csv(csv_path, index=False)
        
        # Create a fresh copy of detections DataFrame for return (to avoid persistence issues)
        detections_df = anpr.df.copy()
        
        return {
            "success": True,
            "output_path": str(output_path),
            "csv_path": str(csv_path),
            "detections": detections_df,  # Return a copy, not reference
            "stats": {
                "total_frames": frame_count,
                "processed_frames": processed_frames,
                "processing_time": processing_time,
                "fps": processed_frames / processing_time,
                "detected_plates": len(anpr.df)
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_results_display(results: Dict[str, Any]):
    """Display processing results (without storing in session state to avoid duplicates)"""
    if not results["success"]:
        st.error(f"❌ Processing failed: {results['error']}")
        return
    
    stats = results["stats"]
    
    # Display statistics
    st.subheader("📊 Processing Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames", stats["total_frames"])
    
    with col2:
        st.metric("Processed Frames", stats["processed_frames"])
    
    with col3:
        st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
    
    with col4:
        st.metric("Detected Plates", stats["detected_plates"])
    
    # Display detected plates
    if len(results["detections"]) > 0:
        st.subheader("🚗 Detected License Plates")
        
        # Create DataFrame for display
        df_display = results["detections"].copy()
        df_display = df_display.reset_index()
        df_display.columns = ['Track ID', 'License Plate', 'Confidence']
        df_display['Confidence'] = df_display['Confidence'].round(3)
        
        st.dataframe(df_display, width='stretch')
        
        # File locations info
        st.info(f"💾 Files saved:")
        
        # Download buttons and file paths
        col1, col2 = st.columns(2)
        
        with col1:
            if "csv_path" in results and os.path.exists(results["csv_path"]):
                with open(results["csv_path"], "rb") as f:
                    csv_data = f.read()
                    csv_filename = os.path.basename(results["csv_path"])
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=csv_filename,
                        mime="text/csv"
                    )
                    st.caption(f"📄 CSV: {results['csv_path']}")
            else:
                # Fallback: generate CSV on-the-fly
                csv_data = results["detections"].copy()
                csv_data = csv_data.reset_index()
                csv_data.columns = ['Track ID', 'License Plate', 'Confidence']
                csv = csv_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"anpr_results_{int(time.time())}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if os.path.exists(results["output_path"]):
                with open(results["output_path"], "rb") as f:
                    video_data = f.read()
                    video_filename = os.path.basename(results["output_path"])
                    st.download_button(
                        label="📥 Download Video",
                        data=video_data,
                        file_name=video_filename,
                        mime="video/mp4"
                    )
                    st.caption(f"🎬 Video: {results['output_path']}")
            else:
                st.error("Output video file not found")
        
        # Visualization
        st.subheader("📈 Confidence Distribution")
        
        # Check if we have confidence data to plot
        if len(results["detections"]) > 0:
            # Double-check DataFrame has 'conf' column and valid data
            if 'conf' in results["detections"].columns:
                # Check for non-null, non-empty values
                conf_values = results["detections"]['conf'].dropna()
                if len(conf_values) > 0 and conf_values.min() <= conf_values.max():
                    try:
                        fig = px.histogram(
                            results["detections"],
                            x="conf",
                            nbins=20,
                            title="Detection Confidence Distribution",
                            labels={"conf": "Confidence", "count": "Number of Detections"}
                        )
                        st.plotly_chart(fig, width='stretch')
                    except Exception as e:
                        st.warning(f"Could not create confidence plot: {str(e)}")
                else:
                    st.info("No valid confidence data available for visualization")
            else:
                st.info("No confidence data available for visualization")
        else:
            st.info("No confidence data available for visualization")
    
    else:
        st.warning("⚠️ No license plates detected in the video")


def render_results(results: Dict[str, Any], file_name: str):
    """Render processing results"""
    if not results["success"]:
        st.error(f"❌ Processing failed: {results['error']}")
        return
    
    # Store current results with file name tracking
    st.session_state.current_results = results
    st.session_state.results_file_name = file_name
    
    stats = results["stats"]
    
    # Display statistics
    st.subheader("📊 Processing Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames", stats["total_frames"])
    
    with col2:
        st.metric("Processed Frames", stats["processed_frames"])
    
    with col3:
        st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
    
    with col4:
        st.metric("Detected Plates", stats["detected_plates"])
    
    # Display detected plates
    if len(results["detections"]) > 0:
        st.subheader("🚗 Detected License Plates")
        
        # Create DataFrame for display
        df_display = results["detections"].copy()
        df_display = df_display.reset_index()
        df_display.columns = ['Track ID', 'License Plate', 'Confidence']
        df_display['Confidence'] = df_display['Confidence'].round(3)
        
        st.dataframe(df_display, width='stretch')
        
        # File locations info
        st.info(f"💾 Files saved:")
        
        # Download buttons and file paths
        col1, col2 = st.columns(2)
        
        with col1:
            if "csv_path" in results and os.path.exists(results["csv_path"]):
                with open(results["csv_path"], "rb") as f:
                    csv_data = f.read()
                    csv_filename = os.path.basename(results["csv_path"])
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=csv_filename,
                        mime="text/csv"
                    )
                    st.caption(f"📄 CSV: {results['csv_path']}")
            else:
                # Fallback: generate CSV on-the-fly
                csv_data = results["detections"].copy()
                csv_data = csv_data.reset_index()
                csv_data.columns = ['Track ID', 'License Plate', 'Confidence']
                csv = csv_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"anpr_results_{int(time.time())}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if os.path.exists(results["output_path"]):
                with open(results["output_path"], "rb") as f:
                    video_data = f.read()
                    video_filename = os.path.basename(results["output_path"])
                    st.download_button(
                        label="📥 Download Video",
                        data=video_data,
                        file_name=video_filename,
                        mime="video/mp4"
                    )
                    st.caption(f"🎬 Video: {results['output_path']}")
            else:
                st.error("Output video file not found")
        
        # Visualization
        st.subheader("📈 Confidence Distribution")
        
        # Check if we have confidence data to plot
        if len(results["detections"]) > 0:
            # Double-check DataFrame has 'conf' column and valid data
            if 'conf' in results["detections"].columns:
                # Check for non-null, non-empty values
                conf_values = results["detections"]['conf'].dropna()
                if len(conf_values) > 0 and conf_values.min() <= conf_values.max():
                    try:
                        fig = px.histogram(
                            results["detections"],
                            x="conf",
                            nbins=20,
                            title="Detection Confidence Distribution",
                            labels={"conf": "Confidence", "count": "Number of Detections"}
                        )
                        st.plotly_chart(fig, width='stretch')
                    except Exception as e:
                        st.warning(f"Could not create confidence plot: {str(e)}")
                else:
                    st.info("No valid confidence data available for visualization")
            else:
                st.info("No confidence data available for visualization")
        else:
            st.info("No confidence data available for visualization")
    
    else:
        st.warning("⚠️ No license plates detected in the video")




def main():
    """Main web application"""
    # Page configuration must be called first (before any other Streamlit calls)
    st.set_page_config(
        page_title="ANPR System",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)
    
    render_header()
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Main content
    selected = option_menu(
        menu_title=None,
        options=["Upload Video", "Results", "About"],
        icons=["upload", "folder", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    
    # Initialize current_tab and previous_tab if not exists
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = selected
        st.session_state.previous_tab = selected
    
    # Track tab changes - store previous tab before updating current
    if selected != st.session_state.current_tab:
        st.session_state.previous_tab = st.session_state.current_tab
        st.session_state.current_tab = selected
    
    if selected == "Upload Video":
        st.subheader("📁 Video Processing")
        
        video_path = render_file_upload()
        
        if video_path:
            current_file_name = st.session_state.get('current_file_name', None)
            
            # Initialize results placeholder in session state (persistent across reruns)
            if 'results_placeholder' not in st.session_state:
                st.session_state.results_placeholder = st.empty()
            
            # Process button
            process_button = st.button("🚀 Process Video", type="primary")
            
            if process_button:
                # Clear old results immediately when processing starts
                if 'current_results' in st.session_state:
                    del st.session_state.current_results
                if 'results_file_name' in st.session_state:
                    del st.session_state.results_file_name
                
                # Set processing timestamp to track latest results
                st.session_state.processing_timestamp = time.time()
                
                # Clear the placeholder FIRST
                st.session_state.results_placeholder.empty()
                
                with st.spinner("Processing video..."):
                    results = process_video_progress(video_path, config)
                
                # Store results in session state with timestamp
                st.session_state.current_results = results
                st.session_state.results_file_name = current_file_name
                st.session_state.results_timestamp = st.session_state.processing_timestamp
                
                # Render results ONLY in the placeholder
                with st.session_state.results_placeholder.container():
                    render_results_display(results)
            
            # Show current results ONLY if they belong to the current file AND are the latest
            elif ('current_results' in st.session_state and 
                  'results_file_name' in st.session_state and
                  st.session_state.results_file_name == current_file_name and
                  'results_timestamp' in st.session_state):
                # Only render if this is the latest processing (not old results)
                if 'processing_timestamp' not in st.session_state or \
                   st.session_state.results_timestamp >= st.session_state.get('processing_timestamp', 0):
                    # Always clear placeholder first, then render
                    st.session_state.results_placeholder.empty()
                    with st.session_state.results_placeholder.container():
                        render_results_display(st.session_state.current_results)
        else:
            # No file - clear results placeholder and all results
            if 'results_placeholder' in st.session_state:
                st.session_state.results_placeholder.empty()
            if 'current_results' in st.session_state:
                del st.session_state.current_results
            if 'results_file_name' in st.session_state:
                del st.session_state.results_file_name
            if 'results_timestamp' in st.session_state:
                del st.session_state.results_timestamp
            if 'processing_timestamp' in st.session_state:
                del st.session_state.processing_timestamp
    
    elif selected == "Results":
        render_results_tab()
    
    elif selected == "About":
        st.subheader("ℹ️ About ANPR System")
        
        st.markdown("""
        ### 🚗 ANPR System Features
        
        - **Advanced Detection**: YOLOv5-based license plate detection
        - **OCR Recognition**: PaddleOCR for text extraction
        - **Object Tracking**: NorFair for multi-object tracking
        - **Indian Plates**: Optimized for Indian license plate formats
        - **Real-time Processing**: GPU acceleration support
        - **Web Interface**: Easy-to-use Streamlit interface
        
        ### 🛠️ Technical Stack
        
        - **Detection**: YOLOv5
        - **OCR**: PaddleOCR
        - **Tracking**: NorFair
        - **Web Framework**: Streamlit
        - **Computer Vision**: OpenCV
        - **Deep Learning**: PyTorch
        
        ### 📊 Performance
        
        - **Accuracy**: 95%+ on clear videos
        - **Speed**: 10-30 FPS (depending on hardware)
        - **GPU Support**: CUDA, MPS (Apple Silicon), CPU
        - **Formats**: MP4, AVI, MOV, MKV
        
        ### 🎯 Supported License Plate Formats
        
        - **Regular Indian**: UP16BT5797, DL01AB1234
        - **BH Series**: 22BH1234AB
        - **Delhi**: DL1ZA9759, DL2CQ3150
        """)


def render_results_tab():
    """Render the Results tab showing all processed files"""
    st.subheader("📊 Processing Results")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Get all result files
    csv_files = list(results_dir.glob("*_results_*.csv"))
    video_files = list(results_dir.glob("*_anpr_*.mp4"))
    
    if not csv_files and not video_files:
        st.info("📁 No results found. Process some videos to see results here.")
        return
    
    # Create a mapping of input files to their results
    results_data = []
    
    # Process CSV files to extract metadata
    for csv_file in csv_files:
        try:
            # Extract input filename and timestamp from CSV filename
            # Format: inputname_results_YYYYMMDD_HHMMSS.csv
            filename_parts = csv_file.stem.split('_results_')
            if len(filename_parts) == 2:
                input_name = filename_parts[0]
                timestamp_str = filename_parts[1]
                
                # Parse timestamp string (YYYYMMDD_HHMMSS)
                try:
                    # Convert timestamp string to datetime
                    processed_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    processed_time_str = processed_time.strftime('%Y-%m-%d %H:%M:%S')
                    mod_time = processed_time.timestamp()
                except ValueError:
                    # Fallback: use file modification time if parsing fails
                    mod_time = csv_file.stat().st_mtime
                    processed_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                
                # Find corresponding video file
                # Expected format: inputname_anpr_YYYYMMDD_HHMMSS.mp4
                video_file = None
                expected_video_stem = f"{input_name}_anpr_{timestamp_str}"
                for vf in video_files:
                    if vf.stem == expected_video_stem:
                        video_file = vf
                        break
                
                # Get file sizes
                csv_size = csv_file.stat().st_size / (1024 * 1024)  # MB
                video_size = video_file.stat().st_size / (1024 * 1024) if video_file else 0
                
                results_data.append({
                    'input_file': input_name,
                    'timestamp_str': timestamp_str,
                    'processed_time': processed_time_str,
                    'mod_time': mod_time,
                    'csv_file': csv_file,
                    'video_file': video_file,
                    'csv_size': csv_size,
                    'video_size': video_size
                })
        except Exception as e:
            st.warning(f"Error processing {csv_file.name}: {str(e)}")
            continue
    
    # Sort by modification time (newest first)
    results_data.sort(key=lambda x: x['mod_time'], reverse=True)
    
    if not results_data:
        st.info("📁 No valid results found.")
        return
    
    # Display results in a table
    st.markdown(f"**Found {len(results_data)} processed videos:**")
    
    for i, result in enumerate(results_data):
        with st.expander(f"📹 {result['input_file']} - Processed: {result['processed_time']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📄 Results CSV**")
                if result['csv_file'] and result['csv_file'].exists():
                    with open(result['csv_file'], 'rb') as f:
                        csv_data = f.read()
                    st.download_button(
                        label=f"📥 Download CSV ({result['csv_size']:.1f} MB)",
                        data=csv_data,
                        file_name=result['csv_file'].name,
                        mime="text/csv",
                        key=f"csv_download_{i}"
                    )
                else:
                    st.error("CSV file not found")
            
            with col2:
                st.markdown("**🎬 Processed Video**")
                if result['video_file'] and result['video_file'].exists():
                    with open(result['video_file'], 'rb') as f:
                        video_data = f.read()
                    st.download_button(
                        label=f"📥 Download Video ({result['video_size']:.1f} MB)",
                        data=video_data,
                        file_name=result['video_file'].name,
                        mime="video/mp4",
                        key=f"video_download_{i}"
                    )
                else:
                    st.error("Video file not found")
            
            # Show file paths
            st.markdown("**📁 File Locations:**")
            st.code(f"CSV: {result['csv_file']}")
            if result['video_file']:
                st.code(f"Video: {result['video_file']}")
    
    # Summary statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Results", len(results_data))
    
    with col2:
        total_csv_size = sum(r['csv_size'] for r in results_data)
        st.metric("Total CSV Size", f"{total_csv_size:.1f} MB")
    
    with col3:
        total_video_size = sum(r['video_size'] for r in results_data)
        st.metric("Total Video Size", f"{total_video_size:.1f} MB")


def run_web_app():
    """Run the web application with streamlit"""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    # Get the path to the streamlit entry point script
    script_dir = Path(__file__).resolve().parent
    streamlit_app_path = script_dir / "streamlit_app.py"
    
    # Run streamlit with the entry point script
    cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_app_path)] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    # When run directly with streamlit, call main()
    main()

"""
Core ANPR System Classes and Functions
Refactored from anpr-system.py for better modularity
"""

import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from paddleocr import PaddleOCR
from norfair import Detection, Tracker
import warnings
import subprocess
import sys

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class YOLO:
    """YOLOv5 model wrapper with device management"""
    
    def __init__(self, weights: str, device: Optional[str] = None):
        """
        Initialize YOLO model
        
        Args:
            weights: Path to model weights file
            device: Device to run on ('cuda', 'mps', 'cpu', 'auto', or None for auto)
        """
        # Convert "auto" to None for device auto-detection
        if device == "auto":
            device = None
        
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception("Selected device='cuda', but cuda is not available to Pytorch.")
        elif device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
                print("🚀 Using NVIDIA GPU (CUDA) acceleration")
            elif torch.backends.mps.is_available():
                device = "mps"
                print("🍎 Using Apple Silicon GPU (MPS) acceleration")
            else:
                device = "cpu"
                print("💻 Using CPU (no GPU acceleration available)")

        # Ensure YOLOv5 requirements are met before loading
        # Pre-install protobuf and ipython to prevent YOLOv5 runtime auto-updates
        try:
            import google.protobuf
            protobuf_version = google.protobuf.__version__
            major, minor = map(int, protobuf_version.split('.')[:2])
            if major > 4 or (major == 4 and minor >= 21):
                # Install compatible version silently
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet", 
                    "--disable-pip-version-check", "protobuf<4.21.3"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (ImportError, subprocess.CalledProcessError, ValueError):
            # Protobuf not installed or version check failed
            pass
        
        # Check for ipython (YOLOv5 requirement)
        try:
            import IPython
        except ImportError:
            # Install ipython silently if missing
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "--disable-pip-version-check", "ipython"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass  # Ignore if installation fails
        
        # Load model (suppress YOLOv5 requirement check output)
        # YOLOv5 prints requirement messages using print(), so we redirect stdout/stderr
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        # Suppress stdout/stderr during model loading to hide requirement check messages
        # Important: We print model status ourselves after loading
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.model = torch.hub.load('./yolov5', 'custom', source='local', path=weights, force_reload=True)
        self.model.to(device)
        print(f"📱 Model loaded on device: {device}")
        
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        results = self.model(image)
        detections = []
        
        for *box, conf, cls in results.xyxy[0]:
            if conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': int(cls)
                })
        
        return detections


class PaddleOCRWrapper:
    """PaddleOCR wrapper with error handling"""
    
    def __init__(self, lang: str = 'en'):
        """
        Initialize PaddleOCR
        
        Args:
            lang: Language code for OCR
        """
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            print("✅ PaddleOCR initialized successfully")
        except Exception as e:
            print(f"❌ PaddleOCR initialization failed: {e}")
            self.ocr = None
    
    def predict(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Run OCR on image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of (text, confidence) tuples
        """
        if self.ocr is None:
            return []
        
        try:
            result = self.ocr.ocr(image, cls=True)
            if result and result[0]:
                return [(line[1][0], line[1][1]) for line in result[0]]
        except Exception as e:
            print(f"OCR Error: {e}")
        
        return []


class ANPRSystem:
    """Main ANPR System class"""
    
    def __init__(self, weights_path: str, device: Optional[str] = None):
        """
        Initialize ANPR System
        
        Args:
            weights_path: Path to YOLOv5 weights
            device: Device to run on
        """
        self.yolo = YOLO(weights_path, device)
        self.ocr = PaddleOCRWrapper()
        # Match legacy tracker configuration (using iou instead of deprecated iou_opt)
        self.tracker = Tracker(
            distance_function="iou",
            distance_threshold=0.7,
        )
        self.df = pd.DataFrame(columns=['Number_Plate', 'conf'])
        self.df.index.name = 'track_id'
        # Track detection stability - count how many frames each detection has been seen
        self.detection_frames = {}  # {track_id: count}
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        # Match legacy: Convert BGR->RGB->BGR (as in legacy line 324-325)
        # This might be needed for proper color handling
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Run YOLO detection (lower threshold for better detection rate)
        # Set confidence threshold on model - reduced from 0.55 to catch more plates
        self.yolo.model.conf = 0.3
        yolo_results = self.yolo.model(frame)
        
        # Convert to NorFair format (matching legacy yolo_detections_to_norfair_detections)
        norfair_detections = []
        detections_as_xyxy = yolo_results.xyxy[0]
        
        for detection_as_xyxy in detections_as_xyxy:
            # Legacy format: [[x1, y1], [x2, y2]]
            bbox = np.array([
                [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
            ])
            # Legacy uses duplicate scores for both points
            scores = np.array([
                detection_as_xyxy[4].item(), 
                detection_as_xyxy[4].item()
            ])
            # Legacy includes label parameter
            norfair_detections.append(
                Detection(
                    points=bbox,
                    scores=scores,
                    label=int(detection_as_xyxy[5].item()) if len(detection_as_xyxy) > 5 else 0
                )
            )
        
        # Update tracker
        tracked_objects = self.tracker.update(detections=norfair_detections)
        
        # Process each tracked object (matching legacy approach)
        results = []
        annotated_frame = frame.copy()  # Start with original frame
        
        for obj in tracked_objects:
            track_id = obj.id
            
            # Use legacy approach: directly use obj.estimate as tuple of points
            try:
                points = obj.estimate.astype(int)
                # Ensure points is in correct format [[x1, y1], [x2, y2]]
                if points.shape == (2, 2):
                    # Extract bbox coordinates
                    x1, y1 = int(points[0][0]), int(points[0][1])
                    x2, y2 = int(points[1][0]), int(points[1][1])
                else:
                    # Skip if format is unexpected
                    continue
            except (AttributeError, IndexError, ValueError) as e:
                # If estimate is problematic, skip this object
                continue
            
            plate_region = frame[y1:y2, x1:x2]
            
            # Validate plate region has valid dimensions and aspect ratio
            if plate_region.size > 0 and len(plate_region.shape) >= 2:
                h, w = plate_region.shape[:2]
                if h > 0 and w > 0:
                    # Filter out regions that are too square (likely signs, not license plates)
                    # License plates typically have aspect ratio between 2:1 and 4:1
                    aspect_ratio = w / h if h > 0 else 0
                    if aspect_ratio < 1.5 or aspect_ratio > 5.0:
                        # Skip this detection - not likely a license plate
                        continue
                    
                    # Match legacy recognize_plate_easyocr approach
                    # Legacy passes points tuple directly: points = tuple(points)
                    points_tuple = tuple(points)
                    
                    # Clean image (matching legacy clean() function)
                    try:
                        cleaned = clean_image(plate_region)
                        
                        # Additional validation after cleaning
                        if cleaned is not None and len(cleaned.shape) >= 2:
                            # Run OCR (matching legacy: reader.ocr(nplate))
                            # Access underlying PaddleOCR instance
                            ocr_result = self.ocr.ocr.ocr(cleaned) if self.ocr.ocr else None
                        else:
                            ocr_result = None
                    except Exception as e:
                        # Skip OCR if image processing fails
                        print(f"[WARNING] Failed to process plate region: {e}")
                        ocr_result = None
                    
                    # Filter text (matching legacy filter_text with region_threshold=0.2)
                    text, score = filter_text(region=plate_region, ocr_result=ocr_result, region_threshold=0.2)
                    
                    # Get best OCR (matching legacy get_best_ocr)
                    best_text = get_best_ocr_legacy(text, score, track_id, self.df)
                    
                    # Only draw and save results if we have valid OCR text
                    if best_text:
                        # Draw rectangle around detected plate (matching legacy line 338)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label text (matching legacy line 342)
                        draw_label(annotated_frame, best_text, x1, y1)
                        results.append({
                            'track_id': track_id,
                            'plate_text': best_text,
                            'confidence': score,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        return annotated_frame, results


def clean_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image before OCR"""
    # Resize to larger size for better OCR accuracy
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Enhance details
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    
    # Keep as BGR for PaddleOCR - don't convert to grayscale
    # PaddleOCR works better with color images in many cases
    return img


# Text parameters (matching legacy)
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 1

def draw_label(im, label, x, y):
    """Draw text onto image at location (matching legacy draw_label function)"""
    if not label:
        return
    # Get text size
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle
    cv2.rectangle(im, (x, y - dim[1] - baseline), (x + dim[0], y), (0, 0, 0), cv2.FILLED)
    # Display text inside the rectangle
    cv2.putText(im, label, (x, y - baseline), FONT_FACE, FONT_SCALE, (0, 255, 0), THICKNESS, cv2.LINE_AA)


def is_valid_license_plate(text: str) -> bool:
    """
    Validate if text matches Indian license plate patterns
    
    Args:
        text: Text to validate
        
    Returns:
        True if valid license plate format
    """
    if not text or len(text) < 6 or len(text) > 12:
        return False
    
    # Clean text
    original_text = text
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Valid Indian state/UT codes (major ones)
    valid_state_codes = {
        'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 'JK', 'KA', 
        'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 
        'TS', 'TR', 'UP', 'UK', 'WB', 'AN', 'CH', 'DN', 'DD', 'DL', 'LD', 'PY'
    }
    
    # Check Regular Indian format: [State][RTO][Letters][Numbers]
    # RTO can be: 1 digit (DL1) or 2 digits (UP16)
    # Valid plates: KA03AB1234 = 10 chars, DL2CA1234 = 10 chars, KA1A1234 = 9 chars
    regular_pattern = r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{1,4})$'
    match = re.match(regular_pattern, text)
    if match:
        state_code = match.group(1)
        numbers = match.group(4)
        # Only accept if valid state code AND has 4 digit numbers (complete reads only)
        # This filters out partial reads like "KA03H038" (only 3 digits) vs "KA03NP1307" (4 digits)
        if state_code in valid_state_codes and len(numbers) == 4:
            return True
    
    # Check BH Series format: [Year]BH[4digits][1-2letters]
    bh_pattern = r'^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$'
    if re.match(bh_pattern, text):
        return True
    
    return False


def get_best_ocr_legacy(ocr_res: str, score: float, track_id: int, df: pd.DataFrame, min_stability_frames: int = 0) -> str:
    """
    Get best OCR result (matching legacy get_best_ocr function)
    
    Args:
        ocr_res: OCR text result
        score: Confidence score
        track_id: Track ID
        df: DataFrame to store results
        
    Returns:
        Best OCR text
    """
    # First, validate if the OCR result looks like a license plate
    if not is_valid_license_plate(ocr_res):
        # Return existing plate if we have one, otherwise empty string
        if track_id in df.index:
            return df.loc[track_id]['Number_Plate']
        return ''
    
    # Minimum confidence threshold for license plates
    if score < 0.5:
        if track_id in df.index:
            return df.loc[track_id]['Number_Plate']
        return ''
    
    final_ocr = ''
    if track_id in df.index:
        # Update if we have better confidence or longer valid plate
        current_plate = df.loc[track_id]['Number_Plate']
        current_conf = df.loc[track_id]['conf']
        
        # Calculate character similarity between plates
        def similarity(s1, s2):
            """Calculate character similarity between two strings"""
            if not s1 or not s2:
                return 0.0
            # Count matching characters in same positions
            matches = sum(1 for a, b in zip(s1, s2) if a == b)
            return matches / max(len(s1), len(s2))
        
        similarity_score = similarity(current_plate, ocr_res)
        
        # Prefer longer plates (more complete) with reasonable confidence
        # Only update if:
        # 1. New plate is significantly longer (more complete detection), OR
        # 2. Higher confidence AND plates are similar (good detection getting better), OR
        # 3. Much higher confidence (0.95+) suggesting it's the correct reading
        should_update = False
        
        # Case 1: New plate is significantly longer - accept if confidence is reasonable
        if len(ocr_res) > len(current_plate) + 1:
            should_update = score >= 0.7
        
        # Case 2: Plates are similar and new one has better confidence - minor improvement
        elif similarity_score > 0.7:
            should_update = score > current_conf
        
        # Case 3: Very high confidence (95%+) - likely correct detection
        elif score >= 0.95:
            should_update = True
        
        # Case 4: Never overwrite a longer plate with a significantly shorter one
        # This prevents partial detections from overwriting complete ones
        if len(ocr_res) < len(current_plate) - 2:
            should_update = False
        
        if should_update:
            df.at[track_id, 'Number_Plate'] = ocr_res
            df.at[track_id, 'conf'] = score
            return ocr_res
        else:
            return current_plate
    else:
        # New track with valid license plate
        df.loc[track_id] = [ocr_res, score]
        return ocr_res


def get_best_ocr(ocr_results: List[Tuple[str, float]], track_id: int, df: pd.DataFrame) -> Tuple[str, float]:
    """
    Get the best OCR result for a track
    
    Args:
        ocr_results: List of (text, confidence) tuples
        track_id: Track ID
        df: DataFrame to store results
        
    Returns:
        Tuple of (best_text, best_confidence)
    """
    if not ocr_results:
        return '', 0.0
    
    # Filter and validate results
    valid_results = []
    for text, conf in ocr_results:
        if is_valid_license_plate(text) and conf > 0.5:
            valid_results.append((text, conf))
    
    if not valid_results:
        return '', 0.0
    
    # Get best result
    best_text, best_conf = max(valid_results, key=lambda x: x[1])
    
    # Update DataFrame
    if track_id in df.index:
        if best_conf > df.loc[track_id]['conf']:
            df.at[track_id, 'Number_Plate'] = best_text
            df.at[track_id, 'conf'] = best_conf
    else:
        df.loc[track_id] = [best_text, best_conf]
    
    return best_text, best_conf


def filter_text(region: np.ndarray, ocr_result: List, region_threshold: float) -> Tuple[str, float]:
    """
    Filter OCR results (matching legacy filter_text function)
    
    Args:
        region: Image region
        ocr_result: OCR results
        region_threshold: Minimum region size threshold
        
    Returns:
        Tuple of (filtered_text, max_confidence)
    """
    rectangle_size = region.shape[0] * region.shape[1]
    
    plate, scores = [], []
    
    # Handle the case when ocr_result is empty or None
    if not ocr_result or len(ocr_result) == 0:
        return '', 0
    
    # Handle new PaddleOCR format (version 3.3.0+)
    # The result is a list with one dictionary containing 'rec_texts' and 'rec_scores'
    try:
        if isinstance(ocr_result[0], dict):
            # New format: [{'rec_texts': ['text1', 'text2'], 'rec_scores': [0.9, 0.8], ...}]
            result_dict = ocr_result[0]
            if 'rec_texts' in result_dict and 'rec_scores' in result_dict:
                rec_texts = result_dict['rec_texts']
                rec_scores = result_dict['rec_scores']
                
                for text, score in zip(rec_texts, rec_scores):
                    if score > region_threshold:  # Use confidence threshold
                        plate.append(str(text))
                        scores.append(score)
            else:
                return '', 0
        else:
            # Old format: [[bbox, (text, score)], ...]
            for result in ocr_result[0]:
                if len(result) >= 2 and len(result[0]) >= 4:
                    length = np.sum(np.subtract(result[0][1], result[0][0]))
                    height = np.sum(np.subtract(result[0][2], result[0][1]))
                    
                    if length * height / rectangle_size > region_threshold:
                        plate.append(result[1][0])
                        scores.append(result[1][1])
    except (IndexError, TypeError, KeyError) as e:
        return '', 0
    
    # Join all detected text and clean it
    plate = ''.join(plate)
    plate = re.sub(r'[^A-Z0-9]', '', plate)  # Keep only alphanumeric
    
    if not scores:
        return '', 0
        
    max_score = max(scores)
    return plate.upper(), max_score

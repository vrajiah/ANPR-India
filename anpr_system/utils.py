"""
Utility functions for ANPR System
"""

import re
import numpy as np
import cv2
from typing import Tuple, List, Optional


def clean_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image before OCR"""
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    # Keep as BGR for PaddleOCR - don't convert to grayscale
    return img


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


def filter_text(region: np.ndarray, ocr_result: List, region_threshold: float) -> Tuple[str, float]:
    """
    Filter OCR results based on region size and confidence
    
    Args:
        region: Image region
        ocr_result: OCR results
        region_threshold: Minimum region size threshold
        
    Returns:
        Tuple of (filtered_text, max_confidence)
    """
    rectangle_size = region.shape[0] * region.shape[1]
    
    plate, scores = [], []
    
    if not ocr_result or len(ocr_result) == 0:
        return '', 0
    
    try:
        for result in ocr_result[0]:
            if len(result) >= 2 and len(result[0]) >= 4:
                length = np.sum(np.subtract(result[0][1], result[0][0]))
                height = np.sum(np.subtract(result[0][2], result[0][1]))
                
                if length * height / rectangle_size > region_threshold:
                    plate.append(result[1][0])
                    scores.append(result[1][1])
    except Exception as e:
        print(f"Error processing OCR result: {e}")
        return '', 0
    
    plate = ''.join(plate)
    plate = re.sub(r'\\W+', '', plate)
    
    if not scores:
        plate = ''
        scores.append(0)
    
    return plate.upper(), max(scores)


def get_best_ocr(ocr_results: List[Tuple[str, float]], track_id: int, df) -> Tuple[str, float]:
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


def format_license_plate(text: str) -> str:
    """
    Format license plate text for display
    
    Args:
        text: Raw license plate text
        
    Returns:
        Formatted license plate text
    """
    if not text:
        return ""
    
    # Clean text
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Format based on pattern
    if re.match(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{1,4})$', text):
        # Regular Indian format: UP16BT5797 -> UP 16 BT 5797
        match = re.match(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{1,4})$', text)
        if match:
            state, rto, letters, numbers = match.groups()
            return f"{state} {rto} {letters} {numbers}"
    
    elif re.match(r'^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$', text):
        # BH Series format: 22BH1234AB -> 22 BH 1234 AB
        match = re.match(r'^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$', text)
        if match:
            year, numbers, letters = match.groups()
            return f"{year} BH {numbers} {letters}"
    
    return text


def validate_video_file(file_path: str) -> bool:
    """
    Validate video file format and accessibility
    
    Args:
        file_path: Path to video file
        
    Returns:
        True if valid video file
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        
        # Check if we can read at least one frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    except Exception:
        return False


def get_video_info(file_path: str) -> dict:
    """
    Get video file information
    
    Args:
        file_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {}
        
        info = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    except Exception:
        return {}


def create_thumbnail(video_path: str, output_path: str, timestamp: float = 1.0) -> bool:
    """
    Create thumbnail from video
    
    Args:
        video_path: Path to video file
        output_path: Path to save thumbnail
        timestamp: Time in seconds to capture frame
        
    Returns:
        True if successful
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Set position
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            cv2.imwrite(output_path, frame)
            return True
        
        return False
    except Exception:
        return False

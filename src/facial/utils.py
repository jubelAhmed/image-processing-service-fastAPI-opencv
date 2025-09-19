"""
Utility functions for facial processing.
"""

import base64
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from src.facial.schemas import LandmarkPoint, MaskContours
from src.facial.exceptions import InvalidImageException, NoFaceDetectedException


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise InvalidImageException("Failed to decode image data")
        
        return image
        
    except Exception as e:
        raise InvalidImageException(f"Invalid image data: {str(e)}")


def decode_segmentation_map(segmentation_data: str) -> np.ndarray:
    """Decode base64 segmentation map to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in segmentation_data:
            segmentation_data = segmentation_data.split(',')[1]
        
        # Decode base64
        map_bytes = base64.b64decode(segmentation_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(map_bytes, np.uint8)
        
        # Decode image
        segmentation_map = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if segmentation_map is None:
            raise InvalidImageException("Failed to decode segmentation map")
        
        return segmentation_map
        
    except Exception as e:
        raise InvalidImageException(f"Invalid segmentation map data: {str(e)}")


def validate_face_mesh(landmarks: List[LandmarkPoint]) -> bool:
    """Validate face mesh landmarks."""
    if not isinstance(landmarks, list):
        return False
    
    # Check minimum points for face detection
    if len(landmarks) < 68:
        return False
    
    # Check if landmarks have valid coordinates
    for landmark in landmarks:
        if not hasattr(landmark, 'x') or not hasattr(landmark, 'y'):
            return False
        if not isinstance(landmark.x, (int, float)) or not isinstance(landmark.y, (int, float)):
            return False
    
    return True


def extract_result_data(result_data: Dict[str, Any]) -> Tuple[str, MaskContours]:
    """Extract SVG and mask contours from the result data."""
    if not result_data:
        raise ValueError("Result data is empty")
    
    svg_data = result_data.get("svg", "")
    mask_contours = result_data.get("mask_contours", {})
    
    if not svg_data:
        raise ValueError("SVG data is missing in the result")
    
    if not isinstance(mask_contours, dict):
        raise ValueError("Mask contours must be a dictionary")
    
    return svg_data, mask_contours


def calculate_image_hash(image: np.ndarray) -> str:
    """Calculate perceptual hash for image similarity."""
    try:
        # Resize image to standard size
        resized = cv2.resize(image, (8, 8))
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate average
        avg = gray.mean()
        
        # Create hash
        hash_bits = []
        for row in gray:
            for pixel in row:
                hash_bits.append('1' if pixel > avg else '0')
        
        # Convert to hex
        hash_string = ''.join(hash_bits)
        return hex(int(hash_string, 2))[2:].zfill(16)
        
    except Exception as e:
        raise InvalidImageException(f"Failed to calculate image hash: {str(e)}")


def normalize_landmarks(landmarks: List[LandmarkPoint], image_shape: Tuple[int, int]) -> List[LandmarkPoint]:
    """Normalize landmarks to image coordinates."""
    height, width = image_shape[:2]
    
    normalized = []
    for landmark in landmarks:
        # Ensure coordinates are within image bounds
        x = max(0, min(width - 1, landmark.x))
        y = max(0, min(height - 1, landmark.y))
        
        normalized.append(LandmarkPoint(x=x, y=y))
    
    return normalized


def calculate_contour_area(contour: List[List[int]]) -> float:
    """Calculate area of contour using shoelace formula."""
    if len(contour) < 3:
        return 0.0
    
    n = len(contour)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += contour[i][0] * contour[j][1]
        area -= contour[j][0] * contour[i][1]
    
    return abs(area) / 2.0


def calculate_contour_centroid(contour: List[List[int]]) -> Dict[str, float]:
    """Calculate centroid of contour points."""
    if not contour:
        return {"x": 0.0, "y": 0.0}
    
    x_coords = [point[0] for point in contour]
    y_coords = [point[1] for point in contour]
    
    return {
        "x": sum(x_coords) / len(x_coords),
        "y": sum(y_coords) / len(y_coords)
    }


def calculate_bounding_box(contour: List[List[int]]) -> Dict[str, int]:
    """Calculate bounding box of contour."""
    if not contour:
        return {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0, "width": 0, "height": 0}
    
    x_coords = [point[0] for point in contour]
    y_coords = [point[1] for point in contour]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": x_max - x_min,
        "height": y_max - y_min
    }

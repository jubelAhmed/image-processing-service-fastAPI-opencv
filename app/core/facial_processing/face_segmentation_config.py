import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class SegmentationConfig:
    """Configuration class for face region segmentation parameters."""
    # Morphology and region processing
    morphology_kernel_size: int = 5
    boundary_thickness: int = 3
    min_region_area: int = 1
    
    # Visual overlay settings
    overlay_alpha: float = 0.1
    dot_spacing: int = 8
    dot_radius: int = 2
    
    # Colors (BGR format)
    default_color: Tuple[int, int, int] = (255, 0, 0)
    
    # Face anatomy ratios
    forehead_height_ratio: float = 0.30
    face_width_center_ratio: float = 0.65
    
    # Ear detection parameters
    ear_detection_threshold: float = 0.15
    ear_min_width: int = 20
    ear_search_start_ratio: float = 0.3
    ear_search_end_ratio: float = 0.8
    
    # Landmark indices (dlib 68-point model)
    left_eye_indices: List[int] = None
    right_eye_indices: List[int] = None
    nose_indices: List[int] = None
    
    def __post_init__(self):
        if self.left_eye_indices is None:
            self.left_eye_indices = list(range(36, 42))
        if self.right_eye_indices is None:
            self.right_eye_indices = list(range(42, 48))
        if self.nose_indices is None:
            self.nose_indices = list(range(27, 36))


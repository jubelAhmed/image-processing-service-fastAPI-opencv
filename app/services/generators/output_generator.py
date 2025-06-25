"""
Abstract interface and output format implementations.
"""

import base64
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from app.schemas.face_schema import MaskContours

class OutputGenerator(ABC):
    """Abstract base class for different output formats."""
    
    @abstractmethod
    def generate(self, image_shape: Tuple[int, int], regions: MaskContours, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        pass



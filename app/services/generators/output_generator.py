"""
Abstract interface and output format implementations.
"""

import base64
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class OutputGenerator(ABC):
    """Abstract base class for different output formats."""
    
    @abstractmethod
    def generate(self, image_shape: Tuple[int, int], facial_regions: Dict[str, List]) -> str:
        pass



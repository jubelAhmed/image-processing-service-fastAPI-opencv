"""
Image generator factory.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
from app.services.generators.output_generator import OutputGenerator
from app.services.generators.svg_generator import SVGGenerator
from app.schemas.face_schema import MaskContours

class ImageGenerator:
    """Main processor for image extraction and output generation."""

    def __init__(self, generator: OutputGenerator = None):
        self.generator = generator or SVGGenerator()

    def create(self, image_shape: Tuple[int, int], contours: MaskContours, processed_image: Optional[np.ndarray] = None) -> str:
        if not self._validate_inputs(image_shape, contours):
            raise ValueError("Invalid input parameters")
        return self.generator.generate(image_shape, contours, processed_image)

    def set_generator(self, generator: OutputGenerator) -> None:
        self.generator = generator

    def _validate_inputs(self, image_shape: Tuple[int, int], contours: MaskContours) -> bool:
        if not image_shape or len(image_shape) != 2:
            return False
        if image_shape[0] <= 0 or image_shape[1] <= 0:
            return False
        if not isinstance(contours, dict):
            return False
        return True

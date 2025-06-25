"""
Image generator factory.
"""

from typing import Tuple, List, Dict
from app.services.generators.output_generator import OutputGenerator

class FileProcessor:
    """Main processor for image extraction and output generation."""

    def __init__(self, generator: OutputGenerator):
        self.generator = generator

    def create(self, image_shape: Tuple[int, int], regions: Dict[str, List]) -> str:
        if not self._validate_inputs(image_shape, regions):
            raise ValueError("Invalid input parameters")
        return self.generator.generate(image_shape, regions)

    def set_generator(self, generator: OutputGenerator) -> None:
        self.generator = generator

    def _validate_inputs(self, image_shape: Tuple[int, int], regions: Dict[str, List]) -> bool:
        if not image_shape or len(image_shape) != 2:
            return False
        if image_shape[0] <= 0 or image_shape[1] <= 0:
            return False
        if not isinstance(regions, dict):
            return False
        return True

"""
Image generator factory with improved dependency injection and error handling.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
from src.services.generators.output_generator import OutputGenerator
from src.services.generators.svg_generator import SVGGenerator
from src.services.generators.png_generator import PNGGenerator
from src.services.generators.json_generator import JSONGenerator
from src.services.style_config import StyleConfig, DefaultStyleConfig
from src.schemas.face_schema import MaskContours
from src.exceptions import InvalidInputError, ProcessingError

class ImageGenerator:
    """Main processor for image extraction and output generation with improved OOP practices."""

    def __init__(self, generator: OutputGenerator = None, style_config: StyleConfig = None):
        """
        Initialize ImageGenerator with dependency injection.
        
        Args:
            generator: Output generator implementation (defaults to SVGGenerator)
            style_config: Style configuration for generators (defaults to DefaultStyleConfig)
        """
        self.generator = generator or SVGGenerator()
        self.style_config = style_config or DefaultStyleConfig()

    def create(self, image_shape: Tuple[int, int], contours: MaskContours, processed_image: Optional[np.ndarray] = None) -> str:
        """
        Create output using the configured generator.
        
        Args:
            image_shape: Tuple of (height, width)
            contours: Dictionary of region contours
            processed_image: Optional processed image array
            
        Returns:
            Generated output as string
            
        Raises:
            InvalidInputError: If input parameters are invalid
            ProcessingError: If generation fails
        """
        try:
            if not self._validate_inputs(image_shape, contours):
                raise InvalidInputError("Invalid input parameters provided")
            
            return self.generator.generate(image_shape, contours, processed_image)
            
        except InvalidInputError:
            raise
        except Exception as e:
            raise ProcessingError(f"Failed to generate output: {str(e)}") from e

    def set_generator(self, generator: OutputGenerator) -> None:
        """Set a new output generator."""
        self.generator = generator

    def set_style_config(self, style_config: StyleConfig) -> None:
        """Set a new style configuration."""
        self.style_config = style_config

    def get_available_generators(self) -> Dict[str, type]:
        """Get dictionary of available generator types."""
        return {
            'svg': SVGGenerator,
            'png': PNGGenerator,
            'json': JSONGenerator
        }

    def create_generator(self, generator_type: str) -> OutputGenerator:
        """
        Create a generator instance by type.
        
        Args:
            generator_type: Type of generator ('svg', 'png', 'json')
            
        Returns:
            Generator instance
            
        Raises:
            ValueError: If generator type is not supported
        """
        generators = self.get_available_generators()
        generator_class = generators.get(generator_type.lower())
        
        if not generator_class:
            raise ValueError(f"Unsupported generator type: {generator_type}")
        
        return generator_class()

    def _validate_inputs(self, image_shape: Tuple[int, int], contours: MaskContours) -> bool:
        """
        Validate input parameters.
        
        Args:
            image_shape: Image dimensions tuple
            contours: Contours dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not image_shape or len(image_shape) != 2:
            return False
        if image_shape[0] <= 0 or image_shape[1] <= 0:
            return False
        if not isinstance(contours, dict):
            return False
        return True

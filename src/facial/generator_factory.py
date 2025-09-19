"""
Factory for creating output generators with different configurations.
"""

from typing import Dict, Type, Optional
from src.services.generators.output_generator import OutputGenerator
from src.services.generators.svg_generator import SVGGenerator
from src.services.generators.png_generator import PNGGenerator
from src.services.generators.json_generator import JSONGenerator
from src.services.style_config import StyleConfig, StyleConfigFactory
from src.exceptions import InvalidInputError


class GeneratorFactory:
    """Factory for creating output generators with different configurations."""
    
    _generators: Dict[str, Type[OutputGenerator]] = {
        'svg': SVGGenerator,
        'png': PNGGenerator,
        'json': JSONGenerator
    }
    
    @classmethod
    def create_generator(cls, generator_type: str, style_config: Optional[StyleConfig] = None) -> OutputGenerator:
        """
        Create a generator instance by type with optional style configuration.
        
        Args:
            generator_type: Type of generator ('svg', 'png', 'json')
            style_config: Optional style configuration (only used by SVG generator)
            
        Returns:
            Generator instance
            
        Raises:
            InvalidInputError: If generator type is not supported
        """
        generator_class = cls._generators.get(generator_type.lower())
        
        if not generator_class:
            raise InvalidInputError(f"Unsupported generator type: {generator_type}. "
                                  f"Supported types: {list(cls._generators.keys())}")
        
        # Only SVG generator uses style configuration
        if generator_type.lower() == 'svg' and style_config:
            return generator_class(style_config)
        else:
            return generator_class()
    
    @classmethod
    def get_available_generators(cls) -> Dict[str, Type[OutputGenerator]]:
        """Get dictionary of available generator types."""
        return cls._generators.copy()
    
    @classmethod
    def create_with_style(cls, generator_type: str, style_type: str = "default") -> OutputGenerator:
        """
        Create a generator with a specific style configuration.
        
        Args:
            generator_type: Type of generator ('svg', 'png', 'json')
            style_type: Type of style configuration ('default', 'colorful', 'minimal')
            
        Returns:
            Generator instance with style configuration
        """
        style_config = StyleConfigFactory.create_style_config(style_type)
        return cls.create_generator(generator_type, style_config)
    
    @classmethod
    def register_generator(cls, name: str, generator_class: Type[OutputGenerator]) -> None:
        """
        Register a new generator type.
        
        Args:
            name: Name of the generator type
            generator_class: Generator class to register
        """
        if not issubclass(generator_class, OutputGenerator):
            raise InvalidInputError(f"Generator class must inherit from OutputGenerator")
        
        cls._generators[name.lower()] = generator_class

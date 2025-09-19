"""
Style configuration strategy pattern for output generators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RegionStyle:
    """Data class for region styling information."""
    stroke: str
    fill: str
    stroke_width: int = 2
    stroke_dasharray: Optional[str] = None
    font_size: int = 26
    text_color: str = "white"


class StyleConfig(ABC):
    """Abstract base class for style configuration strategies."""
    
    @abstractmethod
    def get_region_style(self, region_id: int) -> RegionStyle:
        """Get style configuration for a specific region."""
        pass
    
    @abstractmethod
    def get_default_style(self) -> RegionStyle:
        """Get default style configuration."""
        pass


class DefaultStyleConfig(StyleConfig):
    """Default style configuration with purple theme."""
    
    def __init__(self):
        self.region_styles = {
            1: RegionStyle(
                stroke="#9D57A7", 
                fill="rgba(161, 106, 169, 0.5)",
                stroke_dasharray="5,5"
            ),
            2: RegionStyle(
                stroke="#A16AA9", 
                fill="rgba(161, 106, 169, 0.5)",
                stroke_dasharray="5,5"
            ),
            3: RegionStyle(
                stroke="#A16AA9", 
                fill="rgba(161, 106, 169, 0.5)",
                stroke_dasharray="5,5"
            ),
            4: RegionStyle(
                stroke="#A16AA9", 
                fill="rgba(161, 106, 169, 0.5)",
                stroke_dasharray="5,5"
            ),
            5: RegionStyle(
                stroke="#A16AA9", 
                fill="rgba(161, 106, 169, 0.5)",
                stroke_dasharray="5,5"
            ),
            6: RegionStyle(
                stroke="#A16AA9", 
                fill="rgba(161, 106, 169, 0.5)",
                stroke_dasharray="5,5"
            ),
            7: RegionStyle(
                stroke="#A16AA9", 
                fill="rgba(161, 106, 169, 0.5)",
                stroke_dasharray="5,5"
            )
        }
        
        self.default_style = RegionStyle(
            stroke="#000000",
            fill="rgba(0,0,0,0.2)",
            stroke_dasharray="5,5"
        )
    
    def get_region_style(self, region_id: int) -> RegionStyle:
        """Get style for specific region or default if not found."""
        return self.region_styles.get(region_id, self.default_style)
    
    def get_default_style(self) -> RegionStyle:
        """Get default style configuration."""
        return self.default_style


class ColorfulStyleConfig(StyleConfig):
    """Colorful style configuration with different colors for each region."""
    
    def __init__(self):
        self.region_styles = {
            1: RegionStyle(stroke="#FF6B6B", fill="rgba(255, 107, 107, 0.3)"),  # Red
            2: RegionStyle(stroke="#4ECDC4", fill="rgba(78, 205, 196, 0.3)"),   # Teal
            3: RegionStyle(stroke="#45B7D1", fill="rgba(69, 183, 209, 0.3)"),   # Blue
            4: RegionStyle(stroke="#96CEB4", fill="rgba(150, 206, 180, 0.3)"),  # Green
            5: RegionStyle(stroke="#FFEAA7", fill="rgba(255, 234, 167, 0.3)"),  # Yellow
            6: RegionStyle(stroke="#DDA0DD", fill="rgba(221, 160, 221, 0.3)"),  # Plum
            7: RegionStyle(stroke="#F39C12", fill="rgba(243, 156, 18, 0.3)")    # Orange
        }
        
        self.default_style = RegionStyle(
            stroke="#95A5A6",
            fill="rgba(149, 165, 166, 0.3)"
        )
    
    def get_region_style(self, region_id: int) -> RegionStyle:
        """Get style for specific region or default if not found."""
        return self.region_styles.get(region_id, self.default_style)
    
    def get_default_style(self) -> RegionStyle:
        """Get default style configuration."""
        return self.default_style


class MinimalStyleConfig(StyleConfig):
    """Minimal style configuration with simple black outlines."""
    
    def __init__(self):
        self.default_style = RegionStyle(
            stroke="#000000",
            fill="rgba(0, 0, 0, 0.1)",
            stroke_width=1,
            font_size=20
        )
    
    def get_region_style(self, region_id: int) -> RegionStyle:
        """All regions use the same minimal style."""
        return self.default_style
    
    def get_default_style(self) -> RegionStyle:
        """Get default style configuration."""
        return self.default_style


class StyleConfigFactory:
    """Factory for creating style configurations."""
    
    @staticmethod
    def create_style_config(style_type: str = "default") -> StyleConfig:
        """Create a style configuration based on type."""
        configs = {
            "default": DefaultStyleConfig,
            "colorful": ColorfulStyleConfig,
            "minimal": MinimalStyleConfig
        }
        
        config_class = configs.get(style_type.lower())
        if not config_class:
            raise ValueError(f"Unknown style type: {style_type}")
        
        return config_class()

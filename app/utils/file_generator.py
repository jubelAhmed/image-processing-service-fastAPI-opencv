"""
Image processing module for facial contour extraction and output generation.
"""

import base64
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Dict,Tuple


class OutputGenerator(ABC):
    """Abstract base class for different output formats."""
    
    @abstractmethod
    def generate(self, image_shape: Tuple[int, int], facial_regions: Dict[str, List]) -> str:
        """Generate output in the specific format."""
        pass


class SVGGenerator(OutputGenerator):
    """SVG output generator for contours."""
    
    def __init__(self):
        self.region_styles = {
            "1": {"stroke": "#FF0000", "fill": "rgba(255,0,0,0.2)"},
            "2": {"stroke": "#00FF00", "fill": "rgba(0,255,0,0.2)"},
            "3": {"stroke": "#0000FF", "fill": "rgba(0,0,255,0.2)"},
            "4": {"stroke": "#FFFF00", "fill": "rgba(255,255,0,0.2)"},
            "5": {"stroke": "#FF00FF", "fill": "rgba(255,0,255,0.2)"},
            "6": {"stroke": "#00FFFF", "fill": "rgba(0,255,255,0.2)"},
            "7": {"stroke": "#FFFFFF", "fill": "rgba(255,255,255,0.2)"}
        }
    
    def generate(self, image_shape: Tuple[int, int], regions: Dict[str, List]) -> str:
        """Generate SVG with regions drawn as contours."""
        svg_root = self._create_svg_root(image_shape)
        self._add_regions_to_svg(svg_root, regions)
        return self._encode_svg(svg_root)
    
    def _create_svg_root(self, image_shape: Tuple[int, int]) -> ET.Element:
        """Create the root SVG element."""
        return ET.Element("svg", {
            "width": str(image_shape[1]),
            "height": str(image_shape[0]),
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {image_shape[1]} {image_shape[0]}"
        })
    
    def _add_regions_to_svg(self, svg_root: ET.Element, regions: Dict[str, List]) -> None:
        """Add regions as paths to the SVG."""
        for region_id, contours in regions.items():
            for contour in contours:
                if not contour:  # Skip empty contours
                    continue
                    
                path_data = self._create_path_data(contour)
                self._create_path_element(svg_root, path_data, region_id)
    
    def _create_path_data(self, contour: List[Dict[str, float]]) -> str:
        """Create SVG path data from contour points."""
        if not contour:
            return ""
        
        path_data = f"M{contour[0]['x']},{contour[0]['y']}"
        for point in contour[1:]:
            path_data += f" L{point['x']},{point['y']}"
        path_data += " Z"  # Close the path
        return path_data
    
    def _create_path_element(self, svg_root: ET.Element, path_data: str, region_id: str) -> None:
        """Create and add a path element to the SVG."""
        style = self.region_styles.get(region_id, {"stroke": "#000000", "fill": "rgba(0,0,0,0.2)"})
        
        ET.SubElement(svg_root, "path", {
            "d": path_data,
            "stroke": style["stroke"],
            "stroke-width": "2",
            "stroke-dasharray": "5,5",
            "fill": style["fill"],
            "class": f"region-{region_id}"
        })
    
    def _encode_svg(self, svg_root: ET.Element) -> str:
        """Convert SVG to base64 encoded string."""
        svg_string = ET.tostring(svg_root, encoding="utf-8").decode("utf-8")
        return base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")


class FileProcessor:
    """Main processor for image extraction and output generation."""
    
    def __init__(self, generator: OutputGenerator):
        self.generator = generator
    
    def create(self, image_shape: Tuple[int, int], regions: Dict[str, List]) -> str:
        """Create output using the configured generator."""
        if not self._validate_inputs(image_shape, regions):
            raise ValueError("Invalid input parameters")
        
        return self.generator.generate(image_shape, regions)
    
    def set_generator(self, generator: OutputGenerator) -> None:
        """Change the output generator."""
        self.generator = generator
    
    def _validate_inputs(self, image_shape: Tuple[int, int], regions: Dict[str, List]) -> bool:
        """Validate input parameters."""
        if not image_shape or len(image_shape) != 2:
            return False
        if image_shape[0] <= 0 or image_shape[1] <= 0:
            return False
        if not isinstance(regions, dict):
            return False
        return True


# Factory for easy instantiation
class GeneratorFactory:
    """Factory for creating output generators."""
    
    @staticmethod
    def create_svg_generator() -> SVGGenerator:
        """Create an SVG generator."""
        return SVGGenerator()
    
    @staticmethod
    def create_processor(generator_type: str = 'svg') -> FileProcessor:
        """Create a processor with the specified generator type."""
        generators = {
            'svg': GeneratorFactory.create_svg_generator(),
        }
        
        generator = generators.get(generator_type.lower())
        if not generator:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        return FileProcessor(generator)


# Usage examples:
# processor = GeneratorFactory.create_processor('svg')
# result = processor.create(image_shape, regions)
#
# Or direct usage:
# svg_gen = SVGGenerator()
# processor = ContourProcessor(svg_gen)
# result = processor.create(image_shape, regions)
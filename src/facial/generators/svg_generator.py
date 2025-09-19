import base64
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import xml.etree.ElementTree as ET
from src.facial.generators.output_generator import OutputGenerator
from src.facial.style_config import StyleConfig, DefaultStyleConfig
from src.facial.face_schema import MaskContours
from src.facial.exceptions import ProcessingError
class SVGGenerator(OutputGenerator):
    """SVG output generator for contours with optional background image and improved style configuration."""

    def __init__(self, style_config: StyleConfig = None):
        """
        Initialize SVG generator with style configuration.
        
        Args:
            style_config: Style configuration instance (defaults to DefaultStyleConfig)
        """
        self.style_config = style_config or DefaultStyleConfig()

    def generate(self, image_shape: Tuple[int, int], contours: MaskContours, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """
        Generate SVG from image shape and contours list.
        
        Args:
            image_shape: Tuple of (height, width)
            contours: Dict where each key contains contour points for that region ID
            processed_image: Optional numpy array of the processed image to use as background
        
        Returns:
            Base64-encoded SVG string
            
        Raises:
            ProcessingError: If SVG generation fails
        """
        try:
            svg_root = self._create_svg_root(image_shape)
            
            # Add background image if provided
            if processed_image is not None:
                self._add_background_image(svg_root, processed_image, image_shape)
            
            self._add_regions_to_svg(svg_root, contours)
            return self._encode_svg(svg_root)
            
        except Exception as e:
            raise ProcessingError(f"Failed to generate SVG: {str(e)}") from e

    def _create_svg_root(self, image_shape: Tuple[int, int]) -> ET.Element:
        """Create the root SVG element with proper dimensions."""
        return ET.Element("svg", {
            "width": str(image_shape[1]),
            "height": str(image_shape[0]),
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {image_shape[1]} {image_shape[0]}"
        })

    def _add_background_image(self, svg_root: ET.Element, processed_image: np.ndarray, 
                            image_shape: Tuple[int, int]) -> None:
        """Add the processed image as background to the SVG."""
        # Encode the processed image to PNG format
        _, img_encoded = cv2.imencode('.png', processed_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Create image element and add it as the first child (background)
        ET.SubElement(svg_root, "image", {
            "href": f"data:image/png;base64,{img_base64}",
            "x": "0",
            "y": "0",
            "width": str(image_shape[1]),
            "height": str(image_shape[0]),
            "class": "background-image"
        })

    def _add_regions_to_svg(self, svg_root: ET.Element, contours: MaskContours) -> None:
        """Add all contours with contour data and their labels to the SVG."""
        for region_id, contour in contours.items():
            if not contour or len(contour) < 3:
                continue

            path_data = self._create_path_data(contour)
            if path_data:
                self._create_path_element(svg_root, path_data, region_id)

                # Draw region number at the centroid
                cx, cy = self._compute_centroid(contour)
                if region_id == 4:
                    cx += 160  # shift right 160 px, tweak as needed
                    cy += 0    # shift down 0 px, tweak as needed

                # Get style for text from style configuration
                style = self.style_config.get_region_style(region_id)
                
                ET.SubElement(svg_root, "text", {
                    "x": str(cx),
                    "y": str(cy),
                    "fill": style.text_color,
                    "font-size": str(style.font_size),
                    "text-anchor": "middle",
                    "dominant-baseline": "middle",
                    "class": f"region-label-{region_id}"
                }).text = str(region_id)

    def _create_path_data(self, contour: List[List[int]]) -> str:
        """
        Create SVG path data from contour points.
        
        Args:
            contour: List of [x, y] coordinate pairs
            
        Returns:
            SVG path data string
        """
        if not contour or len(contour) == 0:
            return ""
        
        # Start with Move command to first point
        path_data = f"M{contour[0][0]},{contour[0][1]}"
        
        # Add Line commands for remaining points
        for point in contour[1:]:
            path_data += f" L{point[0]},{point[1]}"
        
        # Close the path
        path_data += " Z"
        return path_data
    
    def _compute_centroid(self, contour: List[List[int]]) -> Tuple[int, int]:
        """Compute the centroid of a polygon given as a list of points."""
        contour_np = np.array(contour)
        M = cv2.moments(contour_np)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # fallback to average of points
            cx = int(np.mean(contour_np[:, 0]))
            cy = int(np.mean(contour_np[:, 1]))
        return cx, cy

    def _create_path_element(self, svg_root: ET.Element, path_data: str, region_id: int) -> None:
        """Create a path element for a region and add it to the SVG."""
        # Get style for this region using style configuration
        style = self.style_config.get_region_style(region_id)
        
        # Create the path element
        path_attrs = {
            "d": path_data,
            "stroke": style.stroke,
            "stroke-width": str(style.stroke_width),
            "fill": style.fill,
            "class": f"region-{region_id}"
        }
        
        # Add stroke dasharray if specified
        if style.stroke_dasharray:
            path_attrs["stroke-dasharray"] = style.stroke_dasharray
        
        ET.SubElement(svg_root, "path", path_attrs)

    def _encode_svg(self, svg_root: ET.Element) -> str:
        """Encode the SVG element to a base64 string."""
        svg_string = ET.tostring(svg_root, encoding="utf-8").decode("utf-8")
        return base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")


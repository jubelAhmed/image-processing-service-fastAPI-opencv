import base64
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import xml.etree.ElementTree as ET
from app.services.generators.output_generator import OutputGenerator

class SVGGenerator(OutputGenerator):
    """SVG output generator for contours with optional background image."""

    def __init__(self):
        self.region_styles = {
            1: {"stroke": "#9D57A7", "fill": "rgba(161, 106, 169, 0.5)"},
            2: {"stroke": "#A16AA9", "fill": "rgba(161, 106, 169, 0.5)"},
            3: {"stroke": "#A16AA9", "fill": "rgba(161, 106, 169, 0.5)"},
            4: {"stroke": "#A16AA9", "fill": "rgba(161, 106, 169, 0.5)"},
            5: {"stroke": "#A16AA9", "fill": "rgba(161, 106, 169, 0.5)"},
            6: {"stroke": "#A16AA9", "fill": "rgba(161, 106, 169, 0.5)"},
            7: {"stroke": "#A16AA9", "fill": "rgba(161, 106, 169, 0.5)"}
        }

    def generate(self, image_shape: Tuple[int, int], regions: List[List[List[int]]], 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """
        Generate SVG from image shape and regions list.
        
        Args:
            image_shape: Tuple of (height, width)
            regions: List where each index contains contour points [[x1,y1], [x2,y2], ...]
                    for that region ID
            processed_image: Optional numpy array of the processed image to use as background
        
        Returns:
            Base64-encoded SVG string
        """
        svg_root = self._create_svg_root(image_shape)
        
        # Add background image if provided
        if processed_image is not None:
            self._add_background_image(svg_root, processed_image, image_shape)
        
        self._add_regions_to_svg(svg_root, regions)
        return self._encode_svg(svg_root)

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

    # def _add_regions_to_svg(self, svg_root: ET.Element, regions: List[List[List[int]]]) -> None:
    #     """Add all regions with contour data to the SVG."""
    #     for region_id, contour in enumerate(regions):
    #         # Skip empty regions
    #         if not contour or len(contour) == 0:
    #             continue
            
    #         # Skip regions with insufficient points
    #         if len(contour) < 3:
    #             continue
                
    #         path_data = self._create_path_data(contour)
    #         if path_data:
    #             self._create_path_element(svg_root, path_data, region_id)
    def _add_regions_to_svg(self, svg_root: ET.Element, regions: List[List[List[int]]]) -> None:
        """Add all regions with contour data and their labels to the SVG."""
        for region_id, contour in enumerate(regions):
            if not contour or len(contour) < 3:
                continue

            path_data = self._create_path_data(contour)
            if path_data:
                self._create_path_element(svg_root, path_data, region_id)

                # Draw region number at the centroid
                cx, cy = self._compute_centroid(contour)
                if region_id == 4:
                    cx += 160  # shift right 180 px, tweak as needed
                    cy += 0 # shift down 0 px, tweak as needed
                ET.SubElement(svg_root, "text", {
                    "x": str(cx),
                    "y": str(cy),
                    "fill": "white",
                    "font-size": "26",
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
        # Get style for this region, fallback to default if not found
        style = self.region_styles.get(region_id, {"stroke": "#000000", "fill": "rgba(0,0,0,0.2)"})
        
        # Create the path element
        ET.SubElement(svg_root, "path", {
            "d": path_data,
            "stroke": style["stroke"],
            "stroke-width": "2",
            "stroke-dasharray": "5,5",
            "fill": style["fill"],
            "class": f"region-{region_id}"
        })

    def _encode_svg(self, svg_root: ET.Element) -> str:
        """Encode the SVG element to a base64 string."""
        svg_string = ET.tostring(svg_root, encoding="utf-8").decode("utf-8")
        return base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")


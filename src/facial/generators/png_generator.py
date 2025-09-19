"""
PNG output generator for contours with optional background image.
"""

import base64
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.facial.generators.output_generator import OutputGenerator
from src.facial.face_schema import MaskContours
from src.facial.exceptions import ProcessingError


class PNGGenerator(OutputGenerator):
    """PNG output generator for contours with optional background image."""

    def __init__(self):
        self.region_colors = {
            1: (157, 87, 167),    # Purple
            2: (161, 106, 169),   # Light purple
            3: (161, 106, 169),   # Light purple
            4: (161, 106, 169),   # Light purple
            5: (161, 106, 169),   # Light purple
            6: (161, 106, 169),   # Light purple
            7: (161, 106, 169),   # Light purple
        }
        self.default_color = (0, 0, 0)  # Black
        self.contour_thickness = 2

    def generate(self, image_shape: Tuple[int, int], contours: MaskContours, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """
        Generate PNG from image shape and contours list.
        
        Args:
            image_shape: Tuple of (height, width)
            contours: Dict where each key contains contour points for that region ID
            processed_image: Optional numpy array of the processed image to use as background
        
        Returns:
            Base64-encoded PNG string
        """
        try:
            # Create base image
            if processed_image is not None:
                output_image = processed_image.copy()
            else:
                output_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
            
            # Draw contours
            self._draw_contours(output_image, contours)
            
            # Encode to PNG
            _, img_encoded = cv2.imencode('.png', output_image)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            raise ProcessingError(f"Failed to generate PNG: {str(e)}") from e

    def _draw_contours(self, image: np.ndarray, contours: MaskContours) -> None:
        """Draw contours on the image."""
        for region_id, contour in contours.items():
            if not contour or len(contour) < 3:
                continue

            # Get color for this region
            color = self.region_colors.get(region_id, self.default_color)
            
            # Convert contour to numpy array
            contour_np = np.array(contour, dtype=np.int32)
            
            # Draw filled contour
            cv2.fillPoly(image, [contour_np], color)
            
            # Draw contour outline
            cv2.drawContours(image, [contour_np], -1, color, self.contour_thickness)
            
            # Add region number label
            self._add_region_label(image, contour, region_id)

    def _add_region_label(self, image: np.ndarray, contour: List[List[int]], region_id: int) -> None:
        """Add region number label at the centroid of the contour."""
        try:
            # Calculate centroid
            contour_np = np.array(contour)
            M = cv2.moments(contour_np)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback to average of points
                cx = int(np.mean(contour_np[:, 0]))
                cy = int(np.mean(contour_np[:, 1]))
            
            # Adjust position for region 4 (special case)
            if region_id == 4:
                cx += 160
                cy += 0
            
            # Draw text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_color = (255, 255, 255)  # White
            
            # Get text size for centering
            (text_width, text_height), _ = cv2.getTextSize(str(region_id), font, font_scale, thickness)
            
            # Draw text centered
            cv2.putText(image, str(region_id), 
                       (cx - text_width // 2, cy + text_height // 2),
                       font, font_scale, text_color, thickness)
                       
        except Exception as e:
            # If label drawing fails, continue without it
            pass

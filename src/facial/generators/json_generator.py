"""
JSON output generator for contours data.
"""

import json
import base64
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.facial.generators.output_generator import OutputGenerator
from src.facial.face_schema import MaskContours
from src.facial.exceptions import ProcessingError


class JSONGenerator(OutputGenerator):
    """JSON output generator for contours data."""

    def __init__(self):
        self.region_names = {
            1: "forehead",
            2: "left_eye",
            3: "right_eye", 
            4: "nose",
            5: "left_cheek",
            6: "right_cheek",
            7: "mouth"
        }

    def generate(self, image_shape: Tuple[int, int], contours: MaskContours, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """
        Generate JSON from image shape and contours list.
        
        Args:
            image_shape: Tuple of (height, width)
            contours: Dict where each key contains contour points for that region ID
            processed_image: Optional numpy array of the processed image (not used in JSON)
        
        Returns:
            Base64-encoded JSON string
        """
        try:
            # Prepare the output data structure
            output_data = {
                "metadata": {
                    "image_width": image_shape[1],
                    "image_height": image_shape[0],
                    "total_regions": len(contours),
                    "format": "json"
                },
                "regions": self._process_contours(contours),
                "statistics": self._calculate_statistics(contours, image_shape)
            }
            
            # Convert to JSON string
            json_string = json.dumps(output_data, indent=2)
            
            # Encode to base64
            json_base64 = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')
            
            return json_base64
            
        except Exception as e:
            raise ProcessingError(f"Failed to generate JSON: {str(e)}") from e

    def _process_contours(self, contours: MaskContours) -> Dict[str, Any]:
        """Process contours into structured data."""
        regions_data = {}
        
        for region_id, contour in contours.items():
            if not contour or len(contour) < 3:
                continue
                
            region_name = self.region_names.get(region_id, f"region_{region_id}")
            
            # Calculate centroid
            centroid = self._calculate_centroid(contour)
            
            # Calculate area (approximate)
            area = self._calculate_area(contour)
            
            regions_data[region_name] = {
                "id": region_id,
                "contour_points": contour,
                "centroid": centroid,
                "area": area,
                "point_count": len(contour),
                "bounding_box": self._calculate_bounding_box(contour)
            }
            
        return regions_data

    def _calculate_centroid(self, contour: List[List[int]]) -> Dict[str, float]:
        """Calculate centroid of contour points."""
        if not contour:
            return {"x": 0.0, "y": 0.0}
            
        x_coords = [point[0] for point in contour]
        y_coords = [point[1] for point in contour]
        
        return {
            "x": sum(x_coords) / len(x_coords),
            "y": sum(y_coords) / len(y_coords)
        }

    def _calculate_area(self, contour: List[List[int]]) -> float:
        """Calculate approximate area of contour using shoelace formula."""
        if len(contour) < 3:
            return 0.0
            
        n = len(contour)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += contour[i][0] * contour[j][1]
            area -= contour[j][0] * contour[i][1]
            
        return abs(area) / 2.0

    def _calculate_bounding_box(self, contour: List[List[int]]) -> Dict[str, int]:
        """Calculate bounding box of contour."""
        if not contour:
            return {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0}
            
        x_coords = [point[0] for point in contour]
        y_coords = [point[1] for point in contour]
        
        return {
            "x_min": min(x_coords),
            "y_min": min(y_coords),
            "x_max": max(x_coords),
            "y_max": max(y_coords),
            "width": max(x_coords) - min(x_coords),
            "height": max(y_coords) - min(y_coords)
        }

    def _calculate_statistics(self, contours: MaskContours, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate overall statistics."""
        total_points = sum(len(contour) for contour in contours.values() if contour)
        total_area = sum(self._calculate_area(contour) for contour in contours.values() if contour)
        image_area = image_shape[0] * image_shape[1]
        
        return {
            "total_contour_points": total_points,
            "total_region_area": total_area,
            "image_area": image_area,
            "coverage_percentage": (total_area / image_area * 100) if image_area > 0 else 0,
            "average_points_per_region": total_points / len(contours) if contours else 0
        }

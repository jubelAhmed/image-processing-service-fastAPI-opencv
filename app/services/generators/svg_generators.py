import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET
from app.services.generators.output_generator import OutputGenerator

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
        svg_root = self._create_svg_root(image_shape)
        self._add_regions_to_svg(svg_root, regions)
        return self._encode_svg(svg_root)

    def _create_svg_root(self, image_shape: Tuple[int, int]) -> ET.Element:
        return ET.Element("svg", {
            "width": str(image_shape[1]),
            "height": str(image_shape[0]),
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {image_shape[1]} {image_shape[0]}"
        })

    def _add_regions_to_svg(self, svg_root: ET.Element, regions: Dict[str, List]) -> None:
        for region_id, contours in regions.items():
            for contour in contours:
                if not contour:
                    continue
                path_data = self._create_path_data(contour)
                self._create_path_element(svg_root, path_data, region_id)

    def _create_path_data(self, contour: List[Dict[str, float]]) -> str:
        if not contour:
            return ""
        path_data = f"M{contour[0]['x']},{contour[0]['y']}"
        for point in contour[1:]:
            path_data += f" L{point['x']},{point['y']}"
        path_data += " Z"
        return path_data

    def _create_path_element(self, svg_root: ET.Element, path_data: str, region_id: str) -> None:
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
        svg_string = ET.tostring(svg_root, encoding="utf-8").decode("utf-8")
        return base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")


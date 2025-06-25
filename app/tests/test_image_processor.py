"""
Test file for the image processor module.
"""
import base64
import unittest
from unittest.mock import patch, MagicMock
import cv2
import numpy as np

from app.core.image_processor import (
    decode_image, 
    decode_segmentation_map, 
    extract_facial_regions,
    rotate_image, 
    generate_svg_mask
)
from app.schemas.facial_processing import LandmarkPoint

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        # Create a simple test image
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw a face-like shape
        cv2.circle(test_img, (150, 150), 100, (255, 255, 255), -1)
        # Encode to base64
        _, buffer = cv2.imencode('.png', test_img)
        self.test_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create a test segmentation map
        seg_map = np.zeros((300, 300), dtype=np.uint8)
        # Fill with different region IDs
        cv2.circle(seg_map, (150, 150), 100, 1, -1)  # Region 1
        cv2.circle(seg_map, (150, 150), 50, 2, -1)   # Region 2
        _, buffer = cv2.imencode('.png', seg_map)
        self.test_segmap_base64 = base64.b64encode(buffer).decode('utf-8')
    
    def test_decode_image(self):
        """Test image decoding from base64."""
        img = decode_image(self.test_image_base64)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (300, 300, 3))
    
    def test_decode_segmentation_map(self):
        """Test segmentation map decoding from base64."""
        seg_map = decode_segmentation_map(self.test_segmap_base64)
        self.assertIsNotNone(seg_map)
        self.assertEqual(seg_map.shape, (300, 300))
    
    @patch('app.core.image_processor.optimize_contour_extraction')
    def test_extract_facial_regions(self, mock_optimize):
        """Test facial region extraction."""
        # Set up mock
        mock_optimize.return_value = [np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])]
        
        # Create test landmarks
        test_landmarks = [
            LandmarkPoint(x=100, y=100),
            LandmarkPoint(x=150, y=100),
            LandmarkPoint(x=200, y=100),
            LandmarkPoint(x=100, y=150),
            LandmarkPoint(x=150, y=150),
            LandmarkPoint(x=200, y=150)
        ]
        
        # Run the test
        img = decode_image(self.test_image_base64)
        seg_map = decode_segmentation_map(self.test_segmap_base64)
        regions = extract_facial_regions(img, seg_map, test_landmarks)
        
        # Check results
        self.assertIsInstance(regions, dict)
        self.assertTrue(len(regions) > 0)
    
    def test_rotate_image(self):
        """Test image rotation."""
        img = decode_image(self.test_image_base64)
        rotated = rotate_image(img, 45)
        
        # The rotated image should have the same shape
        self.assertEqual(img.shape, rotated.shape)
    
    def test_generate_svg_mask(self):
        """Test SVG mask generation."""
        # Create mock contours
        contours = {
            'region1': [np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])],
            'region2': [np.array([[[30, 30]], [[40, 30]], [[40, 40]], [[30, 40]]])]
        }
        
        svg = generate_svg_mask(contours, (300, 300))
        
        # Check that we got valid SVG
        self.assertIsInstance(svg, str)
        self.assertTrue(svg.startswith('<?xml'))
        self.assertTrue('svg' in svg)
        self.assertTrue('path' in svg)

if __name__ == '__main__':
    unittest.main()

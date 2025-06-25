from app.core.facial_processing.face_segmentation_config import SegmentationConfig
from app.core.facial_processing.face_alignment_utils import rotate_and_crop_face
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.services.image_generator import ImageGenerator
from app.schemas.face_schema import LandmarkPoint
from app.utils.base64_utils import decode_image, decode_segmentation_map
from app.utils.logging import logger
class FacialSegmentationProcessor:
    """Advanced face region subdivider for segmented face images."""
    
    def __init__(self, config: SegmentationConfig = None):
        self.config = config or SegmentationConfig()
    
    # ========== CORE PROCESSING METHODS ==========

    async def process_image(self, image_base64: str, segmentation_map_base64: str, landmarks: List[LandmarkPoint]) -> str:
        
        image = decode_image(image_base64)
        segmentation_map = decode_segmentation_map(segmentation_map_base64)
        
        # Check if a face is detected
        if not landmarks or len(landmarks) == 0:
            raise ValueError("No face detected in the image")
    
        image_shape, contours, processed_image = self.process_face_regions(image, segmentation_map, landmarks)
        
        # Debug logging
        logger.debug(f"Image shape: {image_shape}")
        logger.debug(f"Regions type: {type(contours)}, length: {len(contours)}")
        logger.debug(f"Regions structure: {[len(r) if r else 0 for r in contours]}")
        
        # Save or display the result
        # cv2.imwrite("result_image_2.png", result_image)
        
        # Step 2: Create a FileProcessor with that generator
        processor = ImageGenerator()

        # Step 3: Generate the image (Base64-encoded SVG)
        try:
            logger.info("About to call processor.create() with:")
            logger.info(f"image_shape: {image_shape} (type: {type(image_shape)})")
            logger.info(f" contours: {type(contours)} with {len(contours)} items")
            
            svg_base64 = processor.create(image_shape, contours, processed_image)
            logger.info("SVG generation successful")
        except Exception as e:
            logger.error(f"Error in SVG generation: {e}")
            logger.error(f"Image shape passed: {image_shape}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
     
        with open("new_output.svg", "wb") as f:
            import base64
            f.write(base64.b64decode(svg_base64))
        
        return svg_base64, contours
        
    def process_face_regions(self, original_image: np.ndarray, 
                           segmentation_map: np.ndarray, 
                           landmarks_list: List[LandmarkPoint]) -> Tuple[Tuple[int, int], List, np.ndarray]:
        """
        Main processing method that subdivides face regions and applies overlays.
        
        Args:
            original_image: Input face image (H, W, 3)
            segmentation_map: Segmentation mask (H, W, 3)
            landmarks_list: List of facial landmarks with 'x', 'y' coordinates
            
        Returns:
            Tuple containing:
            - image_shape: Shape of the cropped image (height, width)
            - regions: List where each index contains contour points for that region
            - result_image: Result image with region overlays and numbers
        """
        # Prepare images
        original_image, segmentation_map = self._prepare_images(
            original_image, segmentation_map)
        
        # Apply rotation and cropping
        cropped_image, cropped_seg_map, cropped_landmarks = self._apply_rotation_and_crop(
            original_image, segmentation_map, landmarks_list)
        
        # Process regions and get both result image and region data
        contours = self._process_subdivided_regions(
            cropped_image, cropped_seg_map, landmarks_list)
        
        return cropped_image.shape[:2], contours, cropped_image
    
    def _prepare_images(self, original_image: np.ndarray, 
                       segmentation_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and resize images to match dimensions."""
        height, width = original_image.shape[:2]
        
        if segmentation_map.shape[:2] != (height, width):
            segmentation_map = cv2.resize(
                segmentation_map, (width, height), 
                interpolation=cv2.INTER_NEAREST)
        
        return original_image, segmentation_map
    
    def _apply_rotation_and_crop(self, image: np.ndarray, 
                               segmentation: np.ndarray, 
                               landmarks_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply face alignment rotation and cropping."""
        img_h, img_w = image.shape[:2]
        
        # Convert landmarks to numpy array
        landmarks_np = np.array([
            [pt.x, pt.y] for pt in landmarks_list
        ], dtype=np.float32)
        
        # Apply rotation and cropping
        cropped_image, cropped_landmarks, rot_mat, crop_box = rotate_and_crop_face(
            image, landmarks_np, return_crop_box=True)
        
        # Apply same transformation to segmentation map
        rotated_seg_map = cv2.warpAffine(
            segmentation, rot_mat, (img_w, img_h), 
            flags=cv2.INTER_NEAREST)
        
        x1, y1, x2, y2 = crop_box
        cropped_seg_map = rotated_seg_map[y1:y2, x1:x2]
        
        return cropped_image, cropped_seg_map, cropped_landmarks
    
    # ========== REGION ANALYSIS METHODS ==========
    
    def _get_unique_colors(self, segmentation_map: np.ndarray) -> np.ndarray:
        """Extract unique colors from segmentation map."""
        pixels = segmentation_map.reshape(-1, 3)
        return np.unique(pixels, axis=0)
    
    def _create_clean_mask(self, segmentation_map: np.ndarray, 
                          target_color: np.ndarray) -> np.ndarray:
        """Create clean binary mask for specific color with morphological operations."""
        # Create initial mask
        mask = np.all(segmentation_map == target_color, axis=2).astype(np.uint8) * 255
        
        # Apply morphological operations for cleaning
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small gaps, remove noise, then smooth
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smooth, iterations=1)
        
        return mask
    
    def _analyze_face_contour(self, mask: np.ndarray) -> Tuple[Optional[np.ndarray], 
                                                             Optional[Tuple[int, int, int, int]], 
                                                             Optional[Tuple[int, int]]]:
        """Analyze face contour and extract key properties."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        
        main_contour = max(contours, key=cv2.contourArea)
        bbox = cv2.boundingRect(main_contour)
        
        # Calculate centroid
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = bbox
            cx, cy = x + w // 2, y + h // 2
        
        return main_contour, bbox, (cx, cy)
    
    def _calculate_face_boundaries(self, face_mask: np.ndarray, 
                                 landmarks_list: List[Dict]) -> Dict:
        """Calculate comprehensive face boundaries and landmarks."""
        contour, bbox, centroid = self._analyze_face_contour(face_mask)
        if contour is None:
            raise ValueError("No face contour found")
        
        x, y, w, h = bbox
        cx, cy = centroid
        
        boundaries = {
            'bbox': bbox,
            'centroid': centroid,
            'contour': contour,
            'top': y,
            'bottom': y + h,
            'left': x,
            'right': x + w,
            'width': w,
            'height': h
        }
        
        # Add landmark-based boundaries
        if landmarks_list:
            landmarks_array = np.array([
                [pt.x, pt.y] for pt in landmarks_list
            ], dtype=np.float32)
            
            if len(landmarks_array) >= 48:  # Ensure we have eye landmarks
                left_eye_points = landmarks_array[self.config.left_eye_indices]
                right_eye_points = landmarks_array[self.config.right_eye_indices]
                
                boundaries['left_eye_center'] = left_eye_points.mean(axis=0).astype(int)
                boundaries['right_eye_center'] = right_eye_points.mean(axis=0).astype(int)
                boundaries['eye_level'] = int(
                    (boundaries['left_eye_center'][1] + boundaries['right_eye_center'][1]) / 2
                )
        
        return boundaries
    
    # ========== REGION SUBDIVISION METHODS ==========
    
    def _find_forehead_region(self, face_mask: np.ndarray, 
                            boundaries: Dict) -> np.ndarray:
        """Extract forehead region from face mask."""
        bbox = boundaries['bbox']
        x, y, w, h = bbox
        
        forehead_height = int(h * self.config.forehead_height_ratio)
        
        # Create temporary mask for forehead area
        temp_mask = np.zeros_like(face_mask)
        temp_mask[y:y + forehead_height, x:x + w] = 255
        
        # Extract forehead region
        forehead_mask = cv2.bitwise_and(face_mask, temp_mask)
        
        # Clean up and get largest contour
        contours, _ = cv2.findContours(forehead_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            forehead_mask = np.zeros_like(face_mask)
            cv2.fillPoly(forehead_mask, [largest], 255)
        
        return forehead_mask
    
    def _find_ear_regions(self, face_mask: np.ndarray, 
                        boundaries: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract left and right ear regions."""
        bbox = boundaries['bbox']
        x, y, w, h = bbox
        
        left_ear_mask = np.zeros_like(face_mask)
        right_ear_mask = np.zeros_like(face_mask)
        
        # Define ear search area
        ear_search_start = y + int(h * self.config.ear_search_start_ratio)
        ear_search_end = y + int(h * self.config.ear_search_end_ratio)
        
        # Analyze face profiles to find ear regions
        face_profiles = self._analyze_face_profiles(
            face_mask, ear_search_start, ear_search_end)
        
        if not face_profiles:
            return left_ear_mask, right_ear_mask
        
        # Calculate average face boundaries
        avg_left = int(np.mean([p[0] for p in face_profiles]))
        avg_right = int(np.mean([p[1] for p in face_profiles]))
        avg_width = int(np.mean([p[2] for p in face_profiles]))
        
        # Calculate central face region
        central_width = int(avg_width * self.config.face_width_center_ratio * 1.39)
        central_margin = (avg_width - central_width) // 2
        central_left = avg_left + central_margin
        central_right = avg_right - central_margin
        
        # Extract ear pixels
        for row in range(ear_search_start, ear_search_end):
            if row >= face_mask.shape[0]:
                break
                
            profile = face_mask[row, :]
            face_pixels = np.where(profile > 0)[0]
            
            if len(face_pixels) > 0:
                row_left, row_right = face_pixels[0], face_pixels[-1]
                
                # Left ear
                if (row_left < central_left and 
                    (central_left - row_left) >= self.config.ear_min_width):
                    left_ear_mask[row, row_left:min(central_left, row_right + 1)] = 255
                
                # Right ear
                if (row_right > central_right and 
                    (row_right - central_right) >= self.config.ear_min_width):
                    right_ear_mask[row, max(central_right, row_left):row_right + 1] = 255
        
        # Clean ear masks
        left_ear_mask = self._clean_ear_mask(left_ear_mask)
        right_ear_mask = self._clean_ear_mask(right_ear_mask)
        
        return left_ear_mask, right_ear_mask
    
    def _analyze_face_profiles(self, face_mask: np.ndarray, 
                             start_row: int, end_row: int) -> List[Tuple[int, int, int]]:
        """Analyze horizontal face profiles to find boundaries."""
        face_profiles = []
        
        for row in range(start_row, end_row, 5):
            if row < face_mask.shape[0]:
                profile = face_mask[row, :]
                face_pixels = np.where(profile > 0)[0]
                if len(face_pixels) > 0:
                    face_profiles.append((
                        face_pixels[0],    # leftmost
                        face_pixels[-1],   # rightmost
                        len(face_pixels)   # width
                    ))
        
        return face_profiles
    
    def _clean_ear_mask(self, ear_mask: np.ndarray) -> np.ndarray:
        """Clean ear mask and keep only largest valid contour."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ear_mask = cv2.morphologyEx(ear_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(ear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > self.config.min_region_area:
                new_mask = np.zeros_like(ear_mask)
                cv2.fillPoly(new_mask, [largest], 255)
                return new_mask
        
        return np.zeros_like(ear_mask)
    
    def _create_chin_region(self, face_mask: np.ndarray, 
                          boundaries: Dict) -> np.ndarray:
        """Create chin region from lower portion of face."""
        bbox = boundaries['bbox']
        x, y, w, h = bbox
        
        # Chin starts from mouth level (66% down the face)
        mouth_level = y + int(h * 0.66)
        chin_mask = np.zeros_like(face_mask)
        
        # Extract chin pixels
        for row in range(max(mouth_level, y), y + h):
            if row >= face_mask.shape[0]:
                break
            face_pixels = np.where(face_mask[row, :] > 0)[0]
            if len(face_pixels) > 0:
                chin_mask[row, face_pixels] = 255
        
        return self._clean_region_mask(chin_mask)
    
    def _clean_region_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean region mask with morphological operations."""
        if cv2.countNonZero(mask) == 0:
            return mask
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_kernel_size, self.config.morphology_kernel_size)
        )
        
        # Close gaps and remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cleaned_mask = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) >= self.config.min_region_area:
                    cv2.drawContours(cleaned_mask, [contour], -1, 255, -1)
            return cleaned_mask
        
        return mask
    
    # ========== SUBDIVISION ORCHESTRATION ==========
    
    def _subdivide_main_face_region(self, region1_mask: np.ndarray, 
                                  landmarks_list: List[Dict]) -> Tuple[Dict, np.ndarray]:
        """Subdivide main face region into sub-regions."""
        boundaries = self._calculate_face_boundaries(region1_mask, landmarks_list)
        
        # Find sub-regions
        forehead_mask = self._find_forehead_region(region1_mask, boundaries)
        left_ear_mask, right_ear_mask = self._find_ear_regions(region1_mask, boundaries)
        # left_under_eye_mask, right_under_eye_mask =  self._find_under_eye_regions(region1_mask, boundaries)
        
        # Create remaining face mask
        main_face_mask = region1_mask.copy()
        main_face_mask = cv2.subtract(main_face_mask, forehead_mask)
        main_face_mask = cv2.subtract(main_face_mask, left_ear_mask)
        main_face_mask = cv2.subtract(main_face_mask, right_ear_mask)
        
        # Create chin region
        chin_mask = self._create_chin_region(main_face_mask, boundaries)
        
        # Final cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        main_face_mask = cv2.morphologyEx(main_face_mask, cv2.MORPH_OPEN, kernel)
        
        subdivisions = {
            'forehead': {'mask': forehead_mask, 'region': 1},
            'left_ear': {'mask': left_ear_mask, 'region': 7},
            'right_ear': {'mask': right_ear_mask, 'region': 6},
            'chin': {'mask': chin_mask, 'region': 4}
        }
        
        return subdivisions, main_face_mask
    
    # ========== HELPER METHODS FOR REGION DATA EXTRACTION ==========
    
    def _extract_contour_points(self, mask: np.ndarray) -> List[List[int]]:
        """Extract contour points from mask for SVG generation."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Convert contour points to list format
        points = []
        for point in largest_contour:
            x, y = point[0]
            points.append([int(x), int(y)])
        
        return points
    
    def _get_region_centroid(self, mask: np.ndarray) -> Optional[List[int]]:
        """Get centroid of a region mask."""
        if cv2.countNonZero(mask) == 0:
            return None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return [cx, cy]
    
    # ========== VISUALIZATION METHODS ==========
    
    def _apply_region_overlay(self, image: np.ndarray, mask: np.ndarray, 
                            color: Tuple[int, int, int], is_dotted: bool = False,
                            shift_cm: float = 0, dpi: int = 96, 
                            scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
        """Apply colored overlay to image based on mask."""
        original_h, original_w = mask.shape
        
        # Apply transformations
        if scale_x != 1.0 or scale_y != 1.0:
            mask = self._scale_mask(mask, scale_x, scale_y, original_w, original_h)
        
        if shift_cm > 0:
            mask = self._shift_mask(mask, shift_cm, dpi)
        
        if cv2.countNonZero(mask) == 0:
            return image
        
        # Apply overlay
        if is_dotted:
            overlay = self._create_dotted_pattern(mask, color)
            mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
            result = (image * (1 - mask_3d * self.config.overlay_alpha) + 
                     overlay * mask_3d * self.config.overlay_alpha)
        else:
            overlay = np.zeros_like(image)
            overlay[mask > 0] = color
            result = cv2.addWeighted(
                image, 1 - self.config.overlay_alpha, 
                overlay, self.config.overlay_alpha + 0.2, 0)
        
        return result.astype(np.uint8)
    
    def draw_u_shaped_eye_mask(self, width: int, height: int, position: str = 'center') -> np.ndarray:
        """Draw a small U-shaped curve positioned under the eye.

        Args:
            width (int): Width of the mask.
            height (int): Height of the mask.
            position (str): Position of the curve: 'left', 'right', or 'center'.

        Returns:
            np.ndarray: Mask with U-shaped curve.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        center_y = int(height * 0.5)  # vertical position of the curve

        # Determine center_x based on position
        if position == 'left':
            center_x = width // 3
        elif position == 'right':
            # center_x = 3 * width // 4
            center_x = int(width * 0.68)  # shift a bit left from 3/4 (which is 0.75)
        else:  # 'center' or any other value
            center_x = width // 2

        axes = (width // 16, height // 20)  # smaller U-curve
        angle = 0
        startAngle = 0
        endAngle = 180

        # Draw the bottom half ellipse (U shape)
        cv2.ellipse(mask, (center_x, center_y), axes, angle, startAngle, endAngle, 255, -1)

        return mask
    
    def _scale_mask(self, mask: np.ndarray, scale_x: float, scale_y: float,
                   original_w: int, original_h: int) -> np.ndarray:
        """Scale mask and center it."""
        new_w = int(mask.shape[1] * scale_x)
        new_h = int(mask.shape[0] * scale_y)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        padded_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        # Center the resized mask
        center_x, center_y = original_w // 2, original_h // 2
        start_x = center_x - new_w // 2
        start_y = center_y - new_h // 2
        
        # Calculate paste regions
        paste_x1 = max(0, start_x)
        paste_y1 = max(0, start_y)
        paste_x2 = min(paste_x1 + new_w, original_w)
        paste_y2 = min(paste_y1 + new_h, original_h)
        
        src_x1 = max(0, -start_x)
        src_y1 = max(0, -start_y)
        src_x2 = src_x1 + (paste_x2 - paste_x1)
        src_y2 = src_y1 + (paste_y2 - paste_y1)
        
        padded_mask[paste_y1:paste_y2, paste_x1:paste_x2] = \
            resized_mask[src_y1:src_y2, src_x1:src_x2]
        
        return padded_mask
    
    def _shift_mask(self, mask: np.ndarray, shift_cm: float, dpi: int) -> np.ndarray:
        """Shift mask vertically."""
        pixels_per_cm = dpi / 2.54
        shift_pixels = int(shift_cm * pixels_per_cm)
        M = np.float32([[1, 0, 0], [0, 1, shift_pixels]])
        return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    def _create_dotted_pattern(self, mask: np.ndarray, 
                             color: Tuple[int, int, int]) -> np.ndarray:
        """Create dotted overlay pattern."""
        height, width = mask.shape
        dotted_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(0, height, self.config.dot_spacing):
            for x in range(0, width, self.config.dot_spacing):
                if mask[y, x] > 0:
                    cv2.circle(dotted_overlay, (x, y), self.config.dot_radius, color, -1)
        
        return dotted_overlay
    
    def _draw_region_number(self, image: np.ndarray, mask: np.ndarray, 
                          region_number: int, color: Tuple[int, int, int] = (255, 255, 255),
                          offset_x: int = 0, shift_cm: float = 0, dpi: int = 96) -> np.ndarray:
        """Draw region number on mask centroid."""
        if cv2.countNonZero(mask) == 0:
            return image
        
        # Apply shift if needed
        if shift_cm > 0:
            mask = self._shift_mask(mask, shift_cm, dpi)
        
        # Find centroid
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return image
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text = str(region_number)
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = cx - text_width // 2 + offset_x
        text_y = cy + text_height // 2
        
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return image
    
    
    def _process_subdivided_regions(self, original_image: np.ndarray, 
                                  segmentation_map: np.ndarray, 
                                  landmarks_list: List[Dict]) -> Tuple[np.ndarray, List]:
        """Process all subdivided regions and apply overlays."""
        unique_colors = self._get_unique_colors(segmentation_map)
        result_image = original_image.copy()
        contours = []
        
        print(f"🎨 Found {len(unique_colors)} unique colors in segmentation map")
        
        # Find main face region (first non-black color)
        region_1_color = None
        for color in unique_colors:
            if not np.all(color == [0, 0, 0]):
                region_1_color = color
                break
        
        if region_1_color is None:
            print("❌ Region 1 (main face) not found")
            return result_image, contours
        
        # Process main face subdivisions
        print(f"🔍 Processing Region 1 (main face)")
        region1_mask = self._create_clean_mask(segmentation_map, region_1_color)
        subdivisions, main_face_mask = self._subdivide_main_face_region(region1_mask, landmarks_list)
        
        # Apply overlays for subdivisions and extract contour data
        for region_name, mask_obj in subdivisions.items():
            mask = mask_obj['mask']
            region_id = mask_obj['region']
            
            if cv2.countNonZero(mask) < self.config.min_region_area:
                logger.info(f"Skipping {region_name} (region {region_id}): insufficient area")
                continue
            
            # Apply overlay
            result_image = self._apply_region_overlay(
                result_image, mask, self.config.default_color)
            
            # Draw region number
            offset_x = 150 if region_name == 'chin' else 0
            result_image = self._draw_region_number(
                result_image, mask, region_id, offset_x=offset_x)
            
            # Extract contour points
            contour_points = self._extract_contour_points(mask)
            logger.info(f"Extracted {len(contour_points)} points for {region_name} (region {region_id})")
            
            if contour_points:
                # Ensure we have enough slots in the contours list
                while len(contours) <= region_id:
                    contours.append([])
                contours[region_id] = contour_points
                logger.info(f"Added {region_name} to contours[{region_id}]")
        
        # Process additional regions from segmentation map
        result_image, additional_regions = self._process_additional_regions(
            result_image, segmentation_map, unique_colors, region_1_color)
        
        # Merge additional regions
        for region_id, contour_points in additional_regions:
            # Ensure we have enough slots in the contours list
            while len(contours) <= region_id:
                contours.append([])
            contours[region_id] = contour_points
            logger.info(f"Added additional region {region_id} with {len(contour_points)} points")
        
        logger.info(f"Final contours list length: {len(contours)}")
        logger.info(f"Non-empty contours: {[i for i, c in enumerate(contours) if c]}")
        
        return contours
    
    def _process_additional_regions(self, result_image: np.ndarray, 
                                  segmentation_map: np.ndarray,
                                  unique_colors: np.ndarray, 
                                  region_1_color: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, List]]]:
        """Process additional regions beyond the main face."""
        additional_regions = {
            4: {"name": "nose", "region_number": 5, "visible": True},
            5: {"name": "lips", "region_number": 8, "visible": False},
            7: {"name": "inner_mouth", "region_number": 9, "visible": False},
        }
        
        regions_data = []
        
        potential_colors = [
            color for color in unique_colors 
            if not np.all(color == [0, 0, 0]) and not np.array_equal(color, region_1_color)
        ]
        
        for i, color in enumerate(potential_colors):
            region_mask = self._create_clean_mask(segmentation_map, color)
            area = cv2.countNonZero(region_mask)
            print("Analyzing additional region", i + 1, "with area:", area)
            if area < self.config.min_region_area:
                continue
            print(f"🔍 Processing additional region {i + 1} with color {color}")
            
            if i == 4:  # Nose region
                result_image = self._apply_region_overlay(
                    result_image, region_mask, self.config.default_color)
                
                if additional_regions.get(i, {}).get("visible", False):
                    region_id = additional_regions[i]["region_number"]
                    result_image = self._draw_region_number(
                        result_image, region_mask, region_id)
                    
                    # Extract region data for SVG - only contour points
                    contour_points = self._extract_contour_points(region_mask)
                    
                    if contour_points:
                        regions_data.append((region_id, contour_points))
            
            elif i in [2, 3]:  # Other regions with scaling and shifting
                result_image = self._apply_region_overlay(
                    result_image, region_mask, self.config.default_color,
                    shift_cm=1.2, scale_x=1.2)
                
                result_image = self._draw_region_number(
                    result_image, region_mask, i, shift_cm=1.2)
                
                # Extract region data for SVG (apply same transformations)
                # Step 1: Get dimensions of the original region_mask
                h, w = region_mask.shape

                if i == 2:
                    region_mask = self.draw_u_shaped_eye_mask(w, h, position='left')
                else:
                    region_mask = self.draw_u_shaped_eye_mask(w, h, position='right')

                # Step 3: Apply your existing transformations
                # transformed_mask = self._scale_mask(region_mask, 1.2, 1.0, w, h)
                # transformed_mask = self._shift_mask(transformed_mask, 1.2, 96)
                
                contour_points = self._extract_contour_points(region_mask)
                
                if contour_points:
                    regions_data.append((i, contour_points))
        
        return result_image, regions_data
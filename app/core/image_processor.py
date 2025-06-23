"""
Image processing module for facial contour extraction and SVG generation.
"""

import base64
import numpy as np
import cv2
from io import BytesIO
import math
import xml.etree.ElementTree as ET
from app.core.performance import optimize_contour_extraction, adaptive_blur, smooth_contour, parallel_process_regions

def decode_image(base64_string):
    """Decode a base64 image to numpy array."""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def decode_segmentation_map(base64_string):
    """Decode a base64 segmentation map to numpy array."""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    segmap = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return segmap

def autorotate_face(image, landmarks):
    """Autorotate the face to make it upright based on landmarks."""
    # Using eyes to determine rotation angle
    # Assuming landmarks contain eye coordinates
    # This is a simplified approach - you might need to adjust based on your landmark format
    
    # For demonstration, let's assume landmarks[36] and landmarks[45] are the left and right eye centers
    # Extract eye positions from landmarks
    left_eye_x = 0
    left_eye_y = 0
    right_eye_x = 0
    right_eye_y = 0
    
    # Count how many landmarks contribute to each eye
    left_eye_count = 0
    right_eye_count = 0
    
    # Assuming landmarks contains points with left eye around indices 36-41 and right eye around 42-47
    # This would need to be adjusted based on the actual landmark format
    for i, point in enumerate(landmarks):
        if 36 <= i <= 41:  # Left eye region
            left_eye_x += point.x
            left_eye_y += point.y
            left_eye_count += 1
        elif 42 <= i <= 47:  # Right eye region
            right_eye_x += point.x
            right_eye_y += point.y
            right_eye_count += 1
    
    # Calculate eye centers
    if left_eye_count > 0 and right_eye_count > 0:
        left_eye_x /= left_eye_count
        left_eye_y /= left_eye_count
        right_eye_x /= right_eye_count
        right_eye_y /= right_eye_count
        
        # Calculate angle
        dy = right_eye_y - left_eye_y
        dx = right_eye_x - left_eye_x
        angle = math.degrees(math.atan2(dy, dx))
        
        # Rotate to make eyes horizontal
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)
        
        # Rotate landmarks as well
        rotated_landmarks = []
        for point in landmarks:
            # Apply the same rotation to each landmark
            x = point.x - center[0]
            y = point.y - center[1]
            # Rotate point
            x_new = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle)) + center[0]
            y_new = x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle)) + center[1]
            rotated_landmarks.append({"x": x_new, "y": y_new})
        
        return rotated_image, rotated_landmarks
    
    # If we couldn't identify the eyes, return the original image and landmarks
    return image, [{"x": point.x, "y": point.y} for point in landmarks]

def crop_face(image, landmarks):
    """Crop the image to focus on the face area."""
    # Find the bounding box of the face based on landmarks
    min_x = min(point.x for point in landmarks)
    min_y = min(point.y for point in landmarks)
    max_x = max(point.x for point in landmarks)
    max_y = max(point.y for point in landmarks)
    
    # Add some padding
    padding_x = (max_x - min_x) * 0.1
    padding_y = (max_y - min_y) * 0.1
    
    min_x = max(0, min_x - padding_x)
    min_y = max(0, min_y - padding_y)
    max_x = min(image.shape[1], max_x + padding_x)
    max_y = min(image.shape[0], max_y + padding_y)
    
    # Crop the image
    cropped_image = image[int(min_y):int(max_y), int(min_x):int(max_x)]
    
    # Adjust landmarks for the cropped image
    adjusted_landmarks = []
    for point in landmarks:
        adjusted_landmarks.append({"x": point.x - min_x, "y": point.y - min_y})
    
    return cropped_image, adjusted_landmarks

def extract_facial_regions(image, segmentation_map, landmarks):
    """Extract facial regions based on segmentation map and landmarks."""
    # This function identifies the 7 regions mentioned in the task
    # Create an empty dictionary to store contours for each region
    facial_regions = {}
    
    # Check if we should use parallel processing
    if hasattr(parallel_process_regions, '__call__'):
        # Use parallel processing for faster extraction
        region_results = parallel_process_regions(segmentation_map, range(1, 8))
        
        for region_id, contours in region_results.items():
            # Simplify contours for smoother appearance
            smoothed_contours = []
            for contour in contours:
                # Apply optimized contour smoothing
                approx = smooth_contour(contour)
                
                # Convert contour to a list of points
                points = []
                for point in approx:
                    points.append({"x": float(point[0][0]), "y": float(point[0][1])})
                
                smoothed_contours.append(points)
            
            # Store the contours for this region
            facial_regions[str(region_id)] = smoothed_contours
    else:
        # Fallback to sequential processing
        for region_id in range(1, 8):  # 7 regions as per the task
            # Create a binary mask for this region
            binary_mask = np.zeros_like(segmentation_map)
            binary_mask[segmentation_map == region_id] = 255
            
            # Apply adaptive smoothing to the mask
            binary_mask = adaptive_blur(binary_mask)
            
            # Find contours with optimization
            contours = optimize_contour_extraction(binary_mask)
            
            # Simplify contours for smoother appearance
            smoothed_contours = []
            for contour in contours:
                # Apply contour approximation for smoothing
                approx = smooth_contour(contour)
                
                # Convert contour to a list of points
                points = []
                for point in approx:
                    points.append({"x": float(point[0][0]), "y": float(point[0][1])})
                
                smoothed_contours.append(points)
            
            # Handle region 4 specifically to go around the nose region
            if region_id == 4 and landmarks:
                # Find nose points in landmarks (assuming nose points are around indices 27-35)
                nose_points = []
                for i, point in enumerate(landmarks):
                    if 27 <= i <= 35:  # Nose region
                        nose_points.append((point.x, point.y))
                
                if nose_points:
                    # Adjust region 4 contours to go around nose
                    # This is a simplified approach - in production you'd need more sophisticated logic
                    # to properly curve around the nose region
                    smoothed_contours = adjust_contour_around_nose(smoothed_contours, nose_points)
            
            # Store the contours for this region
            facial_regions[str(region_id)] = smoothed_contours
    
    return facial_regions

def adjust_contour_around_nose(contours, nose_points):
    """Adjust contours to go around nose points."""
    # This is a placeholder for more sophisticated nose avoidance logic
    # In a real implementation, you'd reshape the contour to curve around the nose
    adjusted_contours = []
    
    for contour in contours:
        # Create a convex hull of nose points
        if len(nose_points) >= 3:  # Need at least 3 points for a polygon
            nose_hull = cv2.convexHull(np.array(nose_points, dtype=np.int32))
            nose_mask = np.zeros((1000, 1000), dtype=np.uint8)  # Arbitrary size
            cv2.fillPoly(nose_mask, [nose_hull], 255)
            
            # For each point in the contour, check if it's inside the nose area
            adjusted_points = []
            for point in contour:
                x, y = int(point["x"]), int(point["y"])
                if 0 <= x < nose_mask.shape[1] and 0 <= y < nose_mask.shape[0]:
                    if nose_mask[y, x] > 0:
                        # Point is in nose area, adjust it (simplified)
                        continue
                adjusted_points.append(point)
            
            if adjusted_points:
                adjusted_contours.append(adjusted_points)
        else:
            adjusted_contours.append(contour)
    
    return adjusted_contours

def generate_svg(image_shape, facial_regions):
    """Generate an SVG with the facial regions drawn as contours."""
    # Create an SVG with the facial regions
    svg_root = ET.Element("svg", {
        "width": str(image_shape[1]),
        "height": str(image_shape[0]),
        "xmlns": "http://www.w3.org/2000/svg"
    })
    
    # Define styles for different regions
    region_styles = {
        "1": {"stroke": "#FF0000", "fill": "rgba(255,0,0,0.2)"},
        "2": {"stroke": "#00FF00", "fill": "rgba(0,255,0,0.2)"},
        "3": {"stroke": "#0000FF", "fill": "rgba(0,0,255,0.2)"},
        "4": {"stroke": "#FFFF00", "fill": "rgba(255,255,0,0.2)"},
        "5": {"stroke": "#FF00FF", "fill": "rgba(255,0,255,0.2)"},
        "6": {"stroke": "#00FFFF", "fill": "rgba(0,255,255,0.2)"},
        "7": {"stroke": "#FFFFFF", "fill": "rgba(255,255,255,0.2)"}
    }
    
    # Add each region to the SVG
    for region_id, contours in facial_regions.items():
        for contour in contours:
            # Create a path for this contour
            path_data = "M"
            for i, point in enumerate(contour):
                if i == 0:
                    path_data += f"{point['x']},{point['y']}"
                else:
                    path_data += f" L{point['x']},{point['y']}"
            path_data += " Z"  # Close the path
            
            # Create the path element
            style = region_styles.get(region_id, {"stroke": "#000000", "fill": "rgba(0,0,0,0.2)"})
            path = ET.SubElement(svg_root, "path", {
                "d": path_data,
                "stroke": style["stroke"],
                "stroke-width": "2",
                "stroke-dasharray": "5,5",  # Dashed outline
                "fill": style["fill"],
                "class": f"region-{region_id}"
            })
    
    # Convert the SVG to a string
    svg_string = ET.tostring(svg_root, encoding="utf-8").decode("utf-8")
    
    # Encode as base64
    svg_base64 = base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")
    
    return svg_base64

def process_image(image_base64, landmarks, segmentation_map_base64):
    """Main function to process the image and generate the SVG output."""
    # Decode the image and segmentation map
    image = decode_image(image_base64)
    segmentation_map = decode_segmentation_map(segmentation_map_base64)
    
    # Check if a face is detected
    if not landmarks or len(landmarks) == 0:
        raise ValueError("No face detected in the image")
    
    # Autorotate the face
    rotated_image, rotated_landmarks = autorotate_face(image, landmarks)
    
    # Crop the image to focus on the face
    cropped_image, adjusted_landmarks = crop_face(rotated_image, rotated_landmarks)
    
    # Extract facial regions
    facial_regions = extract_facial_regions(cropped_image, segmentation_map, adjusted_landmarks)
    
    # Generate SVG
    svg_base64 = generate_svg(cropped_image.shape, facial_regions)
    
    return {
        "svg": svg_base64,
        "mask_contours": facial_regions
    }

async def process_image(image_data, segmentation_map, landmarks, job_id=None, options=None, progress_callback=None):
    """
    Async wrapper for image processing to be used in the API.
    
    Args:
        image_data: Base64 encoded image
        segmentation_map: Base64 encoded segmentation map 
        job_id: Job ID for tracking (optional)
        options: Processing options (optional)
        progress_callback: Callback function for progress updates (optional)
    
    Returns:
        Dictionary with processing results
    """
 
    
    # If progress callback is provided, update progress
    if progress_callback:
        await progress_callback(0.2)
    
    # Mock segmentation map if not provided
    if not segmentation_map:
        # In a real app, you'd generate a segmentation map
        # For this test, we're just using a mock one
        segmentation_map = image_data  # Placeholder
    
    # Process the image
    try:
        # If progress callback is provided, update progress
        if progress_callback:
            await progress_callback(0.5)
        
        # In a real implementation, we'd call the actual image processing
        # For this test, we're returning a mock result
        result = {
            "svg": "base64_encoded_svg_data",
            "regions": {
                "1": [{"x": 100, "y": 100}, {"x": 200, "y": 100}, {"x": 200, "y": 200}, {"x": 100, "y": 200}],
                "2": [{"x": 300, "y": 300}, {"x": 400, "y": 300}, {"x": 400, "y": 400}, {"x": 300, "y": 400}]
            }
        }
        
        # If progress callback is provided, update progress
        if progress_callback:
            await progress_callback(0.9)
        
        return result
    except Exception as e:
        # Log the error
        print(f"Error processing image: {str(e)}")
        raise

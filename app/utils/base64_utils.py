import base64
import numpy as np
import cv2

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
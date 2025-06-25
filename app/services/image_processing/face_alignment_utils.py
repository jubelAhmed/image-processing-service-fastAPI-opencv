import cv2
import numpy as np

def calculate_asymmetric_padding(w, h, min_pad_x=30, min_pad_top=50, min_pad_bottom=20):
    """
    Calculates asymmetric padding values for cropping around the face.
    Allows extra space at the top and bottom to accommodate forehead/chin.
    """
    padding_x = max(int(0.2 * w), min_pad_x)
    padding_y_top = max(int(0.4 * h), min_pad_top)
    padding_y_bottom = max(int(0.15 * h), min_pad_bottom)
    return padding_x, padding_y_top, padding_y_bottom

def get_padded_bbox(points, img_shape):
    """
    Computes a padded bounding box around given points, ensuring it stays within image bounds.
    
    Args:
        points (ndarray): Landmark points after rotation.
        img_shape (tuple): Shape of the image (height, width, channels).
    
    Returns:
        Tuple[int, int, int, int]: x_start, y_start, x_end, y_end of the cropped region.
    """
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    padding_x, padding_y_top, padding_y_bottom = calculate_asymmetric_padding(w, h)
    
    x_start = max(x - padding_x, 0)
    y_start = max(y - padding_y_top, 0)
    x_end = min(x + w + padding_x, img_shape[1])
    y_end = min(y + h + padding_y_bottom, img_shape[0])
    
    return x_start, y_start, x_end, y_end

def rotate_and_crop_face(image, landmarks, angle=None, return_crop_box=False):
    """
    Rotates the face based on eye alignment and crops it with padding.
    
    Args:
        image (ndarray): Original facial image (RGB or BGR).
        landmarks (ndarray): Facial landmarks as a (N, 2) array.
    
    Returns:
        cropped_image (ndarray): Cropped and aligned face image.
        cropped_landmarks (ndarray): Transformed landmarks in cropped image coordinates.
        M (ndarray): The 2x3 affine transformation matrix used for rotation.
    """
    h, w = image.shape[:2]

    # Get coordinates of eyes using MediaPipe standard landmark indices
    left_eye_pts = landmarks[[33, 133]]
    right_eye_pts = landmarks[[362, 263]]
    left_eye_center = left_eye_pts.mean(axis=0)
    right_eye_center = right_eye_pts.mean(axis=0)

    # Compute angle between eyes
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Get rotation matrix around image center
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image and landmarks
    rotated_image = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(224, 224, 224)
    )
    rotated_landmarks = cv2.transform(landmarks.reshape(-1, 1, 2), M).reshape(-1, 2)

    # Crop a padded region around the rotated landmarks
    x1, y1, x2, y2 = get_padded_bbox(rotated_landmarks, rotated_image.shape)
    cropped_image = rotated_image[y1:y2, x1:x2]
    cropped_landmarks = rotated_landmarks - np.array([x1, y1])  # Shift landmarks to cropped coordinates

    x_start, y_start, x_end, y_end = get_padded_bbox(rotated_landmarks, rotated_image.shape)
    cropped = rotated_image[y_start:y_end, x_start:x_end]
    cropped_landmarks = rotated_landmarks - [x_start, y_start]

    if return_crop_box:
        return cropped, cropped_landmarks, M, (x_start, y_start, x_end, y_end)
    return cropped, cropped_landmarks, M


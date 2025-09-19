"""
Performance optimization utilities for image processing.
"""

import numpy as np
import cv2
import functools
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Global thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=4)

def timeit(func):
    """Decorator to measure execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper

async def run_in_threadpool(func, *args, **kwargs):
    """Run a CPU-bound function in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        thread_pool, 
        functools.partial(func, *args, **kwargs)
    )

def optimize_contour_extraction(binary_mask, min_contour_area=10):
    """Optimized version of contour extraction with filtering for small contours."""
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    significant_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            significant_contours.append(contour)
    
    return significant_contours

def parallel_process_regions(segmentation_map, region_ids):
    """Process multiple regions in parallel."""
    results = {}
    
    def process_region(region_id):
        binary_mask = np.zeros_like(segmentation_map)
        binary_mask[segmentation_map == region_id] = 255
        binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
        contours = optimize_contour_extraction(binary_mask)
        return region_id, contours
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_region, region_id) for region_id in region_ids]
        for future in futures:
            region_id, contours = future.result()
            results[region_id] = contours
    
    return results

def smooth_contour(contour, epsilon_factor=0.001):
    """Apply contour approximation for smoother appearance."""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def adaptive_blur(mask, kernel_size=5, sigma=0):
    """Apply adaptive Gaussian blur based on mask size."""
    height, width = mask.shape[:2]
    # Adjust kernel size based on image dimensions
    adaptive_kernel = max(3, min(kernel_size, width // 30, height // 30))
    # Make sure kernel size is odd
    if adaptive_kernel % 2 == 0:
        adaptive_kernel += 1
    return cv2.GaussianBlur(mask, (adaptive_kernel, adaptive_kernel), sigma)


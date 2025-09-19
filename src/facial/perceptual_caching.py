import base64
import hashlib
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from src.core.database import DatabaseService
from src.facial.face_schema import LandmarkPoint
from src.core.utils import logger

class PerceptualHashCache:
    """Cache system using perceptual hashing for images."""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
    
    async def get_cached_result(self, image_base64: str, landmarks: List[LandmarkPoint] = None, 
                               segmentation_map_base64: str = None) -> Optional[Dict[str, Any]]:
        """Try to get cached result based on perceptual similarity."""
        
        p_hash = self._compute_perceptual_hash(image_base64)
        
        input_data = {"image_hash": p_hash}
        
        if landmarks:
            input_data["landmarks"] = [{"x": lm.x, "y": lm.y} for lm in landmarks]
            
        if segmentation_map_base64:
            seg_hash = self._compute_perceptual_hash(segmentation_map_base64)
            input_data["segmentation_map_hash"] = seg_hash
                
        result = await self.db_service.get_cached_result(input_data)
        
        return result
    
    async def store_result(self, image_base64: str, result: Dict[str, Any], 
                          landmarks: List[LandmarkPoint] = None, 
                          segmentation_map_base64: str = None) -> int:
        """Cache result with perceptual hash and return cache ID."""
        
        p_hash = self._compute_perceptual_hash(image_base64)
        
        input_data = {"image_hash": p_hash}
        
        if landmarks:
            input_data["landmarks"] = [{"x": lm.x, "y": lm.y} for lm in landmarks]
            
        if segmentation_map_base64:
            seg_hash = self._compute_perceptual_hash(segmentation_map_base64)
            input_data["segmentation_map_hash"] = seg_hash
        
        
        try:
            cache_id = await self.db_service.store_cached_result(input_data, result)
            return cache_id
        except Exception as e:
            logger.error(f"Error in store_result: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _compute_perceptual_hash(self, image_base64: str) -> str:
        """Compute perceptual hash of an image using pHash algorithm."""
        try:
            img_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                fallback_hash = hashlib.sha256(image_base64.encode('utf-8')).hexdigest()
                logger.error(f"Image decode failed, using fallback hash: {fallback_hash[:16]}...")
                return fallback_hash
            
            img = cv2.resize(img, (32, 32))
            dct = cv2.dct(np.float32(img))
            dct_low = dct[:8, :8]
            med = np.median(dct_low)
            hash_bits = (dct_low > med).flatten()
            
            hash_hex = ""
            for i in range(0, len(hash_bits), 4):
                value = 0
                for j in range(4):
                    if i + j < len(hash_bits) and hash_bits[i + j]:
                        value += 1 << j
                hash_hex += hex(value)[2:]
            
            return hash_hex
        except Exception as e:
            logger.error(f"Error computing perceptual hash: {e}")
            fallback_hash = hashlib.sha256(image_base64.encode('utf-8')).hexdigest()
            logger.debug(f"Using fallback hash: {fallback_hash[:16]}...")
            return fallback_hash
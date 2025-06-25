from app.db.postgres import PostgresClient
import base64
import hashlib
from typing import List, Dict, Any, Optional
import cv2
from app.schemas.face_schema import LandmarkPoint
import numpy as np

class PerceptualHashCache:
    """Cache system using perceptual hashing for images."""
    
    def __init__(self, postgres_client: PostgresClient):
        self.postgres = postgres_client
    
    async def get_cached_result(self, image_base64: str, landmarks: List[LandmarkPoint] = None, 
                               segmentation_map_base64: str = None) -> Optional[Dict[str, Any]]:
        """Try to get cached result based on perceptual similarity."""
        print(f"=== DEBUG: get_cached_result ===")
        
        p_hash = self._compute_perceptual_hash(image_base64)
        print(f"Computed perceptual hash: {p_hash}")
        
        input_data = {"image_hash": p_hash}
        
        if landmarks:
            input_data["landmarks"] = [{"x": lm.x, "y": lm.y} for lm in landmarks]
            print(f"Added {len(landmarks)} landmarks to cache key")
            
        if segmentation_map_base64:
            seg_hash = self._compute_perceptual_hash(segmentation_map_base64)
            input_data["segmentation_map_hash"] = seg_hash
            print(f"Added segmentation map hash: {seg_hash}")
        
        print(f"Cache lookup input_data keys: {list(input_data.keys())}")
        
        result = await self.postgres.get_cached_result(input_data)
        print(f"Cache lookup result: {'Found' if result else 'Not found'}")
        
        return result
    
    async def store_result(self, image_base64: str, result: Dict[str, Any], 
                          landmarks: List[LandmarkPoint] = None, 
                          segmentation_map_base64: str = None) -> int:
        """Cache result with perceptual hash and return cache ID."""
        print(f"=== DEBUG: store_result ===")
        
        p_hash = self._compute_perceptual_hash(image_base64)
        print(f"Computed perceptual hash for storage: {p_hash}")
        
        input_data = {"image_hash": p_hash}
        
        if landmarks:
            input_data["landmarks"] = [{"x": lm.x, "y": lm.y} for lm in landmarks]
            print(f"Added {len(landmarks)} landmarks to storage key")
            
        if segmentation_map_base64:
            seg_hash = self._compute_perceptual_hash(segmentation_map_base64)
            input_data["segmentation_map_hash"] = seg_hash
            print(f"Added segmentation map hash to storage: {seg_hash}")
        
        print(f"Storage input_data keys: {list(input_data.keys())}")
        print(f"Result to store keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        
        try:
            cache_id = await self.postgres.store_cached_result(input_data, result)
            print(f"Successfully stored result with cache_id: {cache_id}")
            return cache_id
        except Exception as e:
            print(f"Error in store_result: {e}")
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
                print(f"Image decode failed, using fallback hash: {fallback_hash[:16]}...")
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
            
            print(f"Successfully computed perceptual hash: {hash_hex[:16]}...")
            return hash_hex
        except Exception as e:
            print(f"Error computing perceptual hash: {e}")
            fallback_hash = hashlib.sha256(image_base64.encode('utf-8')).hexdigest()
            print(f"Using fallback hash: {fallback_hash[:16]}...")
            return fallback_hash
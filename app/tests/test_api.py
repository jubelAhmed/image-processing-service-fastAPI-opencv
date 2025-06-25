"""
Test file for the API endpoints.
"""
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import json
from fastapi.testclient import TestClient

from app.main import app
from app.routers.facial_processing import process_image_task
from app.schemas.face_schema import (
    ImageProcessingRequest, 
    ProcessingResponse,
    JobStatusResponse,
    LandmarkPoint
)

class TestFacialProcessingAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Mock base64 image data
        self.test_image = "SGVsbG8gV29ybGQ="  # "Hello World" in base64
    
    @patch('app.routers.facial_processing.get_perceptual_hash_cache')
    @patch('app.routers.facial_processing.get_postgres_client')
    def test_process_image_endpoint(self, mock_db_func, mock_cache_func):
        """Test the process-image endpoint."""
        # Setup mocks
        mock_db = AsyncMock()
        mock_cache = AsyncMock()
        mock_db_func.return_value = mock_db
        mock_cache_func.return_value = mock_cache
        mock_cache.get_cached_result.return_value = None
        mock_db.store_job_status = AsyncMock()
        
        # Make request
        response = self.client.post(
            "/api/v1/frontal/crop/submit",
            json={
                "image": self.test_image,
                "segmentation_map": self.test_image,  # Using same base64 for simplicity
                "landmarks": [{"x": 100, "y": 100}, {"x": 200, "y": 200}],
                "options": {"quality": "high"}
            }
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "queued")
    
    @patch('app.routers.facial_processing.get_perceptual_hash_cache')
    @patch('app.routers.facial_processing.get_postgres_client')
    def test_process_image_endpoint_cached(self, mock_db_func, mock_cache_func):
        """Test the process-image endpoint with cached result."""
        # Setup mocks
        mock_db = AsyncMock()
        mock_cache = AsyncMock()
        mock_db_func.return_value = mock_db
        mock_cache_func.return_value = mock_cache
        mock_cache.get_cached_result.return_value = {
            "svg": "<svg></svg>",
            "regions": {"1": []}
        }
        
        # Make request
        test_request = {
            "image": self.test_image,
            "segmentation_map": self.test_image,  # Using same base64 for simplicity
            "landmarks": [{"x": 100, "y": 100}, {"x": 200, "y": 200}],
            "options": {"quality": "high"}
        }
        
        response = self.client.post(
            "/api/v1/frontal/crop/submit",
            json=test_request
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "completed")
    
    @patch('app.routers.facial_processing.get_postgres_client')
    def test_get_job_status(self, mock_db_func):
        """Test the get-job-status endpoint."""
        # Setup mocks
        mock_db = AsyncMock()
        mock_db_func.return_value = mock_db
        mock_db.get_job_status.return_value = {
            "status": "completed",
            "result": {"svg": "<svg></svg>"},
            "start_time": 1655123456.789,
            "end_time": 1655123457.123
        }
        
        # Make request
        response = self.client.get("/api/v1/frontal/crop/status/test-job-id")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["job_id"], "test-job-id")
        self.assertEqual(data["status"], "completed")
    
    @patch('app.routers.facial_processing.get_postgres_client')
    def test_get_job_status_not_found(self, mock_db_func):
        """Test the get-job-status endpoint with non-existent job ID."""
        # Setup mocks
        mock_db = AsyncMock()
        mock_db_func.return_value = mock_db
        mock_db.get_job_status.return_value = None
        
        # Make request
        response = self.client.get("/api/v1/frontal/crop/status/nonexistent-id")
        
        # Check response
        self.assertEqual(response.status_code, 404)
    
    async def test_process_image_task(self):
        """Test the background task for processing images."""
        # Setup
        job_id = "test-job"
        image_data = self.test_image
        options = {"quality": "high"}
        
        # Mock dependencies
        postgres_client = AsyncMock()
        perceptual_hash_cache = AsyncMock()
        
        # Call the function
        with patch('app.core.image_processor.process_image', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"svg": "<svg></svg>"}
            
            await process_image_task(
                job_id=job_id,
                image_data=image_data,
                segmentation_map=None,
                options=options,
                postgres_client=postgres_client,
                perceptual_hash_cache=perceptual_hash_cache
            )
            
            # Verify correct function calls
            postgres_client.update_job_status.assert_called()
            mock_process.assert_called_once()
            perceptual_hash_cache.store_result.assert_called_once()
    
    @patch('app.routers.facial_processing.get_perceptual_hash_cache')
    @patch('app.routers.facial_processing.get_postgres_client')
    @patch('app.routers.facial_processing.process_image')
    def test_process_image_synchronous(self, mock_process, mock_db_func, mock_cache_func):
        """Test the synchronous processing endpoint."""
        # Setup mocks
        mock_db = AsyncMock()
        mock_cache = AsyncMock()
        mock_db_func.return_value = mock_db
        mock_cache_func.return_value = mock_cache
        mock_cache.get_cached_result.return_value = None
        
        # Mock the process_image function
        mock_process.return_value = {
            "svg": "<svg></svg>",
            "regions": {"1": []}
        }
        
        # Make request
        test_request = {
            "image": self.test_image,
            "segmentation_map": self.test_image,  # Using same base64 for simplicity
            "landmarks": [{"x": 100, "y": 100}, {"x": 200, "y": 200}],
            "options": {"quality": "high"}
        }
        
        response = self.client.post(
            "/api/v1/frontal/crop/process",
            json=test_request
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("result", data)
        self.assertIn("processing_time", data)

if __name__ == '__main__':
    unittest.main()

"""
Test file for the PostgreSQL database module.
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from app.db.postgres import PostgresClient, PerceptualHashCache

class TestPostgresClient(unittest.TestCase):
    @patch('app.db.postgres.asyncpg.create_pool')
    async def test_connect(self, mock_create_pool):
        """Test connecting to PostgreSQL."""
        # Setup mock
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool
        
        # Create client and connect
        client = PostgresClient()
        await client.connect("postgresql://user:pass@localhost/testdb")
        
        # Check that pool was created with correct connection string
        mock_create_pool.assert_called_once_with("postgresql://user:pass@localhost/testdb")
        
        # Check that pool is stored
        self.assertEqual(client.pool, mock_pool)
    
    @patch('app.db.postgres.asyncpg.create_pool')
    async def test_create_tables(self, mock_create_pool):
        """Test creating tables."""
        # Setup mock
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool
        
        # Create client, connect and create tables
        client = PostgresClient()
        await client.connect("postgresql://user:pass@localhost/testdb")
        await client.create_tables()
        
        # Check that execute was called (at least once)
        mock_conn.execute.assert_called()
    
    @patch('app.db.postgres.asyncpg.create_pool')
    async def test_store_job_status(self, mock_create_pool):
        """Test storing job status."""
        # Setup mock
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool
        
        # Create client and connect
        client = PostgresClient()
        await client.connect("postgresql://user:pass@localhost/testdb")
        
        # Store job status
        await client.store_job_status("test-job-1", "pending", 0.0)
        
        # Check that execute was called
        mock_conn.execute.assert_called()
    
    @patch('app.db.postgres.asyncpg.create_pool')
    async def test_get_job_status(self, mock_create_pool):
        """Test getting job status."""
        # Setup mock
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "job_id": "test-job-1",
            "status": "completed",
            "progress": 1.0,
            "result": '{"key": "value"}'
        }
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool
        
        # Create client and connect
        client = PostgresClient()
        await client.connect("postgresql://user:pass@localhost/testdb")
        
        # Get job status
        result = await client.get_job_status("test-job-1")
        
        # Check result
        self.assertEqual(result["job_id"], "test-job-1")
        self.assertEqual(result["status"], "completed")

class TestPerceptualHashCache(unittest.TestCase):
    @patch('app.db.postgres.PostgresClient')
    def test_init(self, mock_pg_client):
        """Test cache initialization."""
        cache = PerceptualHashCache(mock_pg_client)
        self.assertEqual(cache.pg_client, mock_pg_client)
    
    @patch('app.db.postgres.PostgresClient')
    @patch('app.db.postgres.PerceptualHashCache._compute_phash')
    async def test_get_cached_result(self, mock_compute_phash, mock_pg_client):
        """Test getting cached result."""
        # Setup mocks
        mock_compute_phash.return_value = "test-hash"
        mock_pg_client.pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "result": '{"svg": "test-svg", "contours": {}}'
        }
        mock_pg_client.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Create cache
        cache = PerceptualHashCache(mock_pg_client)
        
        # Get cached result
        result = await cache.get_cached_result("base64-image", {})
        
        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(result["svg"], "test-svg")
    
    @patch('app.db.postgres.PostgresClient')
    @patch('app.db.postgres.PerceptualHashCache._compute_phash')
    async def test_store_result(self, mock_compute_phash, mock_pg_client):
        """Test storing result in cache."""
        # Setup mocks
        mock_compute_phash.return_value = "test-hash"
        mock_pg_client.pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pg_client.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Create cache
        cache = PerceptualHashCache(mock_pg_client)
        
        # Store result
        await cache.store_result("base64-image", {"svg": "test-svg"}, {})
        
        # Check that execute was called
        mock_conn.execute.assert_called()

if __name__ == '__main__':
    unittest.main()

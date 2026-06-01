#!/usr/bin/env python3
"""Test script for CognitiveEngine connection pool and retry mechanism."""

import asyncio
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)

async def test_connection_pool():
    """Test the connection pool with mock engine."""
    sys.path.insert(0, '/Users/bryan/.aura/live-source')
    
    from core.providers.engine_connection_pool import (
        get_engine_connection_pool,
        ConnectionHealth,
        ConnectionRetryConfig,
    )
    
    print("✅ Successfully imported engine_connection_pool")
    
    # Test pool instantiation
    pool = get_engine_connection_pool()
    print("✅ Created engine connection pool")
    
    # Test retry config
    config = ConnectionRetryConfig()
    print(f"✅ Retry config: max_retries={config.max_retries}, initial_backoff={config.initial_backoff_seconds}s")
    
    # Test backoff calculation
    for attempt in range(config.max_retries):
        delay = config.get_backoff_delay(attempt)
        timeout = config.get_timeout_for_attempt(120.0, attempt)
        print(f"  Attempt {attempt + 1}: backoff={delay:.1f}s, timeout={timeout:.1f}s")
    
    # Test health status tracking
    class MockEngine:
        async def think(self, message, **kwargs):
            return f"Response to: {message}"
    
    mock_engine = MockEngine()
    await pool.acquire_engine_connection(mock_engine, connection_id="test_conn")
    print("✅ Acquired connection")
    
    # Test successful operation
    async def test_operation():
        return await mock_engine.think("Test message")
    
    result = await pool.execute_with_retry(
        "test_operation",
        test_operation,
        connection_id="test_conn",
        timeout=5.0,
    )
    print(f"✅ Operation succeeded: {result}")
    
    # Check health status
    health = pool.get_health_status("test_conn")
    print(f"✅ Connection health: {health['health']}")
    print(f"   - Success count: {health['success_count']}")
    print(f"   - Uptime: {health['uptime_seconds']:.1f}s")
    
    # Test failing operation (should retry and eventually fail)
    async def failing_operation():
        raise ValueError("Simulated failure")
    
    result = await pool.execute_with_retry(
        "failing_operation",
        failing_operation,
        connection_id="test_conn",
        timeout=0.5,  # Short timeout
    )
    print(f"✅ Failing operation returned: {result}")
    
    # Check degraded health status
    health = pool.get_health_status("test_conn")
    print(f"✅ Connection health after failure: {health['health']}")
    print(f"   - Failure count: {health['failure_count']}")
    print(f"   - Consecutive failures: {health['consecutive_failures']}")
    
    # Close connection
    await pool.close_connection("test_conn")
    print("✅ Connection closed")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_connection_pool())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

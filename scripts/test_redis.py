import redis
from typing import Optional

def test_redis_connection() -> None:
    """Test Redis connection and basic operations"""
    try:
        # Connect to Redis
        r: redis.Redis = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # Test connection
        response: Optional[bool] = r.ping()
        print(f"Redis connection test: {'SUCCESS' if response else 'FAILED'}")
        
        # Test basic operations
        r.set('test_key', 'test_value')
        value: Optional[str] = r.get('test_key')
        print(f"Redis get/set test: {'SUCCESS' if value == 'test_value' else 'FAILED'}")
        
        # Clean up
        r.delete('test_key')
        
    except redis.ConnectionError as e:
        print(f"Failed to connect to Redis: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_redis_connection() 
import time

class RateLimiter:
    @staticmethod
    def create_resources(manager, max_calls_per_sec=9):
        """Create and return shared resources for rate limiting"""
        timestamps = manager.list();  # Shared list for timestamps

        return {
            'timestamps': timestamps,
            'max_calls': max_calls_per_sec
        };

    @staticmethod
    def wait(rate_limiter_resources):
        """Wait if necessary to maintain the rate limit"""
        timestamps = rate_limiter_resources['timestamps'];
        max_calls = rate_limiter_resources['max_calls'];
        
        current_time = time.time();
        
        # Filter out old timestamps
        valid_timestamps = [t for t in timestamps if current_time - t < 1];
        
        # Update the shared list with only valid timestamps
        timestamps[:] = valid_timestamps;
        
        # Check if we need to wait
        if len(timestamps) >= max_calls:
            wait_time = 1 - (current_time - timestamps[0]);
            if wait_time > 0:
                # print(f"Rate limit triggered, waiting for {wait_time:.3f} seconds");
                time.sleep(wait_time);

                # Recalculate after waiting
                current_time = time.time();
                timestamps[:] = [t for t in timestamps if current_time - t < 1];
        
        # Add the current timestamp
        timestamps.append(current_time);

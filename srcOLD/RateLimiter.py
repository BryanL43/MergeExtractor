import time
import threading

class RateLimiter:
    def __init__(self, max_calls_per_sec: int = 9):
        self.max_calls = max_calls_per_sec;
        self.lock = threading.Lock();
        self.timestamps = [];

    def wait(self):
        with self.lock:
            current_time = time.time();

            # Remove timestamps older than 1 second
            self.timestamps = [t for t in self.timestamps if current_time - t < 1];

            if len(self.timestamps) >= self.max_calls:
                wait_time = 1 - (current_time - self.timestamps[0]);
                if wait_time > 0:
                    print("We have triggered rate limiter wait");
                    time.sleep(wait_time);
                current_time = time.time();
                self.timestamps = [t for t in self.timestamps if current_time - t < 1];

            self.timestamps.append(time.time());
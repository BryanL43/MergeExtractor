import time

class TokenBucket:
    def __init__(self, max_calls_per_sec: int, manager):
        self.max_tokens = max_calls_per_sec;
        self.tokens = manager.Value('i', max_calls_per_sec);
        self.last_refill = manager.Value('d', time.time());
        self.lock = manager.Lock();

    def wait(self):
        """Wait until a token is available, enforcing rate limits across processes."""
        while True:
            with self.lock:
                now = time.time();
                elapsed = now - self.last_refill.value;

                # Refill tokens based on elapsed time
                new_tokens = int(elapsed * self.max_tokens);
                if new_tokens > 0:
                    self.tokens.value = min(self.max_tokens, self.tokens.value + new_tokens);
                    self.last_refill.value = now;
                    # print("Refilled Rate Limiter Tokens")

                # Consume a token
                if self.tokens.value > 0:
                    self.tokens.value -= 1;
                    # print("Consumed Rate Limiter Token")
                    return;

            # Slight delay to reduce flooding
            time.sleep(0.05);

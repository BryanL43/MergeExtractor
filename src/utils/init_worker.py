from src.dependencies.rate_limiter_globals import setup_rate_limiter

def init_worker(max_calls_per_sec):
    """
        Proxy method to setup the rate limiter for multiprocessing-safe usage.

        Parameters
        ----------
        max_calls_per_sec : int
            The maximum number of calls allowed per second.
    """
    setup_rate_limiter(max_calls_per_sec);

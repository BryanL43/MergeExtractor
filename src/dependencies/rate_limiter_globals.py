manager = None;
global_rate_limiter = None;

def setup_rate_limiter(max_calls_per_sec):
    """
        Initialize the global rate limiter for multiprocessing-safe usage.

        Parameters
        ----------
        max_calls_per_sec : int
            The maximum number of calls allowed per second.
    """
    global manager, global_rate_limiter;
    from multiprocessing import Manager
    manager = Manager();

    from src.dependencies.TokenBucket import TokenBucket
    global_rate_limiter = TokenBucket(max_calls_per_sec, manager);

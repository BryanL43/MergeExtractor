import time
import random
from concurrent.futures import ThreadPoolExecutor

def proxy_function(index):
    random_seconds = random.randint(1, 5)
    time.sleep(random_seconds)
    return f"Thread {index + 1} finished after {random_seconds} seconds."

def main():
    with ThreadPoolExecutor(max_workers = 5) as pool:
        # Submit tasks to the pool (simulating a task for each index)
        futures = [pool.submit(proxy_function, index) for index in range(5)]

        for future in futures:
            try:
                result = future.result()  # This will block until the task finishes
                print(f"Result from future: {result}")
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
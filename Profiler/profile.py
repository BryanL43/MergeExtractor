import cProfile
import pstats
import runpy
import os
import sys
import argparse

# Get absolute path to the project and src directory
file_dir = os.path.dirname(__file__);
project_root = os.path.abspath(os.path.join(file_dir, '..'));
src_dir = os.path.abspath(os.path.join(project_root, "src"));
sys.path.insert(0, src_dir)  # Add src to path so main.py can import properly

def is_user_code(file_path: str) -> bool:
    """Check if the file path is part of the user's own code (i.e., in src folder)."""
    # Normalize to lowercase and use forward slashes for cross-platform support
    normalized = os.path.abspath(file_path).replace("\\", "/").lower();
    return src_dir.replace("\\", "/").lower() in normalized;

def main():
    parser = argparse.ArgumentParser(description="Profile a Python module using cProfile.");
    parser.add_argument("--module", required=True, help="The module to run, e.g. src.crawler.crawl");
    args = parser.parse_args();

    profiler = cProfile.Profile();
    profiler.enable();

    # Run the mpdule as if from command line
    runpy.run_module(args.module, run_name="__main__");

    profiler.disable();

    output_path = os.path.join(file_dir, "profile_output.txt");
    with open(output_path, "w", encoding="utf-8") as file:
        stats = pstats.Stats(profiler, stream=file);
        stats.sort_stats("cumulative");  # Don't strip_dirs() so full paths remain

        file.write("Filtered functions with cumulative time > 0.1 seconds (user code only):\n\n");

        # Log user written method that has a cumulative runtime of greater than 0.1 seconds
        count = 0;
        for func, stat in stats.stats.items():
            filename, lineno, funcname = func;
            cumtime = stat[3];

            if is_user_code(filename) and cumtime > 0.1:
                count += 1;
                file.write(f"{funcname:40} {cumtime:.4f}s @ {filename}:{lineno}\n");

        if count == 0:
            print("\nWARNING: No user-defined functions exceeded 0.1s cumulative time.\n");

    print(f"\nProfiling complete! Results saved to: {output_path}");


if __name__ == "__main__":
    main();

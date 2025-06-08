from datetime import datetime

from src.dependencies.config import LOG_FILE_PATH

class Logger:
    @staticmethod
    def _get_current_timestamp():
        """Returns the current timestamp in 'YYYY-MM-DD HH:MM:SS' format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S");

    @staticmethod
    def logMessage(message: str, time_stamp=True):
        """
            Logs the message to a file.
        """
        
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as file:
            if time_stamp:
                file.write(f"[{Logger._get_current_timestamp()}] ");
            file.write(message + "\n");

        if time_stamp:
            print(f"[{Logger._get_current_timestamp()}] ", end="");
        print(message);

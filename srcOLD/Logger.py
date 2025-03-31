from datetime import datetime

class Logger:
    LOG_FILE_PATH = "./logs.txt"

    @staticmethod
    def get_current_timestamp():
        """Returns the current timestamp in 'YYYY-MM-DD HH:MM:SS' format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S");

    @staticmethod
    def logMessage(message: str):
        """
            Logs the message to a file.
            Date format not included due to need for dumping contents.
        """
        
        with open(Logger.LOG_FILE_PATH, "a", encoding="utf-8") as file:
            file.write(message + "\n");

        print(message);

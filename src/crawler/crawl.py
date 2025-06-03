from src.dependencies.config import *
from concurrent.futures import ThreadPoolExecutor

from src.crawler.CrawlerHandler import CrawlerHandler

def main():
    # backup_agent = BackupAssistant(openai_api_key, "Backup Agent", "gpt-4o-mini");

    crawler = CrawlerHandler();
    crawler.runCrawler(index=24, date_margin=4);
    # crawler.runCrawler(start_index=0, end_index=2, date_margin=4, batch_size=3);


if __name__ == "__main__":
    main();
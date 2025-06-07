from src.crawler.CrawlerHandler import CrawlerHandler

def main():
    # backup_agent = BackupAssistant(openai_api_key, "Backup Agent", "gpt-4o-mini");

    crawler = CrawlerHandler();
    # crawler.runCrawler(index=0, date_margin=4);
    crawler.runCrawler(start_index=50, end_index=54, date_margin=4, batch_size=5);


if __name__ == "__main__":
    main();
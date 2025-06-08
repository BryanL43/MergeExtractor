from src.crawler.CrawlerHandler import CrawlerHandler

def main():
    crawler = CrawlerHandler();
    crawler.runCrawler(index=19, date_margin=4);
    # crawler.runCrawler(start_index=50, end_index=54, date_margin=4, batch_size=5);


if __name__ == "__main__":
    main();
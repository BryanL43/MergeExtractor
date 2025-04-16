"""
    This object persistantly store a document's infomation extracted from the source url:
        - Source URL
        - Content
"""
class Document:
    def __init__(self, url: str, content: str):
        self.__url = url;
        self.__content = content;

    def getUrl(self) -> str:
        return self.__url;

    def getContent(self) -> str:
        return self.__content;

    def setContent(self, newContent):
        self.__content = newContent;
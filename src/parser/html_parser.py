import re
from urllib.request import urlopen
from bs4 import BeautifulSoup

# pip install beautifulsoup4

class ParseHTML:
    def __init__(self, 
                 html_text, 
                 parser_type = "html.parser"):
        self.html_text = html_text
        self.parser_type = parser_type
        self.soup = BeautifulSoup(html_text, parser_type)

    def get_title(self):
        """Returns title of html text."""
        page_title = self.soup.title.string
        return page_title

    def get_tags(self):
        """ Returns tags from html text with removed space. """
        fetch = self.soup.head.stripped_strings
        return [repr(f) for f in fetch]
    
    def find_tag_data(self, tag, id, limit:int=3):
        """ Returns match tag data with id.
            tag: tag name to find
            id: tag id to find
            limit: no of result to fetch
        """
        fetch = self.soup.find_all(tag, id, limit=limit)
        return [f.text for f in fetch]
    
    def find_string_data(self, text:str, limit:int=3):
        """ Returns matched text data.
            text (str): text data to find
            limit: no of result to fetch
        """
        fetch = self.soup.find_all(string=re.compile(text), limit=limit)
        return fetch

if __name__ == "__main__":
    with urlopen('https://docs.python.org/3/library/urllib.request.html') as r:
        body = r.read().decode('utf-8')
  
    p = ParseHTML(body) 
    # result = p.get_title()
    # result = p.get_tags()
    # result = p.find_tag_data('dl', 'py function')
    result = p.find_string_data('must be an object')
    print(result)
 

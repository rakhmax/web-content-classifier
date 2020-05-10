import requests
import html2text
import pandas as pd
from bs4 import BeautifulSoup
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stop_words_ru = get_stop_words('russian')
stop_words_en = get_stop_words('english')
stop_words = stop_words_en + stop_words_ru

h = html2text.HTML2Text()
h.ignore_links = True


def scrap_urls(urls):
    df = pd.DataFrame(urls)
    contents = []
    porter = PorterStemmer()

    for row in df.itertuples():
        try:
            html_code = requests.get(row[1]).content
        except Exception as e:
            print(e)

        try:
            soup = BeautifulSoup(html_code, 'html5lib')
            [s.extract() for s in soup('script')]
            [s.extract() for s in soup('style')]
            title = soup.title(text=True)[0]
            body = soup.body(text=True)
            html_body = ' '.join(body)
        except Exception as e:
            print(e)

        try:
            tokenized_title = word_tokenize(title)
            result_title = [porter.stem(i) for i in tokenized_title
                            if i.lower() not in stop_words
                            and i.isalpha()
                            and len(i) > 3]

            text = h.handle(html_body)
            text_from_html = text.replace('\n', ' ')
            tokenized_html = word_tokenize(text_from_html)
            result_words = [porter.stem(i) for i in tokenized_html
                            if i.lower() not in stop_words
                            and i.isalpha()
                            and len(i) > 3]

            title = ' '.join(result_title).lower()
            content = ' '.join(result_words).lower()

            if len(row) == 3:
                contents.append([' '.join([row[1], title, content]), row[2]])
            else:
                contents.append(' '.join([row[1], title, content]))
        except Exception as e:
            print(e)

    return contents

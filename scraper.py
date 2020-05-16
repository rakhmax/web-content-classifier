import requests
import html2text
import pandas as pd
from bs4 import BeautifulSoup
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import time

stop_words_ru = get_stop_words('russian')
stop_words_en = get_stop_words('english')
stop_words = stop_words_en + stop_words_ru

h = html2text.HTML2Text()
h.ignore_links = True


def scrap_urls(urls):
    df = pd.DataFrame(urls)
    contents = []
    stemmer_ru = SnowballStemmer('russian')
    stemmer_en = SnowballStemmer('english')

    for row in df.itertuples():
        try:
            html_code = requests.get(row[1]).content
        except Exception as e:
            print(e)

        try:
            soup = BeautifulSoup(html_code, 'lxml')
            [s.extract() for s in soup('script')]
            [s.extract() for s in soup('style')]
            title = soup.title(text=True)[0]
            body = soup.body(text=True)
            html_body = ' '.join(body)
        except Exception as e:
            print(e)

        try:
            tokenized_title = word_tokenize(title)
            result_title = [stemmer_ru.stem(stemmer_en.stem(i)) for i in tokenized_title
                            if i.lower() not in stop_words
                            and i.isalpha()
                            and len(i) > 3]

            text_from_html = html_body.replace('\n', ' ')
            tokenized_html = word_tokenize(text_from_html)
            result_words = [stemmer_ru.stem(stemmer_en.stem(i)) for i in tokenized_html
                            if i.lower() not in stop_words
                            and i.isalpha()
                            and len(i) > 3]

            title = ' '.join(result_title).lower()
            content = ' '.join(result_words).lower()

            if len(row) == 3:
                contents.append([row[1], ' '.join([title, content]), row[2]])
            else:
                contents.append(' '.join([title, content]))
        except Exception as e:
            print(e)

    return contents

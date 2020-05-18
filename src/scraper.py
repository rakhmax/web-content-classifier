import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from stop_words import get_stop_words


def scrape_urls(urls):
    stop_words_ru = get_stop_words('russian')
    stop_words_en = get_stop_words('english')
    stop_words = stop_words_en + stop_words_ru

    stemmer_ru = SnowballStemmer('russian')
    stemmer_en = SnowballStemmer('english')

    pattern = re.compile("^https?://")

    df = pd.DataFrame(urls)
    contents = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36 Edg/81.0.416.72'
    }

    for row in df.itertuples():
        if not pattern.match(row[1]):
            print('URL must begin with http:// or https://')
            continue

        try:
            response = requests.get(row[1], headers=headers)
        except Exception as e:
            print(e)
            continue

        if response.status_code == 200:
            try:
                soup = BeautifulSoup(response.content, 'lxml')
                [s.decompose() for s in soup('noscript')]
                [s.decompose() for s in soup('script')]
                [s.decompose() for s in soup('style')]
                title = soup.title(text=True)[0]
                body = soup.body(text=True)
                html_body = ' '.join(body)
            except Exception as e:
                print(e)
                continue

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
                continue

    return contents

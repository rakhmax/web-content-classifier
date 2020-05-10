import json
from pathlib import Path
import html2text
import requests
import pandas as pd
from bs4 import BeautifulSoup
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from vars import Paths


stop_words_ru = get_stop_words('russian')
stop_words_en = get_stop_words('english')
stop_words = stop_words_en + stop_words_ru

h = html2text.HTML2Text()
h.ignore_links = True

def preprocess():
    try:
        with open(Path('./data', 'urls.json'), 'r') as urls:
            data = json.load(urls)

        urls_csv = []
        categories_csv = []

        for category in data:
            categories_csv.append([category, 1, 1, 1, 1])

            for url in data[category]:
                urls_csv.append([url, category])

        names = ['url', 'category']
        df = pd.DataFrame(urls_csv, columns=names)
    except Exception as e:
        print(e)

    header = ['category', 'common', 'kid', 'office', 'student']
    pd.DataFrame(categories_csv).to_csv(Paths.CATEGORIES.value, header=header, index=False)

    contents = []
    porter = PorterStemmer()

    for i, row in df.iterrows():
        try:
            html_code = requests.get(row.url).content
        except Exception as e:
            print(e)
            continue

        try:
            soup = BeautifulSoup(html_code, features='html.parser')
            [s.extract() for s in soup('script')]
            [s.extract() for s in soup('style')]
            title = soup.title(text=True)[0]
            body = soup.body(text=True)
            html_body = ' '.join(body)
        except Exception as e:
            print(e)
            continue

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

            contents.append([' '.join([row.url, title, content]), row.category])
        except Exception as e:
            print(e)
            continue

    df = pd.DataFrame(contents)
    header = ['content', 'category']
    df.to_csv(Paths.URLS.value, header=header, index=False)

    print('Data has been preprocessed')

if __name__ == '__main__':
    preprocess()

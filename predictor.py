import pickle
import requests
import html2text
import sys
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from vars import Paths

stop_words_ru = get_stop_words('russian')
stop_words_en = get_stop_words('english')
stop_words = stop_words_en + stop_words_ru

h = html2text.HTML2Text()
h.ignore_links = True
porter = PorterStemmer()


def pred(urls: list):
    contents = []

    for url in urls:
        try:
            html_code = requests.get(url).content
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
            contents.append(' '.join([url, title, content]))
        except Exception as e:
            print(e)

    tfidf: TfidfVectorizer = pickle.load(open(Paths.FEATURES.value, 'rb'))
    features = tfidf.transform(contents)

    clf = pickle.load(open(Paths.MODEL.value, 'rb'))

    return clf.predict(features)


if __name__ == '__main__':
    print(pred(sys.argv[1:]))

import pickle
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vars import Paths
from scraper import scrap_urls


def predict(urls: list):
    tfidf: TfidfVectorizer = pickle.load(open(Paths.FEATURES.value, 'rb'))
    features = tfidf.transform(scrap_urls(urls))

    clf = pickle.load(open(Paths.MODEL.value, 'rb'))

    return clf.predict(features)


if __name__ == '__main__':
    print(predict(sys.argv[1:]))

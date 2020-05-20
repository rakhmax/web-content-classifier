from argparse import ArgumentParser
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from prepare_db import cursor, conn
from scraper import scrape_urls
from vars import Paths, blacklist, whitelist


def init_args():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument('--url', help='website to analyze',
                        required=True, type=str)
    parser.add_argument('--user', help='user category', type=str, nargs='+')

    return parser.parse_args()


def is_allow(url, ctx=None):
    if not ctx:
        return True

    content = scrape_urls([url])

    if content:
        cats_clf = pickle.load(open(Paths.CATEGORIES_MODEL.value, 'rb'))
        ctx_clf = pickle.load(open(Paths.CTX_MODEL.value, 'rb'))

        predicted_cat = cats_clf['le'].inverse_transform(
            cats_clf['clf'].predict(content))
        predicted_ctx = ctx_clf['mlb'].inverse_transform(
            ctx_clf['clf'].predict(content))

        is_allow = any(i in ctx for i in np.asarray(*predicted_ctx))

        print ('Category:', predicted_cat)
        print ('Context:', predicted_ctx)

        return is_allow
    else:
        print('Nothing to predict')
        return False


if __name__ == '__main__':
    tic = time.perf_counter()
    args = init_args()
    print(is_allow(args.url, args.user))
    print(f'Predicted in {round(time.perf_counter() - tic, 1)} seconds')

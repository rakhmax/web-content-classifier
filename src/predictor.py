import argparse
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from prepare_db import cursor, conn
from scraper import scrape_urls
from vars import Paths, blacklist, whitelist


def init_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--user', help='user category', required=True, type=str)
    parser.add_argument('--url', help='website to analyze', required=True, type=str)

    return parser.parse_args()


def is_allow(ctx, urls) -> bool:
    # blist = blacklist(ctx)
    # wlist = whitelist(ctx)

    # if url in blist:
    #     return False

    # if url in wlist:
    #     return True

    content = scrape_urls(urls)

    if content:
        cats_clf = pickle.load(open(Paths.CATEGORIES_MODEL.value, 'rb'))
        ctx_clf = pickle.load(open(Paths.CTX_MODEL.value, 'rb'))

        predicted_cat = cats_clf['le'].inverse_transform(cats_clf['clf'].predict(content))
        predicted_ctx = ctx_clf['mlb'].inverse_transform(ctx_clf['clf'].predict(content))

        is_allow = ctx in np.asarray(predicted_ctx)

        print(predicted_cat, predicted_ctx)
        print(is_allow)
        
        return is_allow
    else:
        print('Nothing to predict')
        return False


if __name__ == '__main__':
    tic = time.perf_counter()
    # args = init_args()
    # is_allow(args.user, args.url)
    is_allow('underage', ['https://citilink.ru'])
    print(f'Predicted in {round(time.perf_counter() - tic, 1)} seconds')

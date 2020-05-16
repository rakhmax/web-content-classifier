import pickle
import argparse
import time
from io import StringIO
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from database import cursor, conn
from vars import Paths, blacklist, whitelist, banned_categories
from scraper import scrap_urls


def init_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--user', help='user category', required=True)
    parser.add_argument('--url', help='website to analyze', required=True)

    return parser.parse_args()


def predict(user: int, url: str):
    blist = blacklist(user)
    wlist = whitelist(user)
    banned = banned_categories(user)

    if url in blist:
        return 'Access denied'

    elif url in wlist:
        return 'Access granted'

    content = scrap_urls([url])

    tfidf: TfidfVectorizer = pickle.load(open(Paths.FEATURES.value, 'rb'))
    features = tfidf.transform(content)

    clf = pickle.load(open(Paths.MODEL.value, 'rb'))

    category = clf.predict(features)[0]

    if category in banned:
        cursor.execute(f'''INSERT INTO blacklist VALUES (
            {category}, {user}
        )''')
        conn.commit()

        return 'Access denied'

    predicted = clf.predict(features)
    pd.DataFrame([[url, *content, *predicted]]
                 ).to_csv(Paths.URLS.value, mode='a', index=False, header=False)

    return predicted


if __name__ == '__main__':
    tic = time.perf_counter()
    args = init_args()
    print(predict(args.user, args.url))
    print(f'Predicted in {round(time.perf_counter() - tic, 1)} seconds')

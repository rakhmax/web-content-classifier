import pickle
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from database import cursor, conn
from vars import Paths, blacklist, whitelist, banned_categories
from scraper import scrap_urls


def predict(url: str, user_type):
    blacklist = blacklist(user_type)
    whitelist = whitelist(user_type)
    banned_categories = banned_categories(user_type)

    if url in blacklist:
        return 'Access denied'

    elif url in whitelist:
        return 'Access granted'

    else:
        tfidf: TfidfVectorizer = pickle.load(open(Paths.FEATURES.value, 'rb'))
        features = tfidf.transform(scrap_urls([url]))

        clf = pickle.load(open(Paths.MODEL.value, 'rb'))

        category = clf.predict(features)[0]

        if category in banned_categories:
            cursor.execute(f'''INSERT INTO blacklist VALUES (
                {category}, {user_type}
            )''')
            return 'Access denied'

        return clf.predict(features)


if __name__ == '__main__':
    print(predict(sys.argv[1], sys.argv[2]))

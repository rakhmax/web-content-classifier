import json
from pathlib import Path
import pandas as pd
from vars import Paths
from scraper import scrap_urls


def preprocess():
    try:
        with open(Path('./data', 'urls.json'), 'r') as json_data:
            data = json.load(json_data)

        urls = []
        categories = []

        for category in data:
            categories.append([category, 1, 1, 1, 1])

            for url in data[category]:
                urls.append([url, category])
    except Exception as e:
        print(e)

    header = ['category', 'common', 'kid', 'office', 'student']
    pd.DataFrame(categories).to_csv(
        Paths.CATEGORIES.value, header=header, index=False)

    header = ['content', 'category']
    pd.DataFrame(scrap_urls(urls)).to_csv(
        Paths.URLS.value, header=header, index=False)

    print('Data has been preprocessed')


if __name__ == '__main__':
    preprocess()

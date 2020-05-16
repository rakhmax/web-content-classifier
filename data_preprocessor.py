import json
import time
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
            if category == 'Adult':
                categories.append([category, 1, 0, 0, 0])
            elif category == 'Games' or category == 'Recreation' or category == 'Shopping':
                categories.append([category, 1, 1, 0, 0])
            else:
                categories.append([category, 1, 1, 1, 0])

            for url in data[category]:
                urls.append([url, category])
    except Exception as e:
        print(e)

    header = ['category', 'common', 'kid', 'office', 'student']
    pd.DataFrame(categories).to_csv(
        Paths.CATEGORIES.value, header=header, index=False)

    header = ['url', 'content', 'category']
    pd.DataFrame(scrap_urls(urls)).to_csv(
        Paths.URLS.value, header=header, index=False)


if __name__ == '__main__':
    tic = time.perf_counter()
    preprocess()
    print(f'Preprocessed in {round(time.perf_counter() - tic, 1)} seconds')

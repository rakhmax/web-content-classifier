import json
import time
from os import path
from pathlib import Path
import pandas as pd
from scraper import scrap_urls
from vars import Paths


def preprocess():
    url_path = Paths.URLS.value

    if path.isfile(url_path):
        df = pd.read_csv(url_path).drop_duplicates(subset=['url'])
        df.to_csv(url_path, index=False)
    else:
        try:
            with open(Path('./data', 'urls.json'), 'r') as json_data:
                data = json.load(json_data)

            urls = []

            for category in data:
                for url in data[category]:
                    urls.append([url, category])

            scrapped_urls = scrap_urls(urls)

            labeled_urls = []

            for url in scrapped_urls:
                if url[2] == 'Adult':
                    labeled_urls.append([*url, 0, 0, 0])
                elif url[2] == 'Games' or url[2] == 'Recreation' or url[2] == 'Shopping':
                    labeled_urls.append([*url, 1, 0, 0])
                else:
                    labeled_urls.append([*url, 1, 1, 1])

            header = ['url', 'content', 'category', 'underage', 'office', 'student']
            pd.DataFrame(labeled_urls).to_csv(
                url_path, header=header, index=False)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    tic = time.perf_counter()
    preprocess()
    print(f'Preprocessed in {round(time.perf_counter() - tic, 2)} seconds')

from enum import Enum
from pathlib import Path


class Users(Enum):
    COMMON = 0
    UNDERAGE = 1
    STUDENT = 2
    OFFICE = 3


class Paths(Enum):
    CATEGORIES = Path('./data', 'preprocessed_categories.csv')
    URLS = Path('./data', 'preprocessed_urls.csv')
    FEATURES = Path('./models', 'features.pkl')
    MODEL = Path('./models', 'model.pkl')

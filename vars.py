from enum import Enum
from pathlib import Path
from database import cursor


def blacklist(user_type):
    cursor.execute(f'SELECT url FROM blacklist WHERE user_type = {user_type}')
    bl = cursor.fetchall()
    return [b[0] for b in bl]


def whitelist(user_type):
    cursor.execute(f'SELECT url FROM whitelist WHERE user_type = {user_type}')
    wl = cursor.fetchall()
    return [w[0] for w in wl]


def banned_categories(user_type):
    cursor.execute(
        f'SELECT categories FROM banned_categories WHERE user_type = {user_type}')
    bc = cursor.fetchall()
    return [c[0] for c in bc]


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

from enum import Enum
from pathlib import Path
from prepare_db import cursor


def blacklist(user_group: str) -> list:
    cursor.execute(f'SELECT url FROM blacklist WHERE user_group = {user_group}')
    bl = cursor.fetchall()
    return [b[0] for b in bl]


def whitelist(user_group: str) -> list:
    cursor.execute(f'SELECT url FROM whitelist WHERE user_group = {user_group}')
    wl = cursor.fetchall()
    return [w[0] for w in wl]


class Paths(Enum):
    URLS = Path('./data', 'urls.csv')
    CTX_MODEL = Path('./models', 'context_model.pkl')
    CATEGORIES_MODEL = Path('./models', 'categories_model.pkl')

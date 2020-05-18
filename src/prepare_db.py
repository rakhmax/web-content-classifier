import sqlite3


conn = sqlite3.connect('lists.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS blacklist(
    url TEXT NOT NULL UNIQUE,
    user_group TEXT NOT NULL
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS whitelist(
    url TEXT NOT NULL UNIQUE,
    user_group TEXT NOT NULL
)''')

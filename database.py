import sqlite3


conn = sqlite3.connect('mydb.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS banned_categories(
    categories TEXT NOT NULL UNIQUE,
    user_type INTEGER NOT NULL
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS blacklist(
    url TEXT NOT NULL UNIQUE,
    user_type INTEGER NOT NULL
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS whitelist(
    url TEXT NOT NULL UNIQUE,
    user_type INTEGER NOT NULL
)''')

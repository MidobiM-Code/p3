import requests
from bs4 import BeautifulSoup
import sqlite3

def fetch_tgju_price():
    try:
        url = "https://www.tgju.org/profile/price_dollar_rl"
        response = requests.get(url, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        price_tag = soup.find("span", {"id": "last"})
        if price_tag:
            price_str = price_tag.text.replace(",", "")
            return int(price_str)
    except:
        return None

def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            predicted_price REAL,
            model_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction_to_db(date, predicted_price, model_name):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('INSERT INTO predictions (date, predicted_price, model_name) VALUES (?, ?, ?)', 
              (date, predicted_price, model_name))
    conn.commit()
    conn.close()

def load_prediction_history():
    conn = sqlite3.connect("history.db")
    df = None
    try:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY date", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df
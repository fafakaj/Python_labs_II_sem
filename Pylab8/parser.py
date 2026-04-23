import requests
from bs4 import BeautifulSoup
import pandas as pd


itmo_news = []

DOMAIN = 'https://news.itmo.ru'
SEARCH_DOMAIN = 'https://news.itmo.ru/ru/search/?search='
query = 'нейротехнологии'

try:
    response = requests.get(SEARCH_DOMAIN + query)
    response.raise_for_status()
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")


    news = soup.find_all('li', class_='weeklyevent')

    for new in news:
        url = new.h4.a["href"]
        title = new.h4.get_text(strip=True)
        try:
            date = new.find_all('p')[-1].text
        except IndexError:
            date = None

        code = url.split('/')[-2]
        itmo_news.append({"Date": date, "Title": title, "URL": DOMAIN + url, "Code": code})

except requests.exceptions.RequestException as e:
    print(f"Ошибка при запросе {SEARCH_DOMAIN}: {e}")

except Exception as e:
    print(f"Ошибка при парсинге {SEARCH_DOMAIN}: {e}")


df = pd.DataFrame(itmo_news)
df.to_csv('itmo_news.csv', index=False)


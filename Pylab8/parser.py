import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import os

itmo_news_with_query = []

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
        itmo_news_with_query.append({
                "Date": date,
                "Title": title,
                "URL": DOMAIN + url,
                "Code": code
                })

except requests.exceptions.RequestException as e:
    print(f"Ошибка при запросе {SEARCH_DOMAIN}: {e}")

except Exception as e:
    print(f"Ошибка при парсинге {SEARCH_DOMAIN}: {e}")


df = pd.DataFrame(itmo_news_with_query)
df.to_csv('itmo_news_with_query.csv', index=False)


URL = 'https://news.itmo.ru/ru/main_news/'
total_pages = None

for i in range(1, 200, 1):
    time.sleep(random.randint(0, 1))
    data = requests.get(URL + str(i) + '/')
    page = BeautifulSoup(data.text, 'html.parser')

    next_page_href = page.find_all('div', {'class': 'pagination'})[0]\
                                    .find('ul')\
                                    .find_all('li')[1]\
                                    .find('a')['href']

    if next_page_href == '#':
        total_pages = i
        break

os.makedirs('news_content', exist_ok=True)

for i in range(1, total_pages + 1):
    time.sleep(random.randint(0, 1))

    page_news = []

    try:
        data = requests.get(URL + str(i) + '/')
        data.raise_for_status()
        page = BeautifulSoup(data.text, 'html.parser')

        triplet = page.find('ul', class_='triplet')
        if not triplet:
            continue
        news_headers = triplet.find_all('li')


        for _n in news_headers:
            href = _n.find('h4').find('a')['href']
            post_url = DOMAIN + href

            try:
                time.sleep(random.randint(0, 1))
                response = requests.get(post_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                code = post_url.split('/')[-2]

                title = soup.find('h1').text if soup.find('h1') else None

                date = None
                date_tag = soup.find('time')
                if date_tag:
                    date = date_tag.contents[0].strip()

                views = None
                views_tag = soup.find('span', class_='icon eye')
                if views_tag:
                    views = views_tag.text.strip()

                text = None
                text_tag = soup.find('div', class_='post-content')
                if text_tag:
                    text = text_tag.get_text(separator=' ', strip=True)

                tags = []
                tags_ul = soup.find('ul', class_='tags')
                if tags_ul:
                    for li in tags_ul.find_all('li'):
                        tags.append(li.get_text(strip=True))

                news_item = {
                    "Code": code,
                    "Title": title,
                    "Date": date,
                    "Views": views,
                    "Text": text,
                    "Tags": ', '.join(tags)
                }

                page_news.append(news_item)

            except requests.exceptions.RequestException as e:
                print(f"Ошибка при запросе {post_url}: {e}")

            except Exception as e:
                print(f"Ошибка при парсинге {post_url}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе страницы {i}: {e}")

    except Exception as e:
        print(f"Ошибка при парсинге страницы {i}: {e}")

    if page_news:
        df_page = pd.DataFrame(page_news)
        file_path = os.path.join('news_content', f"page_{i}.csv")
        df_page.to_csv(file_path, index=False)
        print(f'Файл успешно записан: page_{i}.csv')

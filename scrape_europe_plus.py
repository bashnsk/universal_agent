import requests
from bs4 import BeautifulSoup

def get_latest_airings(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Здесь нужно будет указать правильные селекторы для элементов, где находятся аиринги
    airings = []
    for item in soup.find_all('div', class_='airing-item'):
        title = item.find('h3').text.strip()
        time = item.find('span', class_='airing-time').text.strip()
        airings.append({'title': title, 'time': time})
    
    return airings

if __name__ == "__main__":
    url = "https://europeplus.ru/programms"
    latest_airings = get_latest_airings(url)
    for airing in latest_airings:
        print(airing)

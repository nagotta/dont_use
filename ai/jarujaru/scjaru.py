from bs4 import BeautifulSoup
import requests

load_url = "https://jarujaru.com/"
html = requests.get(load_url)
soup = BeautifulSoup(html.content, "html.parser")
topic = soup.find(class_="title[data-v-4e051451]")


for element in topic.find_all('span'):
    print(element.text)
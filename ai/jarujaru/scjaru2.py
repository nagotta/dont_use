import requests
import json
from pprint import pprint

title = []

# News
news_url = 'https://jarujaru.com/api/v1/news'
r = requests.get(news_url)
js = json.loads(r.content)
for item in js:
    title.append(item['title'])

# Posts
posts_url = 'https://jarujaru.com/api/v1/posts'
r = requests.get(posts_url)
js = json.loads(r.content)['data']
for item in js:
    title.append(item['title'])

pprint(title)


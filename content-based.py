import os
import json
from bs4 import BeautifulSoup, Comment
JOKE_PATHS = os.path.abspath(
    os.path.dirname(__file__)+'/Dataset/jokes/')
jokes = []
for f in os.listdir(JOKE_PATHS):
    if 'init' in f:
        soup = BeautifulSoup(open(JOKE_PATHS + '/' + f), 'lxml')
        jokes.append(soup.body.text.strip())
with open(os.path.abspath(
        os.path.dirname(__file__)+'/jokes.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(jokes))
    f.close()

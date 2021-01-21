import requests
from bs4 import BeautifulSoup
import warnings
import re

warnings.filterwarnings("ignore")

def GetHTML(link):
    #time.sleep(1)
    return requests.get(link).text

def CreateSongLinks(AlbomLink):
    return [re.sub(r'<a href=\"(.*)\"', r'\1', link) for link in re.findall(r'<a href=\"[a-zA-Z :/\.\-]+\-lyrics\"', GetHTML(AlbomLink))]

print(CreateSongLinks('https://genius.com/albums/Zemfira/Forgive-me-my-love'))

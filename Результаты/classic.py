from bs4 import BeautifulSoup
import time
import re
import os
import requests
page = 1
corpr = 'http://processing.ruscorpora.ru/search.xml?env=alpha&mode=poetic&sort=i_grcreated&text=meta&dpp=50&spd=1&doc_author=%D0%90.%20%D0%90.%20%D0%91%D0%BB%D0%BE%D0%BA'

def GetHTML(link):
    print('Sleep')
    time.sleep(4)
    print('continue')
    return requests.get(link).text

def GetNextPage(link):
    if BeautifulSoup(GetHTML(link)).find_all('p', {'class':'pager'})[-1].find_all('a')[-1].text == "следующая страница":
        return 'http://processing.ruscorpora.ru'+ BeautifulSoup(GetHTML(link)).find_all('p', {'class':'pager'})[-1].find_all('a')[-1].get('href')
    else:
        return 'end'
            
def GetSong(link):
    global page
    print('Loading page - ', page)
    with open(r'C:/Users/epish/Desktop/учеба/ДИПЛОМ/КлассикаТексты/allbase.txt', 'a', encoding = 'utf-8') as f:
        poems = BeautifulSoup(GetHTML(link)).find_all('a', {'class':'b-kwic-expl'})
        for poem in poems:
            poem = BeautifulSoup(GetHTML('http://processing.ruscorpora.ru/'+poem.get('href'))).find_all('table')
            try:
                f.write('\n'.join(poem[-1].text.split('\n')[:-1]))
            except:
                pass
    nextPage = GetNextPage(link)
    print(nextPage)
    if nextPage != 'end':
        page += 1
        GetSong(nextPage)
    else:
        print('Done')
        
GetSong(corpr) 

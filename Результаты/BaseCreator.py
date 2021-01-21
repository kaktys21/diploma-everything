import requests
import re
from bs4 import BeautifulSoup
import time
import warnings
import os
from tkinter import *
import nltk
import pymorphy2
from threading import Thread
from nltk import ngrams, skipgrams
from scipy.spatial import distance

startTime = time.time()
warnings.filterwarnings("ignore")
faledLinks = []
try:
    inDirectory = ['C:/Users/epish/Desktop/учеба/ДИПЛОМ/ПопТексты/Input/', 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/РэпТексты/Input/']
    outDirectory = ['C:/Users/epish/Desktop/учеба/ДИПЛОМ/ПопТексты/Output/', 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/РэпТексты/Output/']
    baseDirectory = ['C:/Users/epish/Desktop/учеба/ДИПЛОМ/ПопТексты/Bases/', 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/РэпТексты/Bases/']
except:
    print("Directory faled")

    
'''------------------------------------------------- К Н О П О Ч К И -------------------------------------------------------------'''
#Вызывает создание базы с нуля
def CreateButton():
    if os.path.exists(outDirectory[0]+'allbase.txt') != True and os.path.exists(outDirectory[1]+'allbase.txt') == True:
        BaseDownloader(InFileAll(inDirectory[0]), 0)
    elif os.path.exists(outDirectory[1]+'allbase.txt') != True and os.path.exists(outDirectory[0]+'allbase.txt') == True:
        BaseDownloader(InFileAll(inDirectory[1]), 1)
    elif os.path.exists(outDirectory[1]+'allbase.txt') != True and os.path.exists(outDirectory[0]+'allbase.txt') == True:
        for directoryNum in len(range(inDirectory)):
            BaseDownloader(InFileAll(inDirectory[directoryNum]), directoryNum)
    
    text.insert('end', 'All bases are created\n')
    text.see("end")

def LemmaCorpButton():
    if os.path.exists(baseDirectory[0]+'lemmas.txt') != True and os.path.exists(baseDirectory[1]+'lemmas.txt') == True:
        MorphCorpras(baseDirectory[0])
    elif os.path.exists(baseDirectory[1]+'lemmas.txt') != True and os.path.exists(baseDirectory[0]+'lemmas.txt') == True:
        MorphCorpras(baseDirectory[1])
    elif os.path.exists(baseDirectory[1]+'lemmas.txt') != True and os.path.exists(baseDirectory[0]+'lemmas.txt') == True:
        for directory in baseDirectory:
            MorphCorpras(directory)
    
    text.insert('end', 'All lemma bases are created\n')
    text.see("end")
    
#Вызывает очистку базы
def CleanButton(*mode):
    global outDirectory
    if len(mode) == 0:
        if os.path.exists(baseDirectory[0]+'cleanbase.txt') != True and os.path.exists(baseDirectory[1]+'cleanbase.txt') == True:
            Cleaner(outDirectory[0])
        elif os.path.exists(baseDirectory[0]+'cleanbase.txt') == True and os.path.exists(baseDirectory[1]+'cleanbase.txt') != True:
            Cleaner(outDirectory[1])
        elif os.path.exists(baseDirectory[0]+'cleanbase.txt') != True and os.path.exists(baseDirectory[1]+'cleanbase.txt') != True:
            for style in range(len(outDirectory)):
                Cleaner(style)
        text.insert('end', 'All bases are cleaned\n')
        text.see("end")

#Вызывает догрузку файлов
def RefreshButton():
    for directoryNum in range(len(inDirectory)):
        notInBase = [name for name in [title for title in list(filter(lambda x: x.endswith('txt'), os.listdir(inDirectory[directoryNum])))]
                     if 'out'+name not in list(filter(lambda x: x.startswith('out'), os.listdir(outDirectory[directoryNum])))]
        text.insert('end', inDirectory[directoryNum]+':\n')
        if len(notInBase) != 0:
            flag = 'worked'
            for name in notInBase:
                text.insert('end', name + ' is not uploaded\n')            
                text.see('end')
            refresh = Tk()
            refresh.title('Refresher')
            refresh.geometry('200x100+300+200')
            btnRefresh = Label(refresh, text = 'Upload new files?')
            btnYes = Button(refresh, text = 'yes', width = 10, command = lambda: Refresher(directoryNum,notInBase, refresh))
            btnNo = Button(refresh, text = 'close', width = 10, command = refresh.destroy)
            btnYes.place(x = 10, y = 50)
            btnNo.place(x = 110, y = 50)
            btnRefresh.pack()
            refresh.mainloop()
        else:
            try:
                if flag == 'worked':
                    refresh.destroy()
                    text.insert('end','All files are uploaded\n')
            except UnboundLocalError:
                text.insert('end','All files are uploaded\n')

#Закрывает окно приложения
def CloseButton():
    root.destroy()
    
'''------------------------------------------------ И Н Ф О Р М А Ц И Я -------------------------------------------------'''

def FullInformation():
    global inDirectory, outDirectory, baseDirectory
    informationText = 'Base information'
    artists = len(list(filter(lambda x: x.startswith('out'), os.listdir(outDirectory[0])))+
                  list(filter(lambda x: x.startswith('out'), os.listdir(outDirectory[1]))))
    informationText += '\nArtists:\t'+str(artists)
    alboms = 0
    artists = 0
    songs = 0
    words = 0
    lemmas = 0
    for i in range(len(inDirectory)):
        for artist in InFileAll(inDirectory[i]):
            alboms += len(FileReader(inDirectory[i],artist))
        try:
            songs += len(FileReader(baseDirectory[i], 'clean.txt'))
        except FileNotFoundError:
            text.insert('end', outDirectory[i]+' Need cleaning!\n')
        try:
            word = FileReader(baseDirectory[i], 'lemmas.txt')
            words += len(word)
            lemmas += len(list(set(word)))
        except:
            text.insert('end', baseDirectory[i]+' Need lemmatization!\n')
        
    informationText += '\nAlbums:\t'+str(alboms)        
    informationText += '\nSongs:\t'+str(songs)
    informationText += '\nWords:\t'+str(words)
    informationText += '\nLemmas:\t'+str(lemmas)
    information['text'] = informationText

    
'''--------------------------------------------- Р А Б О Т А  С  Б А З О Й -------------------------------------------------'''
def FileReader(directory, filename):
    with open(directory+filename, 'r', encoding = 'utf-8') as inFile:
        return [line for line in inFile]

def FileWriter(directory, filename, corpra):
    with open(directory+filename, 'w', encoding = 'utf-8') as outFile:
        for line in corpra:
            outFile.write(line+'\n')

#Возвращает html код страницы по ссылке
def GetHTML(link):
    #time.sleep(1)
    return requests.get(link).text

#Возвращает всё о песне по ссылке
def GetSong(link): 
    soup = BeautifulSoup(GetHTML(link))
    Artist = soup.find('div', {'class':'song_album-info'}).find('a', {'class': 'song_album-info-artist'}).text
    SongTitle = soup.find('div', {'class':'header_with_cover_art'}).find('h1').text
    AlbumTitle = soup.find('div', {'class':'song_album-info'}).find('a').get('title')
    SongText = [re.sub(r'\xa0', ' ', line2) for line2 in
                [re.sub(r'\[.*\]', '', line) for line in soup.find('div', {'class':'lyrics'}).text.split('\n') if len(line) > 0]
                if len(line2) > 0]
    return (AlbumTitle, SongTitle, SongText)

#Возвращает список всех песен в альбоме по ссылке на альбом
def CreateSongLinks(AlbomLink):
    return [re.sub(r'<a href=\"(.*)\"', r'\1', link) for link in re.findall(r'<a href=\"[a-zA-Z 2:/\.\-]+\-lyrics\"', GetHTML(AlbomLink))]

#Возвращает альбом по списку ссылок на каждую песню
def AlbomMaker(SongsLinks):
    albom = []
    linkNum = 0
    while linkNum < len(SongsLinks):
        try:
            albom.append(GetSong(SongsLinks[linkNum]))
            linkNum += 1
        except:
            global faledLinks
            faledLinks.append(SongsLinks[linkNum])
            linkNum += 1
    return albom

#Возвращает все файлы в директории
def InFileAll(directory):
    return list(filter(lambda x: x.endswith('txt'), os.listdir(directory)))

#Возвращает список ссылок на альбомы по имени файла и директории
def AlbomLinksCreator(name, directory):
    return [re.sub(r'\n', '', link) for link in open(directory+'/'+name, 'r', encoding = 'utf-8')]
    

#Догружает в базу песни по списку ссылок на альбомы, имени файла и директории
def BaseCreator(AlbomLinks, name, directoryNum):
    flag = 1
    allBase = open(outDirectory[directoryNum]+'allbase.txt', 'w', encoding = 'utf-8')
    for link in AlbomLinks:
        print('Loading album ', flag, ' of ', len(AlbomLinks), sep = '', end = '\t')
        albom = AlbomMaker(CreateSongLinks(link))
        personalBase = open(outDirectory[directoryNum]+'/out'+name, 'a', encoding = 'utf-8')
        for song in albom:
            print('-', end = '')
            personalBase.write(song[0]+'\n'+song[1]+'\n')
            for lines in song[2]:
                personalBase.write(lines+'\n')
                allBase.write(lines+'\n')
            personalBase.write('\n\n')
            allBase.write('\n\n')
        flag += 1
        print('\tDone')

#Запускает создание базы по заданной директории и входящим файлам
def BaseDownloader(infiles, directoryNum):
    global inDirectory, outDirectory, faledLinks
    Number = 1
    for name in infiles:
        print(Number, 'of', len(infiles),name, sep = ' ', end = '\n')
        BaseCreator(AlbomLinksCreator(name, inDirectory[directoryNum]), name, directoryNum)
        print("--- %s seconds ---", (time.time() - startTime), end = '\n\n')
        Number += 1
    faled = open(outDirectory[directoryNum]+'/faled.txt', 'w', encoding = 'utf-8')
    for i in faledLinks:
        faled.write(i+'\n')
    falelLinks = []
    text.insert('end', 'Base downloaded\n')

#Убирает повторы и приводит базу к виду списка русских слов в заданной директории
def Cleaner(style):
    global baseDirectory, outDirectory
    base = FileReader(outDirectory[style],'/allbase.txt')
    result = [re.sub(r'припев', '', song) for song in [song.lower() for song in [re.sub(r'[,\'\"\-«:»()—]', '', song) for song in
             [' '.join(song.split()) for song in [re.sub(r'\n', '', song) for song in
                                                  [re.sub(r'\u2005', ' ', song) for song in list(set(' '.join(base).split('\n \n')))]]]] if len(re.findall(r'[a-zA-Z]+', song)) == 0]]
    FileWriter(baseDirectory[style], '/clean.txt', result)
    text.insert('end', outDirectory[style]+'\tcleaned\n')

#Дописывает потеряшек
def Refresher(directoryNum, notInBase, refresh):
    BaseDownloader(notInBase, directoryNum)
    Cleaner(outDirectory[directoryNum])
    RefreshButton()

#Создает базу Лемм
def MorphCorpras(directory):
    text.insert('end', directory+' in progres...\n')
    lines = FileReader(directory, 'clean.txt')
    line = re.sub(r'[!?.]+', '', re.sub(r'\n', ' ', ''.join(lines))).split()
    morph = pymorphy2.MorphAnalyzer()
    corpra = [morph.parse(word)[0].normal_form for word in line if len(word) > 0 and word not in ['пр','припев','куплет']]
    FileWriter(directory, 'lemmas.txt', corpra)

'''-------------------------------------------- П О Д С Ч Е Т  Ч А С Т О Т Н О С Т И -----------------------------------'''

def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num

def SentenceSplitter(text):
    split_regex = re.compile(r'[.|!|?|…]')
    return list(filter(lambda t: t, [t.strip() for t in split_regex.split(text)]))

def AverageReducedFrequency(word, n, corpra, L):
    f = len(n)
    d = [n[p + 1] - n[p] for p in range(len(n) - 1)]
    d.append(n[0] + len(corpra) - n[len(n) - 1])
    segmentLen = int_r(L/f)
    summa = 0
    for i in range(len(d)):
        summa += min(d[i],segmentLen)
    print(word, round(summa/segmentLen, 3), sep = '\t')
    return (word, round(summa/segmentLen, 3))


def ARFCounter(directory):
    corpra = [re.sub(r'\n', '', word) for word in FileReader(directory, 'lemmas.txt')]
    words = {word: [] for word in list(nltk.FreqDist(corpra).keys())}
    for i in range(len(corpra)):
        words[corpra[i]].append(i + 1)
    L = len(corpra)
    result = [AverageReducedFrequency(word, words[word], corpra, L) for word in words]
    result.sort(key = lambda x: x[1])
    result = [str(pair[0])+'\t'+str(pair[1]) for pair in result[::-1]]
    FileWriter(directory, 'arf.txt', result)

# Делает частотные списки лемм    
def ARFMaker():
    thread1 = Thread(target = ARFCounter, args = (baseDirectory[1],))
    thread2 = Thread(target = ARFCounter, args = (baseDirectory[0],))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

def SkipGramms(directory):
    base = FileReader(outDirectory[directory], 'allbase.txt')
    corpra = SentenceSplitter(' '.join(list(set([line4 for line4 in [re.sub(r'[,«\-»:—()]', '', line3) for line3 in
                                 [re.sub(r'\u2005', ' ', line2) for line2 in
                                  base]] if len(re.findall(r'[a-z]+', line4)) == 0 and len(line4) > 0]))).lower())
    result = []    
    for line in corpra:
        skip = list(nltk.skipgrams(line.split(),2,1))
        for skipi in skip:
            result.append(skipi)
    freq = nltk.FreqDist(result)
    l = len(result)
    freqDist = [(i,freq[i]/l*1000000) for i in list(freq.keys())]
    freqDist.sort(key = lambda x: x[1])
    freqDist = freqDist[::-1]
    result = [str(freqDist[i][0])+'\t'+str(freqDist[i][1]) for i in range(len(freqDist))]
    FileWriter(baseDirectory[directory], 'skipgram.txt', result)

def Ngrams(directory):
    base = FileReader(outDirectory[directory], 'allbase.txt')
    lines = [line for line in [re.sub(r'\u2005', ' ', song) for song in ''.join(base).lower().split('\n\n')] if len(re.findall(r'[a-zA-Z]+', line)) == 0]
    for p in range(2,6):
        ngramList = []
        for song in lines:
            for ng in list(nltk.ngrams(song,p)):
                ngramList.append(ng)
        l = len(ngramList)
        freq = nltk.FreqDist(ngramList)
        freqDist = [(n,freq[n]/ l * 1000000) for n in list(freq.keys())]
        freqDist.sort(key = lambda x: x[1])
        freqDist = freqDist[::-1]
        result = [str(freqDist[i][0])+'\t'+str(freqDist[i][1]) for i in range(len(freqDist))]
        FileWriter(baseDirectory[directory], str(p)+'ngrams.txt', result)




# Принимает два список из двух файлов и печатает словарь "слово":[попса, рэп]
def Vectors(filename):
    f = [pos.split('\t') for pos in [re.sub(r'\n', '', line) for line in FileReader(baseDirectory[0], filename)]]
    allDict = {a[0]:float(a[1]) for a in f}
    for i in range(1, len(baseDirectory)):
        f2 = [pos.split('\t') for pos in [re.sub(r'\n', '', line) for line in FileReader(baseDirectory[i], filename)]]
        dictNew = {a[0]:float(a[1]) for a in f2}
        for word in allDict:
            if word in dictNew:
                allDict[word] = [allDict[word], dictNew[word]]
            else:
                allDict[word] = [allDict[word], 0]
        for word in dictNew:
            if word not in allDict:
                allDict[word] = [0, dictNew[word]]
    result = [word+'\t'+'\t\t'.join(map(str, allDict[word])) for word in list(allDict.keys())]
    FileWriter('C:/Users/epish/Desktop/учеба/ДИПЛОМ/Results/', filename, result)

'''--------------------------------------------- Р А С С Т О Я Н И Я -------------------------------------------'''

def Distances():
    files = os.listdir(r'C:/Users/epish/Desktop/учеба/ДИПЛОМ/Results')
    for file in files:
        lines = FileReader('C:/Users/epish/Desktop/учеба/ДИПЛОМ/Results/', file)
        s = [[],[]]
        for line in lines:
            for i in range(1,3):
                s[i - 1].append(float(line.split()[-i]))
        with open(r'C:/Users/epish/Desktop/учеба/ДИПЛОМ/Results/Final.txt', 'a', encoding = 'utf-8') as out:
            out.write(file[:-4]+'\n'+'косинусное расстояние'+'\t\t'+str(distance.cosine(s[0], s[1]))+'\n')
            out.write('манхэттенское расстояние'+'\t'+str(distance.cityblock(s[0], s[1]))+'\n')
            out.write('евклидово расстояние'+'\t\t'+str(distance.euclidean(s[0], s[1]))+'\n')

def KeyWords():
    fwords = []
    for directory in baseDirectory:
        words = FileReader(directory, 'arf.txt')
        fwords.append(words[:5001])
    fwords[0] = [pos.split() for pos in fwords[0]]
    fwords[1] = [pos.split() for pos in fwords[1]]
    allwords = {fwords[0][i][0]:[float(fwords[0][i][1])] for i in range(len(fwords[0]))}
    rapwords = {fwords[1][i][0]:float(fwords[1][i][1]) for i in range(len(fwords[1]))}
    for word in allwords:
        if word in rapwords:
            allwords[word].append(rapwords[word])
        else:
            allwords[word].append(0)
    for word in rapwords:
        if word not in allwords:
            allwords[word] = [0, rapwords[word]]
    for n in [1,10,100,1000]:
        popkeywords = [(word,(allwords[word][0] + n) / (allwords[word][1] + n)) for word in list(allwords.keys())]
        rapkeywords = [(word,(allwords[word][1] + n) / (allwords[word][0] + n)) for word in list(allwords.keys())]
        summ = 0
        for N in range(5000):
            summ += max(popkeywords[N][1], rapkeywords[N][1])
            dist = summ/1000
        print("При n равном ", n, ":\t", dist, sep = '')
        if os.path.exists(baseDirectory[0]+str(n)+'keywords.txt') != True:
            popkeywords.sort(key = lambda x: x[1])
            popkeywords = popkeywords[::-1]
            result = [word[0]+'\t'+str(word[1]) for word in popkeywords]
            FileWriter(baseDirectory[0], str(n)+'keywords.txt', result)
        if os.path.exists(baseDirectory[1]+str(n)+'keywords.txt') != True:
            rapkeywords.sort(key = lambda x: x[1])
            rapkeywords = rapkeywords[::-1]
            result = [word[0]+'\t'+str(word[1]) for word in rapkeywords]
            FileWriter(baseDirectory[1], str(n)+'keywords.txt', result)            
    
KeyWords()

'''--------------------------------------------- О К Н О  "П Р И Л О Ж Е Н И Я" -------------------------------------------'''

root = Tk()
root.title("Base options")
sizex = 600
sizey = 400
posx  = 100
posy  = 100
root.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))
root['bg'] = 'black'
mainMenu = Menu()

text = Text(width = 75, height = 10, bg = 'black', fg = 'white')
text.pack(side = BOTTOM)

file_menu = Menu()
file_menu.add_command(label="Create", command = CreateButton)
file_menu.add_command(label="Clean", command = CleanButton)
file_menu.add_command(label="Refresh", command = RefreshButton)
file_menu.add_command(label="Lemmas", command = LemmaCorpButton)
file_menu.add_separator()
file_menu.add_command(label="Exit", command = CloseButton)

edit_menu = Menu()
edit_menu.add_command(label = 'Refresh', command = FullInformation)

mainMenu.add_cascade(label="Base", menu = file_menu)
mainMenu.add_cascade(label="Information", menu = edit_menu)
mainMenu.add_cascade(label="View")

root.config(menu = mainMenu)

information = Label(root, bg = 'black', fg = 'white', font = 'Times 14', height = 6, width = 14, anchor = NW)
try:
    information['text'] = FullInformation()
    information.place(x = 0, y = 0)
except:
    information['text'] = 'No base found'
    information.place(x = 0, y = 0)
    
root.mainloop()

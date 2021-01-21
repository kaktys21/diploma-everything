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
from prettytable import PrettyTable

startTime = time.time()
faledLinks = []
inDirectory = ['C:/Users/epish/Desktop/учеба/ДИПЛОМ/ПопТексты/Input/', 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/РэпТексты/Input/']
outDirectory = ['C:/Users/epish/Desktop/учеба/ДИПЛОМ/ПопТексты/Output/', 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/РэпТексты/Output/']
baseDirectory = ['C:/Users/epish/Desktop/учеба/ДИПЛОМ/ПопТексты/Bases/', 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/РэпТексты/Bases/']
resultDirectory = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/Results/'

'''------------------------------------------------- К Н О П О Ч К И -------------------------------------------------------------'''

#Вызывает создание базы с нуля
def CreateButton():
    for i in range(len(outDirectory)):
        if len(os.listdir(outDirectory[i])) == 0:
               BaseDownloader(InFileAll(inDirectory[i]), inDirectory, outDirectory, i)
    text.insert('end', 'All bases are created\n')
    text.see("end")

# Вызывает создание базы лемм
def LemmaCorpButton():
    for i in range(len(baseDirectory)):
        if os.path.exists(baseDirectory[i]+'lemmas.txt') != True:
            MorphCorpras(baseDirectory[i])    
    text.insert('end', 'All lemma bases are created\n')
    text.see("end")

#Вызывает очистку базы
def CleanButton(*mode):
    if len(mode) == 0:
        for i in range(len(baseDirectory)):
            if os.path.exists(baseDirectory[i]+'cleanbase.txt'):
                Cleaner(baseDirectory, outDirectory, i)
        text.insert('end', 'All bases are cleaned\n')
        text.see("end")

#Вызывает догрузку файлов в первоначальную базу
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
        
    informationText += '\nAlbums:\t' + str(alboms) + '\nSongs:\t' + str(songs) + '\nWords:\t' + str(words) + '\nLemmas:\t' + str(lemmas)
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
    soup = BeautifulSoup(GetHTML(link), features="lxml")
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
def BaseCreator(AlbomLinks, outDir, name, directoryNum):
    flag = 1
    allBase = open(outDir[directoryNum]+'allbase.txt', 'w', encoding = 'utf-8')
    for link in AlbomLinks:
        print('Loading album ', flag, ' of ', len(AlbomLinks), sep = '', end = '\t')
        albom = AlbomMaker(CreateSongLinks(link))
        personalBase = open(outDir[directoryNum]+'/out'+name, 'a', encoding = 'utf-8')
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
def BaseDownloader(infiles, inDir, outDir, directoryNum):
    global faledLinks
    Number = 1
    for name in infiles:
        print(Number, 'of', len(infiles),name, sep = ' ', end = '\n')
        BaseCreator(AlbomLinksCreator(name, inDir[directoryNum]), outDir, name, directoryNum)
        print("--- %s seconds ---", (time.time() - startTime), end = '\n\n')
        Number += 1
    faled = open(outDir[directoryNum]+'/faled.txt', 'w', encoding = 'utf-8')
    for i in faledLinks:
        faled.write(i+'\n')
    falelLinks = []
    text.insert('end', 'Base downloaded\n')

#Убирает повторы и приводит базу к виду списка русских слов в заданной директории
def Cleaner(baseDir, outDir, style):
    base = FileReader(outDir[style],'allbase.txt')
    result = [re.sub(r'припев', '', song) for song in [song.lower() for song in [re.sub(r'[,\'\"\-«:»()—]', '', song) for song in
             [' '.join(song.split()) for song in [re.sub(r'\n', '', song) for song in
                                                  [re.sub(r'\u2005', ' ', song) for song in list(set(' '.join(base).split('\n \n')))]]]] if len(re.findall(r'[a-zA-Z]+', song)) == 0]]
    FileWriter(baseDir[style], 'clean.txt', result)
    text.insert('end', outDir[style]+'\tcleaned\n')

#Дописывает потеряшек
def Refresher(directoryNum, notInBase, refresh):
    BaseDownloader(notInBase, inDirectory, outDirectory, directoryNum)
    Cleaner(baseDirectory, outDirectory, outDirectory[directoryNum])
    RefreshButton()

#Создает базу Лемм
def MorphCorpras(directory):
    if os.path.exists(directory + 'lemmas.txt') != True:
        text.insert('end', directory + 'Lemmas in progres...\n')
        lines = FileReader(directory, 'clean.txt')
        line = re.sub(r'[!?.]+', '', re.sub(r'\n', ' ', ''.join(lines))).split()
        morph = pymorphy2.MorphAnalyzer()
        corpra = [morph.parse(word)[0].normal_form for word in line if len(word) > 0 and word not in ['пр','припев','куплет']]
        FileWriter(directory, 'lemmas.txt', corpra)
        text.insert('end', 'Done...\n')

'''-------------------------------------------- П О Д С Ч Е Т  Ч А С Т О Т Н О С Т И -----------------------------------'''

def int_r(num):
    return int(num + (0.5 if num > 0 else -0.5))

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
    #print(word, round(summa/segmentLen, 3), sep = '\t')
    return (word, round(summa/segmentLen, 3))

# Делает частотные списки по директории
def ARFCounter(directory):
    if os.path.exists(directory + 'arf.txt') != True:
        corpra = [re.sub(r'\n', '', word) for word in FileReader(directory, 'lemmas.txt')]
        words = {word: [] for word in list(nltk.FreqDist(corpra).keys())}
        for i in range(len(corpra)):
            words[corpra[i]].append(i + 1)
        L = len(corpra)
        result = [AverageReducedFrequency(word, words[word], corpra, L) for word in words]
        result.sort(key = lambda x: x[1])
        result = [str(pair[0])+'\t'+str(pair[1]) for pair in result[::-1]]
        FileWriter(directory, 'arf.txt', result)

# Делает частотные списки для первоначальной базы  
def ARFMaker():
    thread1 = Thread(target = ARFCounter, args = (baseDirectory[1],))
    thread2 = Thread(target = ARFCounter, args = (baseDirectory[0],))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

# Создаёт скипграммы по заданной директории
def SkipGramms(inDir, outDir):
    for directory in range(len(inDir)):
        if os.path.exists(outDir[directory]+'skipgram.txt') != True:
            base = FileReader(inDir[directory], 'allbase.txt')
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
            FileWriter(outDir[directory], 'skipgram.txt', result)

def Ngrams(inDir, outDir):
    for directory in range(len(inDir)):
        files = os.listdir(outDir[directory])
        for n in range(2,6):
            if str(n)+'ngrams.txt' not in files:
                base = FileReader(inDir[directory], 'allbase.txt')
                lines = [line for line in [re.sub(r'\u2005', ' ', song) for song in ''.join(base).split('\n\n')] if len(re.findall(r'[a-zA-Z]+', line)) == 0]
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
                    FileWriter(outDir[directory], str(p)+'ngrams.txt', result)
    
    
# Принимает два список из двух файлов и печатает словарь "слово":[попса, рэп]
def Vectors(filename, inDir, outDir):
    f = [pos.split('\t') for pos in [re.sub(r'\n', '', line) for line in FileReader(inDir[0], filename)]]
    allDict = {a[0]:float(a[1]) for a in f}
    for i in range(1, len(inDir)):
        f2 = [pos.split('\t') for pos in [re.sub(r'\n', '', line) for line in FileReader(inDir[i], filename)]]
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
    FileWriter(outDir, 'res'+filename, result)


'''--------------------------------------------- Р А С С Т О Я Н И Я -------------------------------------------'''

def Distances(resultDir):
    files = list(filter(lambda x: x.startswith('res'), os.listdir(resultDir)))
    for file in files:
        lines = FileReader(resultDir, file)
        s = [[],[]]
        for line in lines:
            for i in range(1,3):
                s[i - 1].append(float(line.split()[-i]))
        with open(resultDir+'Final.txt', 'a', encoding = 'utf-8') as out:
            out.write(file[3:-4]+'\n'+'косинусное расстояние'+'\t\t'+str(distance.cosine(s[0], s[1]))+'\n')
            out.write('манхэттенское расстояние'+'\t'+str(distance.cityblock(s[0], s[1]))+'\n')
            out.write('евклидово расстояние'+'\t\t'+str(distance.euclidean(s[0], s[1]))+'\n')

def KeyWords(baseDir, resultDir):
    fwords = []
    for directory in baseDir:
        words = FileReader(directory, 'arf.txt')
        if len(words) > 5000:
            fwords.append(words[:5001])
        else:
            fwords.append(words)
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
        summ = [max(popkeywords[N][1], rapkeywords[N][1]) for N in range(len(popkeywords))]
        dist = sum(summ)/len(summ)
        with open(resultDir+'KeyWordFinal.txt', 'a',encoding = 'utf-8') as f:
            f.write("При n равном "+str(n)+":\t"+str(dist)+'\n')
        if os.path.exists(baseDir[0]+str(n)+'keywords.txt') != True:
            popkeywords.sort(key = lambda x: x[1])
            popkeywords = popkeywords[::-1]
            result = [word[0]+'\t'+str(word[1]) for word in popkeywords]
            FileWriter(baseDir[0], str(n)+'keywords.txt', result)
        if os.path.exists(baseDir[1]+str(n)+'keywords.txt') != True:
            rapkeywords.sort(key = lambda x: x[1])
            rapkeywords = rapkeywords[::-1]
            result = [word[0]+'\t'+str(word[1]) for word in rapkeywords]
            FileWriter(baseDir[1], str(n)+'keywords.txt', result)

'''------------------------------------------------ Д Е Л А Е Т  Ч А С Т О Т Ы  Д Л Я  П Е Р В О Н А Ч А Л Ь Н О Й  Б А З Ы ----------------------------'''

if os.path.exists(baseDirectory[0]+'arf.txt') != True:
    ARFMaker()
if os.path.exists(baseDirectory[0]+'skipgram.txt') != True:
    SkipGramms(outDirectory, baseDirectory)
if os.path.exists(resultDirectory+'resarf.txt') != True:
    Vectors('arf.txt', baseDirectory, resultDirectory)
if os.path.exists(resultDirectory+'resskipgram.txt') != True:
    Vectors('skipgram.txt', baseDirectory, resultDirectory)
Ngrams(outDirectory, baseDirectory)
for i in range(2,6):
    if os.path.exists(resultDirectory+'res'+str(i)+'ngrams.txt') != True:
        Vectors(str(i)+'ngrams.txt', baseDirectory, resultDirectory)
if os.path.exists(resultDirectory+'KeyWordFinal.txt') != True:
    KeyWords(baseDirectory, resultDirectory)
if os.path.exists(resultDirectory+'Final.txt') != True:
    Distances(resultDirectory)


'''------------------------------------------------------ П Р О В Е Р К А  Н О В Ы Х  Т Е К С Т О В -----------------------------'''

def StyleFinder(event, resultLabel,resultLabel2, final):
    newDirectory = ['/'.join(str(dirEntry.get()).split('\\'))+'/']
    dirEntry.delete(0, END)
    os.mkdir(newDirectory[0]+'Output/')
    os.mkdir(newDirectory[0]+'Bases/')
    outNewDirectory = [newDirectory[0] + 'Output/']
    baseNewDirectory = [newDirectory[0] + 'Bases/']
    BaseDownloader(InFileAll(newDirectory[0]), newDirectory, outNewDirectory, 0)
    Cleaner(baseNewDirectory, outNewDirectory, 0)
    MorphCorpras(baseNewDirectory[0])
    ARFCounter(baseNewDirectory[0])
    Ngrams(outNewDirectory, baseNewDirectory)
    SkipGramms(outNewDirectory, baseNewDirectory)
    baseStyleCheck = [[baseNewDirectory[0], baseDirectory[0]],[baseNewDirectory[0], baseDirectory[1]]]
    res = ['ResultPop/', 'ResultRap/']
    for i in range(len(baseStyleCheck)):
        os.mkdir(newDirectory[0]+res[i])
        resultNewDirectory = newDirectory[0] + res[i]
        Vectors('arf.txt', baseStyleCheck[i], resultNewDirectory)
        Vectors('skipgram.txt', baseStyleCheck[i], resultNewDirectory)
        for n in range(2,6):
            Vectors(str(n)+'ngrams.txt', baseStyleCheck[i], resultNewDirectory)
        KeyWords(baseStyleCheck[i], resultNewDirectory)
        Distances(resultNewDirectory)
    ResultWriter(newDirectory[0], resultLabel,resultLabel2, final)

def StyleFinder2(event, textTaker, song, resultLabel,resultLabel2, final):
    with open(r'C:/Users/epish/Desktop/учеба/ДИПЛОМ/text/Output/allbase.txt', 'w', encoding = 'utf-8') as inf:
        inf.write(song.get('1.0', END))
    textTaker.destroy()
    newDirectory = ['C:/Users/epish/Desktop/учеба/ДИПЛОМ/text/']
    os.mkdir(newDirectory[0]+'Bases/')
    outNewDirectory = [newDirectory[0] + 'Output/']
    baseNewDirectory = [newDirectory[0] + 'Bases/']
    Cleaner(baseNewDirectory, outNewDirectory, 0)
    MorphCorpras(baseNewDirectory[0])
    ARFCounter(baseNewDirectory[0])
    Ngrams(outNewDirectory, baseNewDirectory)
    SkipGramms(outNewDirectory, baseNewDirectory)
    baseStyleCheck = [[baseNewDirectory[0], baseDirectory[0]],[baseNewDirectory[0], baseDirectory[1]]]
    res = ['ResultPop/', 'ResultRap/']
    for i in range(len(baseStyleCheck)):
        os.mkdir(newDirectory[0]+res[i])
        resultNewDirectory = newDirectory[0] + res[i]
        Vectors('arf.txt', baseStyleCheck[i], resultNewDirectory)
        Vectors('skipgram.txt', baseStyleCheck[i], resultNewDirectory)
        for n in range(2,6):
            Vectors(str(n)+'ngrams.txt', baseStyleCheck[i], resultNewDirectory)
        KeyWords(baseStyleCheck[i], resultNewDirectory)
        Distances(resultNewDirectory)
    ResultWriter(newDirectory[0], resultLabel,resultLabel2, final)

def StyleFinder3(event, resultLabel,resultLabel2, final):
    newDirectory = ['/'.join(str(dirEntry2.get()).split('\\'))+'/']
    dirEntry2.delete(0, END)
    os.mkdir(newDirectory[0]+'Bases/')
    outNewDirectory = [newDirectory[0] + 'Output/']
    baseNewDirectory = [newDirectory[0] + 'Bases/']
    Cleaner(baseNewDirectory, outNewDirectory, 0)
    MorphCorpras(baseNewDirectory[0])
    ARFCounter(baseNewDirectory[0])
    Ngrams(outNewDirectory, baseNewDirectory)
    SkipGramms(outNewDirectory, baseNewDirectory)
    baseStyleCheck = [[baseNewDirectory[0], baseDirectory[0]],[baseNewDirectory[0], baseDirectory[1]]]
    res = ['ResultPop/', 'ResultRap/']
    for i in range(len(baseStyleCheck)):
        os.mkdir(newDirectory[0]+res[i])
        resultNewDirectory = newDirectory[0] + res[i]
        Vectors('arf.txt', baseStyleCheck[i], resultNewDirectory)
        Vectors('skipgram.txt', baseStyleCheck[i], resultNewDirectory)
        for n in range(2,6):
            Vectors(str(n)+'ngrams.txt', baseStyleCheck[i], resultNewDirectory)
        KeyWords(baseStyleCheck[i], resultNewDirectory)
        Distances(resultNewDirectory)
    ResultWriter(newDirectory[0], resultLabel,resultLabel2, final)
    
def TextTaker():
    textTaker = Tk()
    textTaker.title("Songs text")
    textTaker['bg'] = 'black'
    sizex = 403
    sizey = 703
    posx  = 200
    posy  = 120
    textTaker.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))
    song = Text(textTaker, width = 50, height = 40)
    song.place(x = 0, y = 0)
    checkButton2 = Button(textTaker, text = 'Find out style!')
    checkButton2.bind('<Button-1>', lambda event: StyleFinder2(event, textTaker, song, resultLabel, resultLabel2, final))
    checkButton2.place(x = 150, y = 660)
    textTaker.mainloop()

def ResultWriter(resDir, resultLabel, resultLabel2, final):
    res = ['ResultPop/', 'ResultRap/']
    result = PrettyTable()
    result.field_names = ["Test", "Param", "Pop", "Rap"]
    #resDir = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/КлассикаТексты/'
    resPop = FileReader(resDir+res[0], 'Final.txt')
    resRap = FileReader(resDir+res[1], 'Final.txt')
    lines = [['','cos'], ['2gram','man'], ['','ev'],
             ['','cos'], ['3gram','man'], ['','ev'],
             ['','cos'], ['4gram','man'], ['','ev'],
             ['','cos'], ['5gram','man'], ['','ev'],
             ['','cos'], ['arf','man'], ['','ev'],
             ['','cos'], ['skipgrams','man'], ['','ev'],]
    resPop = [line.split()[-1] for line in resPop if len(line.split()) > 1]
    resRap = [line.split()[-1] for line in resRap if len(line.split()) > 1]
    for i in range(len(lines)):
        lines[i]. append(round(float(resPop[i]), 3))
        lines[i]. append(round(float(resRap[i]),3))
        result.add_row(lines[i])
    resultLabel.insert('end', 'FreqDist parameters\n')
    resultLabel.insert('end', result)
    resPop2 = FileReader(resDir+res[0], 'KeyWordFinal.txt')
    resRap2 = FileReader(resDir+res[1], 'KeyWordFinal.txt')
    result = PrettyTable()
    result.field_names = ["N", "Pop", "Rap"]
    lines = [['   1'],['  10'],[' 100'],['1000']]
    resPop2 = [line.split()[-1] for line in resPop]
    resRap2 = [line.split()[-1] for line in resRap]
    for i in range(len(lines)):
        lines[i]. append(round(float(resPop2[i]), 3))
        lines[i]. append(round(float(resRap2[i]),3))
        result.add_row(lines[i])
    resultLabel2.insert('end', 'KeyWords\n')
    resultLabel2.insert('end', result)
    popMax = 0
    resPop += resPop2
    resRap += resRap2
    for i in range(len(resPop)):
        if resPop[i] < resRap[i]:
            popMax += 1
    res = [['Pop',popMax/len(resPop)*100], ['Rap', (len(resPop) - popMax)/len(resPop)*100]]
    res.sort(key = lambda x: x[1])
    final.insert('end', 'Style result:\n')
    for i in res[::-1]:
        final.insert('end','\t' + i[0] + '\t' + str(round(i[1]))+'%' + '\n')
    


    
'''--------------------------------------------------- О К Н О  "П Р И Л О Ж Е Н И Я" -------------------------------------------'''

root = Tk()
root.title("Base options")
sizex = 800
sizey = 700
posx  = 100
posy  = 100
root.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))
root['bg'] = 'black'
mainMenu = Menu()

text = Text(width = 100, height = 10, bg = 'black', fg = 'white', highlightthickness = 0, bd = 0)
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


root.config(menu = mainMenu)

information = Label(root, bg = 'black', fg = 'white', font = 'Times 14', height = 6, width = 14, anchor = NW)
try:
    information['text'] = FullInformation()
    information.place(x = 10, y = 10)
except:
    information['text'] = 'No base found'
    information.place(x = 10, y = 10)

newDirectory = StringVar()
classDirectory  = StringVar()

dirTaker = Label(root, text = 'Enter directory:', bg = 'black', fg = 'white', font = 'Times 14', height = 1, anchor = NW)
dirTaker.place(x = 150, y = 10)

dirEntry = Entry(root, textvariable = newDirectory, width = 54)
dirEntry.place(x = 300, y = 14)

checkButton = Button(root, text = 'Find out style!')
checkButton.place(x = 650, y = 13)

dirTaker2 = Label(root, text = 'Enter songs\' texts:', bg = 'black', fg = 'white', font = 'Times 14', height = 1, anchor = NW)
dirTaker2.place(x = 150, y = 50)

dirEntry2 = Button(root, width = 46, height = 1, bg = 'white', command = TextTaker)
dirEntry2.place(x = 300, y = 52)

checkButton2 = Button(root, text = 'Find out style!', command = TextTaker)
checkButton2.place(x = 650, y = 53)

dirTaker2 = Label(root, text = 'Enter allbase dir:', bg = 'black', fg = 'white', font = 'Times 14', height = 1, anchor = NW)
dirTaker2.place(x = 150, y = 90)

dirEntry2 = Entry(root, textvariable = classDirectory, width = 54)
dirEntry2.place(x = 300, y = 94)

checkButton3 = Button(root, text = 'Find out style!')
checkButton3.place(x = 650, y = 93)

resultLabel = Text(root, height = 23, width = 50, bg = 'black', fg = 'white', highlightthickness = 0, bd = 0)
resultLabel.place(x = 40, y = 155)

resultLabel2 = Text(root, height = 10, width = 40, bg = 'black', fg = 'white', highlightthickness = 0, bd = 0)
resultLabel2.place(x = 440, y = 155)

final = Text(root, height = 10, width = 20, bg = 'black', fg = 'white', bd = 0)
final.place(x = 440, y = 333)

checkButton.bind('<Button-1>', lambda event: StyleFinder(event, resultLabel,resultLabel2, final))
checkButton3.bind('<Button-1>', lambda event: StyleFinder3(event, resultLabel,resultLabel2, final))

root.mainloop()

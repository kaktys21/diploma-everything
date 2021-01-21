import os
import nltk
import re
import pymorphy2
from nltk import ngrams, skipgrams
from scipy.spatial import distance
import pandas as pd

Dir = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/РэпТексты/Output/'

def InFiles(start):
    return list(filter(lambda x: x.startswith(start), os.listdir(Dir)))

def FileReader(directory, name):
    with open(directory+name, 'r', encoding = "utf-8") as file:
        return [line for line in file]

def FileWriter(directory, filename, file):
    with open(directory+filename, 'w', encoding = 'utf-8') as outFile:
        for line in file:
            outFile.write(line+'\n')

def Cleaner(directory, name, base):
    result = [re.sub(r'[!?]', ' ', song) for song in [re.sub(r'пр\.', '', song) for song in [song.lower() for song in [re.sub(r'[,\'\"\-«:»()—]', '', song) for song in
             [' '.join(song.split()) for song in [re.sub(r'\n', ' ', song) for song in
                                                  [re.sub(r'\u2005', ' ', song) for song in base]]]] if len(re.findall(r'[a-zA-Z]+', song)) == 0]]]
    FileWriter(Dir, 'clean'+name, result)

def SentenceSplitter(text):
    split_regex = re.compile(r'[.|!|?|…]')
    return list(filter(lambda t: t, [t.strip() for t in split_regex.split(text)]))

def Clean():
    files = InFiles('out')
    for file in files:
        base = FileReader(Dir, file)
        base = ['\n'.join(''.join(base).split('\n\n\n')[i].split('\n')[2:]) for i in range(len(''.join(base).split('\n\n\n')))]
        Cleaner(Dir, file, set(base))
    
def Lemmatisation():
    files = InFiles('clean')
    morph = pymorphy2.MorphAnalyzer()
    for file in files:
        lines = ' '.join(FileReader(Dir, file)).split()
        print(lines[:100])
        base = [morph.parse(word)[0].normal_form for word in lines if len(word) > 0 and word not in ['пр','припев','куплет']]
        print('Done')
        FileWriter(Dir, 'lemma'+file, base)

def int_r(num):
    return int(num + (0.5 if num > 0 else -0.5))

def AverageReducedFrequency(word, n, corpra, L):
    f = len(n)
    d = [n[p + 1] - n[p] for p in range(len(n) - 1)]
    d.append(n[0] + len(corpra) - n[len(n) - 1])
    segmentLen = int_r(L/f)
    summa = 0
    for i in range(len(d)):
        summa += min(d[i],segmentLen)
    return (word, round(summa/segmentLen, 3))

# Делает частотные списки по директории
def ARFCounter():
    files = InFiles('lemma')
    for file in files:
        corpra = [re.sub(r'\n', '', word) for word in FileReader(Dir, file)]
        words = {word: [] for word in list(nltk.FreqDist(corpra).keys())}
        for i in range(len(corpra)):
            words[corpra[i]].append(i + 1)
        L = len(corpra)
        result = [AverageReducedFrequency(word, words[word], corpra, L) for word in words]
        result.sort(key = lambda x: x[1])
        result = [str(pair[0])+'\t'+str(pair[1]) for pair in result[::-1]]
        FileWriter(Dir, 'arf'+file, result)

# Создаёт скипграммы по заданной директории
def SkipGramms():
    files = InFiles('out')
    for file in files:
        base = FileReader(Dir, file)
        base = set(['\n'.join(''.join(base).split('\n\n\n')[i].split('\n')[2:]) for i in range(len(''.join(base).split('\n\n\n')))])
        corpra = SentenceSplitter(' '.join([re.sub(r'пр\.', '', song) for song in [song.lower() for song in [re.sub(r'[,\'\"\-«:»()—\–]', '', song) for song in
             [' '.join(song.split()) for song in [re.sub(r'\n', ' ', song) for song in
                                                  [re.sub(r'\u2005', ' ', song) for song in set(base)]]]] if len(re.findall(r'[a-zA-Z]+', song)) == 0]]))
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
        FileWriter(Dir, 'skipgram'+file, result)

def Ngrams():
    files = InFiles('out')
    for file in files:
        base = FileReader(Dir, file)
        base = set(['\n'.join(''.join(base).split('\n\n\n')[i].split('\n')[2:]) for i in range(len(''.join(base).split('\n\n\n')))])
        lines = [line for line in [re.sub(r'\u2005', ' ', song) for song in base] if len(re.findall(r'[a-zA-Z]+', line)) == 0]
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
            FileWriter(Dir, 'ngrams'+str(p)+file, result)

def KeyWords():
    files = InFiles('arf')
    for i in range(len(files)):
        for j in range(len(files)):
            if files[i][16:] != files[j][16:]:
                print(files[i][16:], files[j][16:])
                fwords = []
                words = FileReader(Dir, files[i])
                if len(words) > 5000:
                    fwords.append(words[:5001])
                else:
                    fwords.append(words)
                words = FileReader(Dir, files[j])
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
                    summ = [max(popkeywords[N][1], rapkeywords[N][1]) for N in range(len(popkeywords))]
                    dist = sum(summ)/len(summ)
                    with open(Dir+'keydist'+files[i][16:-4]+'vs'+files[j][16:], 'a', encoding = 'utf-8') as f:
                        f.write("При n равном "+str(n)+":\t"+str(dist)+'\n')
                    popkeywords.sort(key = lambda x: x[1])
                    popkeywords = popkeywords[::-1]
                    result = [word[0]+'\t'+str(word[1]) for word in popkeywords]
                    FileWriter(Dir, str(n)+'keyword'+files[i][16:-4]+'vs'+files[j][16:], result)
                    rapkeywords.sort(key = lambda x: x[1])
                    rapkeywords = rapkeywords[::-1]
                    result = [word[0]+'\t'+str(word[1]) for word in rapkeywords]
                    FileWriter(Dir, str(n)+'keyword'+files[i][16:-4]+'vs'+files[j][16:], result)

def VectorCreator(typ):
    allsong = []
    files = InFiles(typ)
    art = list(filter(lambda x: x.startswith('out'), os.listdir(Dir)))
    for file in files:
        l = FileReader(Dir, file)
        allsong += [line.split('\t')[0] for line in l]
    res = pd.DataFrame(columns = [name[3:-4] for name in art], index = set(allsong))
    for i in range(len(files)):
        t = {line.split('\t')[0]:float(line.split('\t')[1]) for line in FileReader(Dir, files[i])}
        for word in res.index:
            if word not in t:
                res.loc[word][art[i][3:-4]] = 0
            else:
                res.loc[word][art[i][3:-4]] = t[word]
    res.to_excel(typ+'.xlsx')
    return [list(res[m]) for m in res]

def Cosin(vectors, s):
    print('cos',s)
    files = list(filter(lambda x: x.startswith('out'), os.listdir(Dir)))
    tab = pd.DataFrame({x[3:-4]:'' for x in files},
                             index = [y[3:-4] for y in files])
    for i in range(len(vectors) - 1):
        for j in range(i + 1, len(vectors)):
            tab[files[i][3:-4]][files[j][3:-4]] = distance.cosine(vectors[i], vectors[j])
    print(tab)
    tab.to_excel(s+'cos.xlsx')

def Ev(vectors, s):
    print('ev',s)
    files = list(filter(lambda x: x.startswith('out'), os.listdir(Dir)))
    tab = pd.DataFrame({x[3:-4]:'' for x in files},
                             index = [y[3:-4] for y in files])
    for i in range(len(vectors) - 1):
        for j in range(i + 1, len(vectors)):
            tab[files[i][3:-4]][files[j][3:-4]] = distance.euclidean(vectors[i], vectors[j])
    print(tab)
    tab.to_excel(s+'euclidean.xlsx')

def Cityblock(vectors, s):
    print('ev',s)
    files = list(filter(lambda x: x.startswith('out'), os.listdir(Dir)))
    tab = pd.DataFrame({x[3:-4]:'' for x in files},
                             index = [y[3:-4] for y in files])
    for i in range(len(vectors) - 1):
        for j in range(i + 1, len(vectors)):
            tab[files[i][3:-4]][files[j][3:-4]] = distance.cityblock(vectors[i], vectors[j])
    print(tab)
    tab.to_excel(s+'cityblock.xlsx')
                    
def Dist():
    s = ['arf', 'ngrams2', 'ngrams3','ngrams4','ngrams5','skipgram']
    for typ in s:
        vector = VectorCreator(typ)
        Cosin(vector, typ)
        Ev(vector, typ)
        Cityblock(vector, typ)
        
        
def KeyTables():
    files = InFiles('keydist')
    tableFiles = [file[7:-4].split('vs') for file in files]
    tables = [['n01key'],['n10key'],['n100key'],['n1000key']]
    for tab in tables:
        tab.append(pd.DataFrame({x[3:-4]:'' for x in list(filter(lambda x: x.startswith('out'), os.listdir(Dir)))},
                         index = [y[3:-4] for y in list(filter(lambda x: x.startswith('out'), os.listdir(Dir)))]))
    for i in range(len(files)):
        dist = [round(float(line.split('\t')[1]),3) for line in FileReader(Dir, files[i])]
        for j in range(len(dist)):
            tables[j][1][tableFiles[i][0]][tableFiles[i][1]] = dist[j]
    for i in range(len(tables)):
        tables[i][1].to_excel(tables[i][0]+'.xlsx')

def ARFOsob(corpra, direct, file):
    words = {word: [] for word in list(nltk.FreqDist(corpra).keys())}
    for i in range(len(corpra)):
        words[corpra[i]].append(i + 1)
    L = len(corpra)
    result = [AverageReducedFrequency(word, words[word], corpra, L) for word in words]
    result.sort(key = lambda x: x[1])
    result = [str(pair[0])+'\t'+str(pair[1]) for pair in result[::-1]]
    FileWriter(direct, 'arf'+file, result)
    
def Osobennie():
    art = ['noizemc','kasta','krovostok']
    for artist in art:
        direct = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/РэпТексты/'+artist+'/'
        files = list(filter(lambda x:x!=artist+'.txt', os.listdir(direct)))
        others = [' '.join(FileReader(direct, file)) for file in files]
        others = ' '.join(others).split()
        morph = pymorphy2.MorphAnalyzer()
        arttext = re.sub(r'\n', '', ' '.join(FileReader(direct,artist+'.txt'))).split()
        lemmaArtist = [morph.parse(word)[0].normal_form for word in arttext if len(word) > 0 and word not in ['пр','припев','куплет']]
        print(artist, '\tmorph\Done')
        lemmaOthers = [morph.parse(word)[0].normal_form for word in others if len(word) > 0 and word not in ['пр','припев','куплет']]
        print(artist, '\tAllmorph\Done')
        stopword = ['и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а', 'по', 'это', 'она', 'этот', 'к', 'но', 'они', 'мы', 'как', 'из', 'у', 'который', 'то', 'за', 'свой', 'что', 'весь', 'год', 'от', 'так', 'о', 'для', 'ты', 'же', 'все', 'тот', 'мочь', 'вы', 'человек', 'такой', 'его', 'сказать', 'только', 'или', 'ещё', 'бы', 'себя', 'один', 'как', 'уже', 'до', 'время', 'если', 'сам', 'когда', 'другой', 'вот', 'говорить', 'наш', 'мой', 'знать', 'стать', 'при', 'чтобы', 'дело', 'жизнь', 'кто', 'первый', 'очень', 'два', 'день', 'её', 'новый', 'рука', 'даже', 'во', 'со', 'раз', 'где', 'там', 'под', 'можно', 'ну', 'какой', 'после', 'их', 'работа', 'без', 'самый', 'потом', 'надо', 'хотеть', 'ли', 'слово', 'идти', 'большой', 'должен', 'место', 'иметь', 'ничто']
        with open(direct+'lemmas'+artist+'.txt', 'w', encoding = 'utf-8') as outFile:
            for line in lemmaArtist:
                if line not in stopword:
                    outFile.write(line+' ')
        with open(direct+'lemmas'+'all.txt', 'w', encoding = 'utf-8') as outFile:
            for line in lemmaOthers:
                if line not in stopword:
                    outFile.write(line+' ')
        print(artist, '\tDone')
        #ARFOsob(lemmaArtist, direct, artist+'.txt')
        #ARFOsob(lemmaOthers, direct, 'all.txt')

KeyWords()
KeyTables()

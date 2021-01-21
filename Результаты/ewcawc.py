import os
import nltk
import re
import pymorphy2
from nltk import ngrams, skipgrams
from scipy.spatial import distance
import pandas as pd
import math
from operator import itemgetter

Dir = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/one/'

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
    FileWriter(Dir, 'clean.txt', result)

def SentenceSplitter(text):
    split_regex = re.compile(r'[.|!|?|…]')
    return list(filter(lambda t: t, [t.strip() for t in split_regex.split(text)]))

def Clean():
    files = InFiles('text')
    for file in files:
        base = FileReader(Dir, file)
        base = ['\n'.join(''.join(base).split('\n\n\n')[i].split('\n')[2:]) for i in range(len(''.join(base).split('\n\n\n')))]
        Cleaner(Dir, file, set(base))

def Lemmatisation():
    files = InFiles('clean')
    morph = pymorphy2.MorphAnalyzer()
    for file in files:
        lines = ' '.join(FileReader(Dir, file)).split()
        #print(lines[:100])
        base = [morph.parse(word)[0].normal_form for word in lines if len(word) > 0 and word not in ['пр','припев','куплет']]
        #print('Done')
        FileWriter(Dir, 'lemma.txt', base)
        
def SkipGramms():
    files = InFiles('text')
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
        FileWriter(Dir, 'skip.txt', result)

def Ngrams():
    files = InFiles('text')
    for file in files:
        base = FileReader(Dir, file)
        base = set(['\n'.join(''.join(base).split('\n\n\n')[i].split('\n')[2:]) for i in range(len(''.join(base).split('\n\n\n')))])
        lines = [line for line in [re.sub(r'\u2005', ' ', song) for song in base] if len(re.findall(r'[a-zA-Z]+', line)) == 0]
        ngramList = []
        for song in lines:
            for ng in list(nltk.ngrams(song,4)):
                ngramList.append(ng)
        l = len(ngramList)
        freq = nltk.FreqDist(ngramList)
        freqDist = [(n,freq[n]/ l * 1000000) for n in list(freq.keys())]
        freqDist.sort(key = lambda x: x[1])
        freqDist = freqDist[::-1]
        result = [str(freqDist[i][0])+'\t'+str(freqDist[i][1]) for i in range(len(freqDist))]
        FileWriter(Dir, 'ngram4.txt', result)
            
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
        FileWriter(Dir, 'arf.txt', result)

def KeyCounter(one, other, n):
    for word in list(other.keys()):
        if word not in one:
            one[word] = 0
    for word in list(one.keys()):
        if word not in other:
            other[word] = 0
    res = [max((one[word] + n) / (other[word] + n), (other[word] + n) / (one[word] + n)) for word in list(one.keys())]
    dist = sum(res)/len(res)
    return dist

def KeyWords():
    directory = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/'
    one = [(word.split('\t')[0],float(word.split('\t')[1])) for word in FileReader('C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/one/', 'arf.txt')]
    if len(one) > 5000:
        one = one[:5000]
    one = {one[i][0]:one[i][1] for i in range(len(one))}
    files = os.listdir('C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/arf/')
    others = []
    for file in files:
        s = [(word.split('\t')[0],float(word.split('\t')[1])) for word in FileReader(directory+'arf/',file)]
        if len(s) > 5000:
            s = s[:5000]
        others.append({s[i][0]:s[i][1] for i in range(len(s))})
    res = []
    for n in [1,1000]:
        for other in others:
            res.append(str(KeyCounter(one, other, n)))
    return res

def TableReader():
    #sheetNames = ['key1', 'key1000', 'ngram4cos', 'skipcos']
    sheetNames = ['ngram4cos', 'skipcos', 'arfcos']
    return  {name:pd.read_excel(r'C:/Users/epish/Desktop/учеба/ДИПЛОМ/сводная.xlsb.xlsx', sheet_name = name) for name in sheetNames}

def VectorCreator(typ, file, fileone):
    table = pd.read_excel(r'C:/Users/epish/Desktop/учеба/ДИПЛОМ/'+file)
    with open(r'C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/one/'+fileone, 'r', encoding = 'utf-8') as f:
            one = {line.split('\t')[0]:float(line.split('\t')[1]) for line in f}
    res = []
    for ngram in list(table['Unnamed: 0']):
            if ngram in one:
                    res.append(one[ngram])
                    one.pop(ngram)
            else:
                    res.append(0)
    table['one'] = res
    print(len(one))
    for i in range(len(list(one))):
        #print(str(i) + ' of ' + str(len(list(one))))
        table.loc[len(table), 'one'] = one[list(one)[i]]
        table['Unnamed: 0'][len(table) - 1] = list(one)[i]
    #print(table)
    art = list(table)[1:-1]
    vectors = [list(table[art]) for art in list(table)[1:]]
    return vectors

def Cos(vectors):
    vectorOne = vectors[len(vectors) - 1]
    t = [distance.cosine(vectorOne, vectors[i]) for i in range(len(vectors))]
    return t

def FirstType(dist):
    #param = ['key1', 'key1000', 'ngram4cos', 'skipcos']
    #dist = {param[i]:dist[i] for i in range(4)}
    param = ['ngram4cos', 'skipcos', 'arfcos']
    dist = {param[i]:dist[i] for i in range(3)}
    tables = TableReader()
    res = []
    minIndex = ''
    for p in param:
        minim = 10000000000000000000000000000000000000000000000000000000000000000
        for d in range(len(tables[p][:20])):
            delta = []
            for i in range(len(list(tables[p].iloc[d])[1:])):
                if math.isnan(list(tables[p].iloc[d])[1:][i]):
                    delta.append(abs(float(dist[p][i]) - 0))
                else:
                    delta.append(abs(float(dist[p][i]) - list(tables[p].iloc[d])[1:][i]))
            deltaSum = sum(delta)
            #print(deltaSum)
            if deltaSum < minim:
                minim = deltaSum
                minIndex = list(tables[p].iloc[d])[0]
        res.append((p, minIndex, minim))
    dist = dict.fromkeys(list(tables['arfcos']['Unnamed: 0'][:20]), 0)
    #print(res)
    for param in res:
        for i in range(20):
            if math.isnan(tables[param[0]].iloc[i][param[1]]):
                dist[tables['arfcos']['Unnamed: 0'][i]] += 0
            else:
                dist[tables['arfcos']['Unnamed: 0'][i]] += tables[param[0]].iloc[i][param[1]]
    p = [(art,dist[art]) for art in list(dist)]
    p.sort(key = lambda x: x[1])
    return p[0][0]
    
def SecondType(dist):
    res = [[]]
    for i in range(20):
        #for j in range(4):
        for j in range(3):
            res[i].append(float(dist[j][i]))
        res.append([])
    vectors = []
    for i in range(len(res) - 1):
        #mass = int_r(max(res[i]))
        mass = 1
        if math.isnan(res[i][0]):
            x_1 = 0
        else:
            x_1 = res[i][0]
        if math.isnan(res[i][1]):
            x_2 = 0
        else:
            x_2 = res[i][1]
        if math.isnan(res[i][2]):
            x_3 = 0
        else:
            x_3 = res[i][2]
        if math.isnan(res[i][2]):
            x_4 = 0
        else:
            x_4 = res[i][2]
        sqrt = (x_1/mass)**2 + (x_2/mass)**2 + x_3**2 #+ res[i][3]**2
        vectors.append(sqrt)
    auth = ['2rbina2rista', 'basta', 'brb', 'eldzhey', 'face', 'feduk', 'gnoiny', 'guf', 'husky', 'kasta', 'korzh', 'krovostok', 'kunteynir', 'lizer', 'lsp', 'morgenshtern', 'noizemc', 'skriptonite', 'timati', 'xleb']
    auth = {auth[i]:vectors[i] for i in range(len(vectors))}
    return list(auth.keys())[list(auth.values()).index(min(list(auth.values())))]

def FilePrepare():
    if len(os.listdir(Dir)) != 6:
        Clean()
        Lemmatisation()
        SkipGramms()
        Ngrams()
        ARFCounter()

def FinalDistComp():
    FilePrepare()
    files = {'ngram4':['ngrams4.xlsx', 'ngram4.txt'],'skip':['skipgram.xlsx', 'skip.txt'], 'arf':['arf.xlsx', 'arf.txt']}
    typs = ['ngram4', 'skip', 'arf']
    #res = KeyWords()
    dist = []
    #dist.append(res[:20]) #20 расстояний до каждого N = 1
    #dist.append(res[20:]) #20 расстояний до каждого N = 1000
    for typ in typs:
        print(typ)
        dist.append(Cos(VectorCreator(typ, files[typ][0], files[typ][1]))) # 2 по 20 расстояний из нграмм и скипграмм
    f_1 = FirstType(dist)
    f_2 = SecondType(dist)
    return f_1, f_2

def Testing():
    files = os.listdir('C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/texts/')
    steps = [2, 4, 8, 16]
    FirstTy = pd.DataFrame(columns = steps, index = [file[:-4] for file in files])
    SecondTy = pd.DataFrame(columns = steps, index = [file[:-4] for file in files])
    for file in files:
        text = FileReader('C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/texts/', file)
        text = ''.join(text).split('\n\n\n')
        for step in steps:
            print(file+'\t'+str(step))
            for f in os.listdir('C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/one/'):
                if f != 'text.txt':
                    os.remove('C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/one/'+f)
            with open('C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/one/text.txt', 'w', encoding = 'utf-8') as f:
                for i in range(0,len(text), step):
                    for line in text[i]:
                        f.write(line)
                    f.write('\n\n\n')
            try:
                f_1, f_2 = FinalDistComp()
                FirstTy.loc[file[:-4]][step] = f_1
                SecondTy.loc[file[:-4]][step] = f_2
                print(FirstTy, SecondTy, sep = '\n')
            except KeyError:
                FirstTy.loc[file[:-4]][step] = 'Faled'
                SecondTy.loc[file[:-4]][step] = 'Faled'
                print(FirstTy, SecondTy, sep = '\n')

Testing()
    

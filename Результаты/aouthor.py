import pandas as pd
import os
import re
import nltk
import pymorphy2
from scipy.spatial import distance
import math
from operator import itemgetter

def FileReader(directory, filename):
    with open(directory+filename, 'r', encoding = 'utf-8') as inFile:
        return [line for line in inFile]

def int_r(num):
    return int(num + (0.5 if num > 0 else -0.5))

def SentenceSplitter(text):
    split_regex = re.compile(r'[.|!|?|…]')
    return list(filter(lambda t: t, [t.strip() for t in split_regex.split(text)]))

			
def NSkipGramms(inDir, outDir):
    files = ['lemma.txt']
    for file in files:
            with open(inDir+file, 'r', encoding='utf-8') as inFile:
                    base = [line for line in inFile]
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
            with open(outDir+'skip.txt', 'w', encoding='utf-8') as outFile:
                    for line in result:
                            outFile.write(line+'\n')

def NLemmas(text, d):
    morph = pymorphy2.MorphAnalyzer()
    lines = [morph.parse(word)[0].normal_form for word in re.sub(r'\n', '', re.sub(r'[\.,?! \- —«»]', ' ', text)).lower().split()]
    with open(d+'lemma.txt', 'w', encoding = 'utf-8') as f:
	    for word in lines:
		    f.write(word+' ')

def NNgrams(d):
    with open(d+'text.txt', 'r', encoding = 'utf-8') as f:
	    lines = [line for line in f if len(re.findall(r'[a-zA-Z]', line)) == 0]
    ngramList = list(nltk.ngrams(' '.join(lines),4))
    l = len(ngramList)
    freq = nltk.FreqDist(ngramList)
    freqDist = [(n,freq[n]/ l * 1000000) for n in list(freq.keys())]
    freqDist.sort(key = lambda x: x[1])
    freqDist = freqDist[::-1]
    result = [str(freqDist[i][0])+'\t'+str(freqDist[i][1]) for i in range(len(freqDist))]
    with open(d+'ngram4.txt', 'w', encoding = 'utf-8') as f:
            for i in result:
                    f.write(i+'\n')
                    
def AverageReducedFrequency(word, n, corpra, L):
    f = len(n)
    d = [n[p + 1] - n[p] for p in range(len(n) - 1)]
    d.append(n[0] + len(corpra) - n[len(n) - 1])
    segmentLen = int_r(L/f)
    summa = 0
    for i in range(len(d)):
        summa += min(d[i],segmentLen)
    return (word, round(summa/segmentLen, 3))

def NARFCounter(text, d):
    corpra = ' '.join(text).split()
    words = {word: [] for word in list(nltk.FreqDist(corpra).keys())}
    for i in range(len(corpra)):
        words[corpra[i]].append(i + 1)
    L = len(corpra)
    result = [AverageReducedFrequency(word, words[word], corpra, L) for word in words]
    result.sort(key = lambda x: x[1])
    result = [str(pair[0])+'\t'+str(pair[1]) for pair in result[::-1]]
    with open(d+'arf.txt', 'w', encoding = 'utf-8') as f:
            for i in result:
                    f.write(i+'\n')
    
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
       
def NewFile():
    d = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/one/'
    if os.listdir(d) == ['text.txt']:
        direct = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/auth/'
        with open(d+'text.txt', 'r', encoding = 'utf-8') as f:
                lines = [line for line in f if len(re.findall(r'[a-zA-Z]', line)) == 0]
        NLemmas(' '.join(lines),d)
        NSkipGramms(d,d)
        NNgrams(d)
        NARFCounter(FileReader(d,'lemma.txt'), d)

def TableReader():
    sheetNames = ['key1', 'key1000', 'ngram4cos', 'skipcos']
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
        print(str(i) + ' of ' + str(len(list(one))))
        table.loc[len(table), 'one'] = one[list(one)[i]]
        table['Unnamed: 0'][len(table) - 1] = list(one)[i]
    print(table)
    art = list(table)[1:-1]
    vectors = [list(table[art]) for art in list(table)[1:]]
    return vectors

def Cos(vectors):
    vectorOne = vectors[len(vectors) - 1]
    t = [distance.cosine(vectorOne, vectors[i]) for i in range(len(vectors))]
    return t

def FirstType(dist):
    param = ['key1', 'key1000', 'ngram4cos', 'skipcos']
    dist = {param[i]:dist[i] for i in range(4)}
    tables = TableReader()
    res = []
    minIndex = ''
    for p in param:
        minim = 10000000000
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
    dist = dict.fromkeys(list(tables['key1']['Unnamed: 0'][:20]), 0)
    print(res)
    for param in res:
        for i in range(20):
            if math.isnan(tables[param[0]].iloc[i][param[1]]):
                dist[tables['key1']['Unnamed: 0'][i]] += 0
            else:
                dist[tables['key1']['Unnamed: 0'][i]] += tables[param[0]].iloc[i][param[1]]
    p = [(art,dist[art]) for art in list(dist)]
    p.sort(key = lambda x: x[1])
    print("First param " + p[0][0])
    
def SecondType(dist):
    res = [[]]
    for i in range(20):
        for j in range(4):
            res[i].append(float(dist[j][i]))
        res.append([])
    vectors = []
    for i in range(len(res) - 1):
        mass = int_r(max(res[i]))
        sqrt = (res[i][0]/mass)**2 + (res[i][1]/mass)**2 + res[i][2]**2 + res[i][3]**2
        vectors.append(sqrt)
    auth = ['2rbina2rista', 'basta', 'brb', 'eldzhey', 'face', 'feduk', 'gnoiny', 'guf', 'husky', 'kasta', 'korzh', 'krovostok', 'kunteynir', 'lizer', 'lsp', 'morgenshtern', 'noizemc', 'skriptonite', 'timati', 'xleb']
    print("Second param " + auth[min(enumerate(vectors), key=itemgetter(1))[0]])
   
def FinalDistComp():
    NewFile()
    files = {'ngram4':['ngrams4.xlsx', 'ngram4.txt'],'skip':['skipgram.xlsx', 'skip.txt']}
    typs = ['ngram4', 'skip']
    res = KeyWords()
    dist = []
    dist.append(res[:20]) #20 расстояний до каждого N = 1
    dist.append(res[20:]) #20 расстояний до каждого N = 1000
    vectors = []
    for typ in typs:
        print(typ)
        dist.append(Cos(VectorCreator(typ, files[typ][0], files[typ][1]))) # 2 по 20 расстояний из нграмм и скипграмм
    FirstType(dist)
    SecondType(dist)
    
FinalDistComp()

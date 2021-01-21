import os
import pandas as pd
import nltk
from scipy.spatial import distance
import math
import re
import pymorphy2
import time
from operator import itemgetter

OutDir = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/'
Dir = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/TEST/Unprocessed/'

CleanDirk = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Known/Clean/'
LemmaDirk = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Known/Lemmas/'
NgramDirk = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Known/Ngrams/'
ARFDirk = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Known/ARF/'
SkipDirk = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Known/Skip/'

CleanDiru = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Unknown/Clean/'
LemmaDiru = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Unknown/Lemmas/'
NgramDiru = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Unknown/Ngrams/'
ARFDiru = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Unknown/ARF/'
SkipDiru = 'C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/Unknown/Skip/'


def SentenceSplitter(text):
    split_regex = re.compile(r'[.|!|?|…]')
    return list(filter(lambda t: t, [t.strip() for t in split_regex.split(text)]))

def FilesPrepare():
    pages = os.listdir(Dir)
    for page in pages:
        files = os.listdir(Dir+page)
        for i in range(len(files)):
            with open(Dir+page+'/'+files[i], 'r', encoding = 'utf-8') as f:
                lines = [line for line in f]
            lines = ''.join(''.join(lines).split('\n\n')[1:]).split('\n')
            if i%8 == 0:
                with open(OutDir+'Known/'+page+'.txt', 'a', encoding = 'utf-8') as f:
                    for line in lines:
                        f.write(line+'\n')
                    f.write('\n')
            if i%8 == 1:   
                with open(OutDir+'Unknown/'+page+'.txt', 'a', encoding = 'utf-8') as f:
                    for line in lines:
                        f.write(line+'\n')
                    f.write('\n')

def FileReader(directory, name):
    with open(directory+name, 'r', encoding = "utf-8") as file:
        return [line for line in file if line != '\n']
    
def Clean(word):
    files = list(filter(lambda x: x.endswith('txt'), os.listdir(OutDir+word)))
    for file in files:
        base = FileReader(OutDir+word, file)
        base = re.sub(r'\n', ' ', re.sub(r'[,\'\"\-«:»()—…\/?!\.―–0-9;:*]', ' ', ' '.join(base))).lower()
        with open('C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/'+ word + 'Clean/'+file, 'w', encoding = 'utf-8') as f:
            f.write(base)

def SkipGramms(word):
    files = list(filter(lambda x: x.endswith('txt'), os.listdir(OutDir+word)))
    for file in files:
        corpra = SentenceSplitter(re.sub(r'[,\'\"\-«:»()—\–―–0-9;:*]',' ',' '.join(FileReader(OutDir+word, file)).lower()))
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
        with open('C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/' + word + 'Skip/'+file, 'w', encoding = 'utf-8') as f:
            for skip in result:
                f.write(skip+'\n')                         
       
def Lemmatisation(word):
    files = os.listdir(CleanDir)
    morph = pymorphy2.MorphAnalyzer()
    for file in files:
        print(file)
        lines = ''.join(FileReader(CleanDir, file)).split(' ')
        base = [morph.parse(word)[0].normal_form for word in lines if len(word) > 0]
        with open('C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/' + word + 'Lemmas/'+file, 'w', encoding = 'utf-8') as f:
            for lemm in base:
                f.write(lemm+' ')

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
def ARFCounter(word):
    files = os.listdir(LemmaDir)
    for file in files:
        corpra = [arf for arf in ''.join(FileReader(LemmaDir, file)).split() if len(word) > 0]
        words = {arf: [] for arf in list(nltk.FreqDist(corpra).keys())}
        for i in range(len(corpra)):
            words[corpra[i]].append(i + 1)
        L = len(corpra)
        result = [AverageReducedFrequency(arf, words[arf], corpra, L) for arf in words]
        result.sort(key = lambda x: x[1])
        result = [str(pair[0])+'\t'+str(pair[1]) for pair in result[::-1]]
        with open('C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/' + word + 'ARF/'+file, 'w', encoding = 'utf-8') as f:
            for arf in result:
                f.write(arf+'\n')

def Ngrams(word):
    files = list(filter(lambda x: x.endswith('txt'), os.listdir(OutDir+word)))
    for file in files:
        lines = FileReader(OutDir+word, file)
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
        with open('C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/' + word + 'Ngrams/'+file, 'w', encoding = 'utf-8') as f:
            for n in result:
                f.write(n+'\n')
def Known():
    word = 'Known/'
    Clean(word)
    SkipGramms(word)
    Lemmatisation(word)
    ARFCounter(word)
    Ngrams(word)

def Unknown():
    word = 'Unknown/'
    Clean(word)
    SkipGramms(word)
    Lemmatisation(word)
    ARFCounter(word)
    Ngrams(word)

#FilesPrepare()
#Known()
#Unknown()

def VectorCreator(knownfile, unknownfile, knowndir, unknowndir):
    start_time_def = time.time()
    p = [knownfile, 'un'+unknownfile]
    with open(knowndir+knownfile, 'r', encoding = 'utf-8') as f:
        lines = {line.split('\t')[0]:float(line.split('\t')[1]) for line in f}
    with open(unknowndir+unknownfile, 'r', encoding = 'utf-8') as f:
        unlines = {line.split('\t')[0]:float(line.split('\t')[1]) for line in f}
    known = dict.fromkeys(list(lines)+list(unlines))
    unknown = {}
    for word in known:
        if word in list(lines):
            known[word] = lines[word]
        else:
            known[word] = 0
        if word in list(unlines):
            unknown[word] = unlines[word]
        else:
            unknown[word] = 0
    vectors = [list(known.values()), list(unknown.values())]
    print (distance.cosine(vectors[0], vectors[1]), time.time() - start_time_def, sep = '\t')
    return distance.cosine(vectors[0], vectors[1])
        
def SecondType(dist, art):
    vectors = []
    for i in range(len(dist) - 1):
            sqrt = dist[i][0]**2 + dist[i][1]**2 + dist[i][2]**2
            vectors.append(sqrt)
    return art[min(enumerate(vectors), key=itemgetter(1))[0]][:-4]
    
def FTables():
    knownp = [NgramDirk , ARFDirk , SkipDirk]
    unknownp = [NgramDiru , ARFDiru , SkipDiru]
    art = os.listdir(knownp[0])
    restable = pd.DataFrame(columns = ['Result', 'Time'], index = [artist[6:-4] for artist in art])
    print(restable)
    start_time = time.time()
    for unknown in range(len(art)):
        res = [[]]
        for known in range(len(art)):
            print(art[unknown][:-4], ' vs ', art[known][:-4])
            for i in range(len(knownp)):
                res[known].append(VectorCreator(art[known], art[unknown], knownp[i], unknownp[i]))
            res.append([])
        name = SecondType(res, art)
        restable.loc[art[unknown][6:-4], 'Result'] = name
        restable.loc[art[unknown][6:-4], 'Time'] = time.time() - start_time
        print(restable)
        restable.to_excel('C:/Users/epish/Desktop/учеба/ДИПЛОМ/authTest/text.xlsx')

FTables()

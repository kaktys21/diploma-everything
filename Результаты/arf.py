import re
from threading import Thread
import pymorphy2
import nltk

def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num

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
    
    
def BasePrepare(filename):
    inFile = open(r'C:/Users/epish/Desktop/учеба/ДИПЛОМ/'+filename, 'r', encoding = 'utf-8')
    lines = [line for line in inFile]
    inFile.close()
    morph = pymorphy2.MorphAnalyzer()
    lines =  re.sub(r'[\.!?]+', '' , re.sub(r'пр\.', '', re.sub(r'\n', '', ' '.join(lines)))).split(' ')
    corpra = [morph.parse(word)[0].normal_form for word in lines if len(word) > 0 and word not in ['р','припев','куплет']]
    return corpra

def OutFileWriter(filename, corpra):
    outFile = open(r'C:/Users/epish/Desktop/учеба/ДИПЛОМ/out'+filename, 'w', encoding = 'utf-8')
    words = {word: [] for word in list(nltk.FreqDist(corpra).keys())}
    for i in range(len(corpra)):
        words[corpra[i]].append(i + 1)
    L = len(corpra)
    result = [AverageReducedFrequency(word, words[word], corpra, L) for word in words]
    result.sort(key = lambda x: x[1])
    for pair in result[::-1]:
        outFile.write(str(pair[0])+'\t'+str(pair[1])+'\n')

    
def Main():
    files = ['popcleanbase.txt', 'rapcleanbase.txt']
    thread1 = Thread(target = OutFileWriter, args = (files[0], BasePrepare(files[0])))
    thread2 = Thread(target = OutFileWriter, args = (files[1], BasePrepare(files[1])))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

Main()

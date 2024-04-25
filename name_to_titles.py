import requests 
import json
import re
import crossref
from langdetect import detect
#from crossref.restful import Works

def works_of(name):
    url = 'https://api.crossref.org/works?query.author=' + name + '&cursor=*'
    print(url)
    r = requests.get(url)
    return r


def GetTitlesandIssn(name, Small):
    titles = []
    issn = []
    url = 'https://api.crossref.org/works?query.author=' + name + '&cursor=*'
    print(url)
     
    r = requests.get(url)
    data = json.loads(r.text)
    while data["message"]["items"]:
        for (index, _) in enumerate(data["message"]["items"]):
            #issn.append(data["message"]["items"][index]["ISSN"][issnindex])
            if('ISSN' in data["message"]["items"][index]):
                for(issnindex, _) in enumerate(data["message"]["items"][index]['ISSN']):
                    issn.append(re.sub('-', '', data["message"]["items"][index]['ISSN'][issnindex]))
            if('title' in data["message"]["items"][index]):
                for(titleindex, _) in enumerate(data["message"]["items"][index]["title"]):
                    title = (data["message"]["items"][index]["title"][titleindex])
                    try:
                        if(detect(title) == 'en'):
                            titles.append(title)
                    except:
                        pass
        cursor = data["message"]['next-cursor']
        print(titles)
        if(Small == True):break
        url ='https://api.crossref.org/works?query.author=' + name + '&cursor=' + cursor
        r = requests.get(url)
        data = json.loads(r.text)
    return (titles, issn)
        
class Journal:
    def __init__(self, ISSN, h_index, q, frequancy):
        self.ISSN = ISSN
        self.h_index = h_index
        self.q = q
        self.frequancy = frequancy
    def Print(jour):
        print("ISSN: " + jour.ISSN + "; H-Index: " + jour.h_index + "; Q:" + jour.q + "; Frequancy: " + str(jour.frequancy))
    def ToStr(jour):
        return ("ISSN: " + jour.ISSN + "; H-Index: " + jour.h_index + "; Q:" + jour.q + "; Frequancy: " + str(jour.frequancy))

def HIndex(issns):
    indexes = []
    # string to search in file
    with open(r'journals.csv', 'r') as fp:
        # read all lines using readline()
        lines = fp.readlines()
        for row in lines:
            line = row.split(';')
            current_issn1 = ""
            current_issn2 = ""
            coma_count = 0
            for r in line[4]:
                if(coma_count != 0 and r != '"' and r != " "):
                    current_issn2 += r
                elif(coma_count == 0 and r != '"' and r != " " and r != ","):
                    current_issn1 += r
                elif( r == ","):
                    coma_count += 1 
            amount1 = countX(issns, current_issn1)
            amount2 = countX(issns, current_issn2)
            if(amount1 > 0):
                journal = Journal(current_issn1, line[7], line[6], amount1)
                indexes.append(journal)
            elif(amount2 > 0):
                journal = Journal(current_issn2, line[7], line[6], amount2)
                indexes.append(journal)
    return indexes

def CleanList(titles):
    bad_chars = [';', ':', '!', "*", ".",",","(",")", "/"]
    res = []
    for title in titles:
        title = ''.join(j for j in title if not j in bad_chars)
        res.append(title)
    return res

def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

def ToFile(titles,name): 
    with open(name +'.txt', 'w', encoding="utf-8") as f:
        for title in titles:
            f.write(title + '\n')

def AuthorRank(indexes_file):
    sum = 0
    indexes_file = open(indexes_file,'r', encoding="utf-8")
    for line in indexes_file:
        index = line.split(':')[2]
        index = int(index.split(';')[0])
        q = line.split(':')[3]
        q = q.split(';')[0]
        q = q.replace('Q','')
        if(q != '-'):
            q = int(q)
            freq = int(line.split(':')[4])
            if(q >=3): index = -(index * index)/100000
            else: index = pow(index, 2)/100000
            sum += freq * pow(2.71828, index)
    return int(sum)


def NameToFile(name,Small):   
    res = GetTitlesandIssn(name,Small)
    titles = res[0]
    issn = res[1]
    ind = []
    indexes = HIndex(issn)
    for index in indexes:
        ind.append(index.ToStr())
    ToFile(titles,'titles')
    ToFile(ind, 'indexes')
    print(AuthorRank('indexes.txt'))








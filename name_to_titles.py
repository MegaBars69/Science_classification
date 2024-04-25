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

def ClearStr(string):
    bad_chars = [';', ':', '!', "*", ".",",","(",")", "in", "and", "or", "at", "of", "from", "/", "i",]
    separated_string = string.split()
    for el in bad_chars:
        if el in separated_string:
            separated_string.remove(el)
    res = ""
    for b in range(0, len(separated_string), 1):
        res += separated_string[b]+" "
    return res

def ToFile(titles,name): 
    with open(name +'.txt', 'w', encoding="utf-8") as f:
        for title in titles:
            f.write(title + '\n')
            
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






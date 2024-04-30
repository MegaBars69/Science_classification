from crossref.restful import Journals
from collections import defaultdict
import os
journals = Journals()
from contextlib import suppress



def ClearStr(string: str):
    bad_chars = [';', ':', '!', "*", ".",",","(",")","'\'", "/", "i","'",'"',',']
    string = string.replace('(Q1)','')
    string = string.replace('(Q2)','')
    string = string.replace('(Q3)','')
    string = string.replace('(Q4)','')
    string = string.replace(' ','_')
    string = string.replace('"','')


    separated_string = string.split()
    for el in bad_chars:
        if el in separated_string:
            separated_string.replace(el, '')
    res = ""
    for b in range(0, len(separated_string), 1):
        res += separated_string[b]+" "
    return res

def journals_analize(Area : str):
    data = open('journals.csv','r')
    i = 0
    docs = []
    for line in data:
        if(i != 0):
            line = line.split('";"')
            parts = line[3].split(';')
            if(Area in line[3] and len(parts) == 1):
                area = ClearStr(line[3])
                category = ClearStr(line[2])
                #category = category.rstrip(area[-4])
                line = line[0].split(';')
                docs.append((ClearStr(line[4]), category, area))
        i+=1
    print(docs)
    return docs

def issn_to_file(filename: str, issn : str):
    i = 0
    titles = []
    try:    
        for item in journals.works(issn): 
            if(i<=3500):
                if('title' in item):
                    i += 1
                    titles.append(str(item['title'][0]))
                else:break
            else:break
    except:
        return 0
    titles = set(titles)
    try: 
        with open(filename, 'w', encoding="utf-8") as f:
            for title in titles:
                if(len(title.split(' ')) >= 3):
                    f.write(title + '\n')   
    except:
        return 0 
    return i

def find_min_file():
    dir_list = os.listdir('areas') 
    file_size = []
    for dir in dir_list:
        file_size.append((dir, os.path.getsize('areas\\' + dir)))
    min_size = file_size[0][1]
    min_value = file_size[0][0]
    for f in file_size:
        if(f[1] < min_size):
            min_size = f[1]
            min_value = f[0]
    return min_value


def create_dataset(docs):
    i = 0
    category_usage = defaultdict(list)
    docs_len = len(docs)
    step = 1
    for doc in docs:
        print(((100*step)/docs_len), "% \n")
        area = doc[2]
        category = doc[1]
        issn = doc[0].split(',')
        if(category in category_usage.keys()):
            category_usage[category] += 1
        else:
            category_usage[category] = 1
        if(category_usage[category] <= 6):
            filename = area + '_' + category + str(issn) + '.txt'
            filename = ClearStr(filename)
            filename = "dataset\\" + filename
            
            if(len(issn) == 2 and issn_to_file(filename, issn[0]) < 3):            
                i = issn_to_file(filename, issn[1])
                if(i <= 2):
                    category_usage[category] -=1
            elif(issn_to_file(filename, issn[0]) < 2 and len(issn) == 1):
                if(i <= 2):
                    category_usage[category] -=1
            print(filename, 'is passed')
        step += 1

dir_list_areas = os.listdir("areas")
for dir in dir_list_areas:
    area = dir.replace('_',' ')
    area = area.replace('.csv','')
    docs  = journals_analize(area)
    create_dataset(docs)


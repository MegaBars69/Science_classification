from collections import defaultdict
import os
import random
import string
import pandas as pd
import csv

def title_is_fine(title, norm_len):
    digits = '123456789'
    allowed_chars = string.ascii_letters + "'"+ "-" + " " + "." + "!" + digits + '?' + '*' + '/' + '–' + '#' + '"' + ")" + "(" + "%" + "^" + ',' +  ':' + '&' + '{' + "}" + '[' +']' + '=' + '№' + ';' + '_' 
    title = title.replace("  ", "")
    length = len(title.split(' '))
    if(length <= norm_len): return False
    for t in title:
        if t not in allowed_chars:
            return False
    return True

def normolize_dataset(filename,param):
    # Load the input CSV file
    print(filename)
    df = pd.read_csv(filename, quoting=csv.QUOTE_NONE,encoding='utf-8')

    # Calculate the average number of titles per category
    avg_titles = df[param].value_counts().min()
    if(avg_titles >=300):
        # Create a new DataFrame with the same number of titles in each category
        new_df = pd.DataFrame()
        for category in df[param].unique():
            category_df = df[df[param] == category].sample(int(avg_titles))
            new_df = pd.concat([new_df, category_df], ignore_index=True)

        # Save the new DataFrame to a CSV file
        new_df.to_csv(filename, index=False)
    print(df[param].value_counts())


def get_areas(dir_list,Set):
    areas = []
    for dir in dir_list:
        parts = dir.split('__')
        area = parts[0]
        areas.append(area)
    if(Set):return set(areas)
    else: return areas


def concat_in_area(area, dir_list, Normalaize):
    filename = 'areas\\' + area + '.csv'
    with open(filename, 'w', encoding="utf-8") as f:
        f.write('Category,Title\n')
        for dir in dir_list:
            if area in dir:
                parts = dir.split('__')
                categories = []
                cat = ''
                length = len(parts[1])
                i = 0
                for c in parts[1]:
                    i+=1
                    if(length == i):
                        cat+=c
                        categories.append(cat)
                    if(c != ';'):
                        cat+=c
                    else:
                        categories.append(cat)
                        cat = ''
                if(len(categories) == 1 or (((area + '_(miscellaneous)') in  categories) and len(categories)==2)):
                    with open('dataset\\' + dir, 'r', encoding="utf-8") as file:
                        for l in file:
                            l = l.replace(',', ' ')
                            if(Normalaize): 
                                for category in categories:
                                    #if(title_is_fine(l)):
                                    #category = parts[1]
                                    category = category.replace('.txt','')
                                    category = category.replace(',', ' ')
                                    category = category.replace('[','')
                                    category = category.replace(']','') 
                                    category = category.replace("'",'') 
                                    for i in range(0,10,1):
                                        category = category.replace(str(i), '')

                                    miscellaneous = [area + '_(miscellaneous)_', '_' + area + '_(miscellaneous)','_' + area + '_(miscellaneous)_', area + '_(miscellaneous)']
                                    if((len(categories) == 2 and (('miscellaneous' not in category))) or ((len(categories) == 1) and ('miscellaneous' not in category))):
                                        category = category.replace(',', ' ')
                                        category = category.replace('_',' ')
                                        category = category.replace('[','')
                                        category = category.replace(']','') 
                                        category = category.replace('"','')                                                        
                                        line = category + ","+ l 
                                        f.write(line)
                            else:
                                category = categories[0]
                                category = category.replace(',', ' ')
                                category = category.replace('_',' ')
                                category = category.replace('[','')
                                category = category.replace(']','') 
                                category = category.replace('"','')                                                        
                                line = category + ","+ l 
                                f.write(line)
    if(Normalaize): normolize_dataset(filename,'Category')
        
def concat_all(dir_list, Normalize):
    list_of_areas = get_areas(dir_list,True)
    area_count = 0
    for area in list_of_areas:
        concat_in_area(area, dir_list, Normalize)
        area_count+=1
        print(area_count/len(list_of_areas))

def concat_f(list_of_files):
    with open('final_dataset.csv', 'w', encoding="utf-8") as f:
        f.write('Area,Title\n')
        for name in list_of_files:
            amount_of_strings = 0
            filename = 'areas\\' + name
            file = open(filename, 'r', encoding="utf-8")
            file = file.read()
            file = file.split('\n')
            #random.shuffle(file)
            for l in file:
                l = l.split(',')
                if(l != file[0]):
                    name = name.replace(',','')
                    name = name.replace('.csv','')
                    if(len(l)>1 and (title_is_fine(l[1],13) and name == 'Economics_Econometrics_and_Finance' or title_is_fine(l[1],15) and name != 'Economics_Econometrics_and_Finance')):
                        l[1] = l[1].replace("  ", '')
                        line = name + ","+ l[1] + '\n'
                        f.write(line)
                        amount_of_strings+=1   

def concat_f3(list_of_files):
    with open('final_dataset2_train.csv', 'w', encoding="utf-8") as f:
        with open('final_dataset2_test.csv', 'w', encoding="utf-8") as f2:
            all = open('final_dataset.csv', 'r', encoding="utf-8")
            all = all.read()
            all = all.split('\n')
            amount_of_strings = 0
            first_str = ''
            for name in list_of_files:
                n = name.replace('.csv','')
                n = n.replace(',','')
                first_str += n + ', '
            f.write('Title,' + first_str + '\n')
            f2.write('Title,' + first_str + '\n')
            for l in all:
                l = l.split(',')
                name = l[0]
                name = name.replace('.csv','')
                name = name.replace(',','')
                if(l != all[0]):
                    if(len(l)>1 and title_is_fine(l[1])):
                        l[1] = l[1].replace("  ", '')
                        line = l[1] +','
                        i = 0
                        while (name != (list_of_files[i].replace(',','')).replace('.csv','')):
                            if(i!= 0): line +=',0'
                            else: line +='0'
                            i+=1
                        if(i!=0):line+=',1'
                        else:line+='1'
                        i+=1
                        for j in range(i, len(list_of_files)):
                            line +=',0' 
                        line +='\n'
                        if(amount_of_strings%4 == 0):
                            f2.write(line)
                        else: 
                            f.write(line)
                        amount_of_strings+=1  
                

def concat_f2(list_of_files):
    with open('final_dataset2_train.csv', 'w', encoding="utf-8") as f:
        with open('final_dataset2_test.csv', 'w', encoding="utf-8") as f2:
            first_str = ''
            for name in list_of_files:
                n = name.replace('.csv','')
                n = n.replace(',','')
                print(n)
                first_str += n + ', '
            f.write('Title,' + first_str + '\n')
            f2.write('Title,' + first_str + '\n')
            index = 0
            for name in list_of_files:
                amount_of_strings = 0
                filename = 'areas\\' + name
                file = open(filename, 'r', encoding="utf-8")
                file = file.read()
                file = file.split('\n')
                #random.shuffle(file)
                for l in file:
                    l = l.split(',')
                    if(l != file[0]):
                        if(len(l)>1 and title_is_fine(l[1])):
                            l[1] = l[1].replace("  ", '')
                            line = l[1] +','
                            i = 0
                            while (name != list_of_files[i]):
                                if(i!= 0): line +=',0'
                                else: line +='0'
                                i+=1
                            if(i!=0):line+=',1'
                            else:line+='1'
                            i+=1
                            for j in range(i, len(list_of_files)):
                                line +=',0' 
                            line +='\n'
                            if(amount_of_strings%4 == 0):
                                f2.write(line)
                            else: 
                                f.write(line)
                            amount_of_strings+=1  
                index+=1
                print(index/len(list_of_files))            
               
    
#concat_f(titles_list)  
dir_list_areas = os.listdir("areas")
dir_list_dataset = os.listdir("dataset")

"""res = []
for dir in dir_list_areas:
    na = dir.replace('.csv','')
    na = na.replace(',','')
    res.append(na)
print(res)
"""
"""
#normilize_dataset('final_dataset.csv', 'normilized.csv')
concat_all(dir_list_dataset, False)
print('Concacted_all')
concat_f(get_areas(dir_list_areas, False))
print('Final Datasset is ready')
train_df = pd.read_csv('final_dataset.csv')
print(train_df['Area'].value_counts())
normolize_dataset('final_dataset.csv','Area')
concat_all(dir_list_dataset, True)

"""
import os, random, string
from nltk import word_tokenize, FreqDist
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
import pandas as pd
from Utils.name_to_titles import *


def setup_docs(filename):
    docs = [] #(label, text)
    with open(filename, 'r', encoding="utf-8") as f:
        for row in f:
            parts = row.split(',')
            doc = (parts[0], parts[1].strip())

            docs.append(doc)
    return docs

def clean_text(text):
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.lower()
    return text

def get_tokens(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stop_words.add('said')
    stop_words.add('mr')
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

def print_frquency_dist(docs):
    tokens = defaultdict(list)

    for doc in docs:
        doc_label = doc[0]
        doc_text = clean_text(doc[1])
        
        doc_tokens = get_tokens(doc_text)
        tokens[doc_label].extend(doc_tokens)
    
    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))

def get_splits(docs):
    random.shuffle(docs)
    X_train = [] #trainning documents
    X_test = [] 
    
    y_train = [] #trainning labels
    y_test = []
    pivot = int(0.9*len(docs))

    for i in range(0, pivot):
        X_train.append(docs[i][1])
        y_train.append(docs[i][0])
    for i in range(pivot, len(docs)):
        X_test.append(docs[i][1])
        y_test.append(docs[i][0])
    return X_train, X_test, y_train, y_test

def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    precision = metrics.precision_score(y_test, y_pred, average= 'weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    
    print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))

def train_classifier(docs, save_filename):
    save_filename = save_filename.replace(',','')
    X_train, X_test, y_train, y_test = get_splits(docs)

    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3), min_df=3,analyzer='word')

    dtm = vectorizer.fit_transform(X_train) #creates dic-term matrix

    naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)
    
    print(save_filename + '\n')

    evaluate_classifier('Naive Bayes\t Train', naive_bayes_classifier, vectorizer, X_train, y_train)
    evaluate_classifier('Naive Bayes\t Test', naive_bayes_classifier, vectorizer, X_test, y_test) 

    #store classifier
    clf_filename = 'weights/' + save_filename + '.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))

    vec_filename = 'weights/' + save_filename + '_vectorizer.pkl'
    pickle.dump(vectorizer, open(vec_filename, 'wb'))

def categorization(path):
    dir_list = os.listdir(path)
    for name in dir_list:
        docs = setup_docs(path + '/' + name)
        name = name.replace('.csv', '')
        train_classifier(docs, name)  

if __name__ == '__main__':
    
    docs = setup_docs('final_dataset.csv')
    train_classifier(docs, 'Areas')

    categorization('areas')
    
from googletrans import Translator
import os
import nltk
from nltk.util import bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from habanero import Crossref
from Utils.train_model import *
from Utils.name_to_titles import *
from text_classifier import *
# Create a Crossref object
cr = Crossref()
import string
from nltk.stem import WordNetLemmatizer
import time
import spacy
import numpy as np
import asyncio
# Load the spaCy model
from nltk.stem import PorterStemmer

from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

from search_in_csv import istina_search
from transformers import BertForSequenceClassification

from sklearn.metrics.pairwise import cosine_similarity
stop_words_eng = set(stopwords.words('english'))
stop_words_rus = set(stopwords.words('russian'))

import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка токенизатора и модели DistilBERT
tokenizer_rus = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model_rus = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
model_rus.eval()  # Установка модели в режим оценки

# Перемещение модели на GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_rus.to(device)

# Функция для получения векторов из DistilBERT
def get_distilbert_embeddings_rus(text):
    inputs = tokenizer_rus(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model_rus(**inputs)
    # Получаем вектор из последнего скрытого слоя
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

def get_bert_embeddings_rus(texts):
    # Токенизация пакета текстов
    inputs = tokenizer_rus(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        # Получаем выходные данные из модели DistilBERT
        outputs = model_rus(**inputs)
    
    # Получаем векторы для [CLS] токена (или усредняем по всем токенам)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Усреднение по всем токенам
    return embeddings.cpu().numpy()  # Конвертация в numpy массив и возврат на CPU

# Функция для сравнения текстов с использованием DistilBERT
def compare_similarity_bert_batch_rus(texts1, vectorized_text2):
    embeddings1 = get_bert_embeddings_rus(texts1)
    
    # Вычисляем косинусное сходство
    similarities = []
    for emb1 in embeddings1:
        similarity = cosine_similarity(emb1.reshape(1, -1), vectorized_text2)
        similarities.append(similarity[0][0])  # Сохраняем только одно значение сходства
    
    return similarities


tokenizer_eng = BertTokenizer.from_pretrained('bert-base-uncased')
model_eng = BertModel.from_pretrained('bert-base-uncased')
model_eng.eval()  # Set the model to evaluation mode

# Move the model to GPU if available
model_eng.to(device)

def get_bert_embeddings_batch_eng(texts):
    # Tokenize the batch of texts
    inputs = tokenizer_eng(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        # Get the outputs from the BERT model
        outputs = model_eng(**inputs)
    
    # Get the embeddings for the [CLS] token (or average over all tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average over all tokens
    return embeddings.cpu().numpy()  # Convert to numpy array and return to CPU

def compare_similarity_bert_batch_eng(texts1, emb_of_main):
    # Get embeddings for both batches of texts
    embeddings1 = get_bert_embeddings_batch_eng(texts1)
    
    # Calculate cosine similarities
    similarities = []
    for emb1 in embeddings1:
        similarity = 1 - cosine(emb1, emb_of_main)
        similarities.append(similarity)
    
    return similarities


def get_bert_embeddings_eng(text):
    # Загрузка модели BERT и токенизатора
    
    # Токенизация текста
    inputs = tokenizer_eng(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        # Получение выходных данных из модели BERT
        outputs = model_eng(**inputs)
    
    # Получаем вектор, соответствующий [CLS] токену (используем его как представление всего текста)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Среднее по всем токенам
    return embeddings.squeeze().numpy()  # Преобразуем в одномерный массив

def compare_similarity_bert_eng(text1, vectorized_text2):
    # Получаем векторы для двух текстов
    embeddings1 = get_bert_embeddings_eng(text1)
    
    # Считаем косинусное расстояние между векторами
    similarity = 1 - cosine(embeddings1, vectorized_text2)
    return similarity


def extract_important_words_en(text):
    nlp = spacy.load("en_core_web_md")

    # Process the text using spaCy
    doc = nlp(text)
    
    # Extract nouns and verbs (which tend to have more significance in a sentence)
    words = [token.lemma_ for token in doc if (not token.is_stop or token.is_digit) and not token.is_punct]
    
    return words

def extract_important_words_ru(text):
    # Load the Russian language model
    nlp = spacy.load('ru_core_news_md')  # Ensure you have the Russian model installed

    # Process the text using spaCy
    doc = nlp(text)
    
    # Extract nouns and verbs (which tend to have more significance in a sentence)
    words = [token.lemma_ for token in doc if (not token.is_stop or token.is_digit) and not token.is_punct]
    
    return words


def compare_words_to_sentence(word_list, sentence):
    nlp = spacy.load("en_core_web_md")

    # Create a spaCy Doc object for the sentence
    sentence_doc = nlp(sentence)
    
    # Create a spaCy Doc object for the list of words
    word_docs = [nlp(word) for word in word_list]
    
    # Calculate the average vector of the list of words
    word_vector = np.mean([word_doc.vector for word_doc in word_docs], axis=0)
    
    # Create a fake Doc object using the word vector to represent the word list
    # Since similarity expects a Doc, we can create a temporary Doc with no actual words, just the vector.
    word_vector_doc = nlp.make_doc("")  # Create an empty Doc
    word_vector_doc.vector = word_vector  # Assign the custom vector
    
    # Calculate the similarity between the sentence Doc and the "word list" Doc
    similarity = sentence_doc.similarity(word_vector_doc)
    
    return similarity

def remove_repeating_words(input_string):
    # Remove punctuation using translate
    table = str.maketrans('', '', string.punctuation)
    input_string = input_string.translate(table)
    
    # Split the input string into words
    words = input_string.split()
    seen = set()  # To keep track of the words we've already encountered
    result = []

    for word in words:
        # Convert word to lowercase for case-insensitive comparison
        word_lower = word.lower()

        if word_lower not in seen:
            seen.add(word_lower)
            result.append(word)  # Append original word (preserving case)

    return ' '.join(result)  # Join the words back into a string


# Define the function to find titles and authors by keywords
def find_titles_and_authors_by_keywords(keywords):
    query = ' OR '.join(keywords)  # Combine keywords with OR for the search
    results = cr.works(query=query)
    titles_authors = []
    if results['message']['total-results'] > 0:
        for item in results['message']['items']:
            title = item['title'][0] if 'title' in item else 'No Title'
            # Safely extract authors
            authors = []
            for author in item.get('author', []):
                given = author.get('given')
                family = author.get('family')
                if given and family:  # Only add if both names are present
                    authors.append(f"{given} {family}")
            if authors:  # Only add to results if there are authors
                titles_authors.append((title, ', '.join(authors)))  # Store title and authors
    return titles_authors


def get_titles_with_keywords(keywords):
    results = []
    
    for keyword in keywords:
        # Search CrossRef API with the given keyword
        response = cr.works(query=keyword)
        
        # Loop through the works returned by the API
        for item in response['message']['items']:
            title = item.get('title', ['No title available'])[0]
            authors = item.get('author', [])
            
            # Check if title has at least 3 words
            title_for_work = remove_repeating_words(title).split()
            if len(title_for_work) >= 2:
                # Find distinct occurrences of the keyword in the title
                keyword_instances = [word.lower() for word in title_for_work if keyword.lower() in word.lower()]
                
                # We need at least two distinct occurrences of the keyword
                if len(set(keyword_instances)) >= 2:
                    if authors and authors != " " and authors != "_" and authors != '' and len(authors) >=1:  # Only add the article if there are authors
                        author_names = [f"{author.get('given', '')} {author.get('family', '')}" for author in authors]
                        author_str = ", ".join(author_names) if author_names else 'No authors listed'
                        results.append((title, author_str,'cros'))  # Store title and authors
    
    return results

async def translate_file(input_file_path, output_file_path):
    # Create a Translator object
    translator = Translator()

    # Read the content of the input file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        russian_text = file.read()

    # Translate the text from Russian to English
    translated_text = await translator.translate(russian_text, src='ru', dest='en')

    # Write the translated text to the output file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(translated_text)

def extract_topic_and_keywords_eng(text, amount_of_key_words):
    # Tokenize the text
    words = word_tokenize(text.lower())
    
    # Remove stop words and non-alphabetic tokens
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Identify the topic as the most common word
    topic = word_counts.most_common(1)[0][0] if word_counts else None
    
    # Use TF-IDF to extract keywords, including n-grams
    vectorizer = TfidfVectorizer(stop_words='english', max_features=amount_of_key_words , ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the top keywords based on TF-IDF scores
    keywords = feature_names[tfidf_matrix.toarray().argsort()[0][-(amount_of_key_words ):]].tolist()
    
    # Filter out individual words that are part of any bigram
    filtered_keywords = []
    bigrams = set()

    for keyword in keywords:
        if ' ' in keyword:  # It's a bigram
            bigrams.add(keyword)
        else:  # It's a unigram
            # Only add the unigram if it's not part of any bigram
            if not any(keyword in bigram for bigram in bigrams):
                filtered_keywords.append(keyword)

    # Add the bigrams to the final list of keywords
    filtered_keywords.extend(bigrams)

    return topic, keywords

def extract_topic_and_keywords_ru(text, amount_of_key_words):
    # Tokenize the text
    words = word_tokenize(text.lower(), language='russian')
    
    # Remove stop words and non-alphabetic tokens
    stop_words = set(stopwords.words('russian'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Identify the topic as the most common word
    topic = word_counts.most_common(1)[0][0] if word_counts else None
    
    # Use TF-IDF to extract keywords, including n-grams
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=amount_of_key_words, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the top keywords based on TF-IDF scores
    keywords = feature_names[tfidf_matrix.toarray().argsort()[0][-(amount_of_key_words):]].tolist()
    
    # Filter out individual words that are part of any bigram
    filtered_keywords = []
    bigrams = set()

    for keyword in keywords:
        if ' ' in keyword:  # It's a bigram
            bigrams.add(keyword)
        else:  # It's a unigram
            # Only add the unigram if it's not part of any bigram
            if not any(keyword in bigram for bigram in bigrams):
                filtered_keywords.append(keyword)

    # Add the bigrams to the final list of keywords
    filtered_keywords.extend(bigrams)

    return topic, filtered_keywords

from googlesearch import search

def find_social_media_profiles(name):    
    # Perform a Google search for the person's name
    search_results = search(name)  # Get results
    
    # Convert the search results to a list and limit to top 5
    results_list = list(search_results)[:5]  # Get top 5 results
    
    return results_list

def write_titles_sorted_by_index(titles_authors, output_file_path, key_words):    
    # Collect all titles with their authors and indices
    all_titles = []

    for title, authors in titles_authors:

        index = compare_words_to_sentence(key_words, title)  # Calculate the index for the title
        all_titles.append((title, authors, index))  # Store title, authors, and index

    # Sort all titles by index in descending order
    sorted_titles = sorted(all_titles, key=lambda x: x[2], reverse=True)

    # Write the sorted titles to a file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for title, authors, index in sorted_titles:
            file.write(f"Authors: {authors};  Category: {classify(title)}; Title: {title}; Index: {index}\n")

    print(f"Titles written to {output_file_path} sorted by index.")


def remove_stop_words(text,lang):
    words = word_tokenize(text.lower())
    words_title = ""
    if lang == 'eng':
        stop_words = stop_words_eng
    else:
        stop_words = stop_words_rus

    for word in words:
        if word.isalpha() and word not in stop_words:
            words_title += word + " "
    return words_title

def sort_titles_by_index_eng(titles_authors, key_words):
    # Collect all titles with their authors and indices
    all_titles = []
    sentence_of_key_words = ""
    for word in key_words:
        sentence_of_key_words += " " + word
    
    vectorized_title = get_bert_embeddings_eng(sentence_of_key_words)

    titles = [remove_stop_words(el[0],'eng') for el in titles_authors]
    authors = [el[1] for el in titles_authors]
    sourses = [el[2] for el in titles_authors]

    indexes = compare_similarity_bert_batch_eng(titles, vectorized_title)  # Calculate the index for the title
    
    all_titles = list(zip((el[0] for el in titles_authors), authors, indexes,sourses))  # Store title, authors, and index

    # Sort all titles by index in descending order
    sorted_titles = sorted(all_titles, key=lambda x: x[2], reverse=True)
    sorted_titles_list = []
    # Return the sorted titles as a list of strings (or as tuples, depending on your preference)
    for title, authors, index,sour in sorted_titles:
        sorted_titles_list.append({'author': authors, 'title': title, 'index': index, 'references' : [], 'category' : 'classify(title)', 'source':sour})
    return sorted_titles_list

def sort_titles_by_index_rus(titles_authors, key_words):
    all_titles = []
    sentence_of_key_words = ""
    for word in key_words:
        sentence_of_key_words += " " + word
    
    vectorized_title = get_bert_embeddings_rus(sentence_of_key_words)
    vectorized_title = vectorized_title.reshape(1, -1)

    titles = [remove_stop_words(el[0],'rus') for el in titles_authors]
    authors = [el[1] for el in titles_authors]
    sourses = [el[2] for el in titles_authors]

    indexes = compare_similarity_bert_batch_rus(titles, vectorized_title)  # Calculate the index for the title
    
    all_titles = list(zip((el[0] for el in titles_authors), authors, indexes,sourses))  # Store title, authors, and index

    # Sort all titles by index in descending order
    sorted_titles = sorted(all_titles, key=lambda x: x[2], reverse=True)
    sorted_titles_list = []
    # Return the sorted titles as a list of strings (or as tuples, depending on your preference)
    for title, authors, index,sour in sorted_titles:
        sorted_titles_list.append({'author': authors, 'title': title, 'index': index, 'references' : [], 'category' : 'classify(title)', 'source':sour})
    return sorted_titles_list
    
def extract_important_words_and_bigrams(sentence):
    # Define stopwords list
    stop_words = set(stopwords.words('english'))
    
    # Tokenize the sentence
    words = word_tokenize(sentence.lower())  # Convert to lower case for consistency
    
    # Remove stopwords and non-alphabetical words
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Extract bigrams
    bigrams_list = list(bigrams(filtered_words))
    
    # Combine important words and bigrams
    result = filtered_words + [' '.join(bigram) for bigram in bigrams_list]
    
    return result


def extract_key_phrases(text):
    nlp = spacy.load("en_core_web_md")

    doc = nlp(text)
    
    key_phrases = []
    
    # Извлекаем существительные и их фразы (игнорируя стоп-слова)
    for np in doc.noun_chunks:
        # Проверяем, что в фразе нет стоп-слов
        if not any(token.is_stop for token in np):
            key_phrases.append(np.text.lower())  # Нормализуем в нижний регистр
    
    # Извлекаем глаголы (игнорируя стоп-слова)
    for token in doc:
        if token.pos_ == "VERB" and not token.is_stop:
            key_phrases.append(token.lemma_)  # Лемматизация глагола (например, "develop")
    
    # Извлекаем именованные сущности (игнорируя стоп-слова)
    for ent in doc.ents:
        if not any(token.is_stop for token in ent):
            key_phrases.append(ent.text.lower())
    
    # Убираем дубликаты
    key_phrases = list(set(key_phrases))
    
    # Убираем слишком короткие или незначительные фразы
    key_phrases = [phrase for phrase in key_phrases if len(phrase) > 2]
    
    return key_phrases

lemmatizer = WordNetLemmatizer()

def get_word_forms(words):
    word_forms = set()

    for word in words:
        # Add the original word
        word_forms.add(word)

        # Get the base form (lemma)
        base_form = lemmatizer.lemmatize(word)

        # Add the base form
        word_forms.add(base_form)

        # Handle specific cases for common forms
        if word.endswith('ing'):
            # Add the base verb form (e.g., swimming -> swim)
            word_forms.add(lemmatizer.lemmatize(base_form, pos='v'))
        elif word.endswith('s'):
            # Add the singular form (e.g., conditions -> condition)
            word_forms.add(lemmatizer.lemmatize(base_form))
        elif word.endswith('ment'):
            word_forms.add(word[:-4])
    
    return list(word_forms)


def search_crossref(keywords):
    # Преобразуем список ключевых слов в строку для запроса
    if isinstance(keywords, list):
        query = ' OR '.join(keywords)
    else:
        query = keywords

    # Формируем URL для запроса
    url = f"https://api.crossref.org/works?query={query}&rows=100"

    # Выполняем GET-запрос
    response = requests.get(url)

    # Проверяем статус ответа
    if response.status_code == 200:
        data = response.json()
        articles = data.get('message', {}).get('items', [])
    else:
        print(f"Ошибка: {response.status_code}")
        return []
    answer = [] 
    for article in articles:
        title = article.get('title', [''])[0]  # Получаем заголовок статьи
        if len(title.split()) >2:
            #doi = article.get('DOI', 'Нет DOI')  # Получаем DOI статьи
            authors = article.get('author', [])  # Получаем список авторов
            author_names = ', '.join([f"{author.get('given', '')} {author.get('family', '')}" for author in authors])  # Формируем строку с именами авторов
            #print(f"Title: {title}, DOI: {doi}, Authors: {author_names if author_names else 'Нет авторов'}")
            if author_names:
                answer.append((title, author_names, 'Crossref'))
    return answer


async def get_people_from_query_eng(query_title, query_content, dict_data):
    # Get the input file path from the user
    translator = Translator()
    input_title_name_eng = await translator.translate(query_title, src='ru', dest='en')
    input_title_name_eng = input_title_name_eng.text
    #ENGLISH SEARCH
    start_time = time.time()

    input_text_eng = await translator.translate(query_content, src='ru', dest='en')
    topic, keywords_eng = extract_topic_and_keywords_eng(input_text_eng.text, 6)
    title_main_words_eng = extract_important_words_en(input_title_name_eng)
    all_keywords_eng = get_word_forms(list(set(keywords_eng + title_main_words_eng)))
    
    words = word_tokenize(input_title_name_eng.lower())
    
    # Remove stop words and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    filtered_words_eng = [word for word in words if word.isalpha() and word not in stop_words]
    end_time = time.time()
    execution_time = end_time - start_time
    print("Important words: ",all_keywords_eng )
    print(execution_time)

    # Crossref Search
    start_time = time.time()

    titles_authors = search_crossref(all_keywords_eng)
    end_time = time.time()
    execution_time = end_time - start_time

    print("Crossref searched")
    print(execution_time)
    #Istina search
    start_time = time.time()
    titles_from_istina = istina_search(all_keywords_eng, dict_data)
    end_time = time.time()
    execution_time = end_time - start_time

    print("Istina searched")
    print(execution_time)
    all_authors = titles_authors + titles_from_istina
    start_time = time.time()

    #write_titles_sorted_by_index(titles_authors, output_file_path, keywords)
    sort_res_eng = sort_titles_by_index_eng(all_authors, filtered_words_eng + keywords_eng)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Sorted")
    print(execution_time)
    return sort_res_eng

def get_people_from_query_rus(query_title, query_content, dict_data):
    
    start_time = time.time()

    #RUSSIAN SEARCH    
    topic, keywords_rus = extract_topic_and_keywords_ru(query_content, 6)
    title_main_words_rus = extract_important_words_ru(query_title)
    all_keywords_rus = get_word_forms(list(set(keywords_rus + title_main_words_rus)))
    
    words = word_tokenize(query_title.lower())
    
    # Remove stop words and non-alphabetic tokens
    stop_words = set(stopwords.words('russian'))
    filtered_words_rus = [word for word in words if word.isalpha() and word not in stop_words]
    end_time = time.time()
    execution_time = end_time - start_time
    print("Important words", all_keywords_rus)
    print(execution_time)

    start_time = time.time()
    # Crossref Search
    titles_authors_rus = search_crossref(all_keywords_rus)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Crossref searched")
    print(execution_time)
    
    start_time = time.time()

    #Istina search
    titles_from_istina_rus = istina_search(all_keywords_rus,dict_data)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Istina searched")
    print(execution_time)

    start_time = time.time()

    all_authors_rus = titles_authors_rus + titles_from_istina_rus
 
    #write_titles_sorted_by_index(titles_authors, output_file_path, keywords)
    sort_res_rus = sort_titles_by_index_rus(all_authors_rus, filtered_words_rus + keywords_rus)
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)

    return sort_res_rus

def write_list_to_file(file_path, lines):
    """
    Writes each ring or dictionary from the list 'lines' to a file specified by 'file_path', 
    with each entry on a new line.

    :param file_path: The path to the file where the lines will be written.
    :param lines: A list of strings or dictionaries to write to the file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:  # Open the file in write mode
        for line in lines:
            if isinstance(line, dict):
                # Customize the output format as needed
                # For example, writing 'author' and 'title' fields
                file.write(f"Author: {line.get('author', 'N/A')}, Title: {line.get('title', 'N/A')}\n")
            else:
                file.write(line + '\n')
   
if __name__ == "__main__":
    
    """
    query_title = "Разработка трехмерного акустического расчетного кода"
    query_content = "Разработать трехмерный акустический расчетный код, применимый для расчета распространения шума в каналах турбомашин, а также во входных и выходных устройствах авиационных двигателей Обеспечить в расчетном коде следующие возможности: 1. выполнять расчеты на структурированных и неструктурированных расчетных сетках. Иметь инструменты построения расчетных сеток и/или их импорта, 2. выполнять расчеты на расчетных сетках размерностью не менее 1,5 млрд. ячеек 3. учитывать неоднородное стационарное среднее течение внутри расчетной области, 4. использовать средние газодинамические поля, рассчитанные в сторонних решателях (таких как Ansys CFX и Fluent, Логос), 5. учитывать звукопоглощающие конструкции на поверхностях каналов в виде импедансных граничных условий, 6. задавать неотражающие граничных условий, моделирующие уход волн из расчетной области без отражений, 7. задавать модальные граничные условий для вноса возмущений в расчетную область в виде суперпозиции собственных форм акустических колебаний для цилиндрического, кольцевого и прямоугольного каналов на различных частотах, 8. иметь Linux реализацию, 9. поддерживать параллелизацию между вычислительными узлами."
    
    write_list_to_file('answ.txt',get_people_from_query_eng(query_title, query_content))
    
    translator = Translator()
    input_title_name_eng = translator.translate(query_title, src='ru', dest='en').text

    #ENGLISH SEARCH
    input_text_eng = translator.translate(query_content, src='ru', dest='en').text
    print(input_title_name_eng)
    topic, keywords_eng = extract_topic_and_keywords_eng(input_text_eng, 7)
    print(keywords_eng)
    title_main_words_eng = extract_important_words_en(input_title_name_eng)
    print(ti_main_words_eng)
    all_keywords_eng = get_word_forms(list(set(keywords_eng + title_main_words_eng)))
    print(all_keywords_eng)
    
    
    # Example usage
    if __name__ == "__main__":
        first_name = input("Enter first name: ").strip()
        last_name = input("Enter last name: ").strip()
        
        profiles = find_social_media_profiles(first_name, last_name)
        
        print("Top 5 social media profiles or websites:")*
        for index, profile in enumerate(profiles, start=1):
            print(f"{index}. {profile}")
    """


import pandas as pd

# Загрузка данных из CSV-файла в словарь
def load_istina_data(csv_file='data_upd.csv'):
    df = pd.read_csv(csv_file, encoding='utf-8')
    istina_dict = df.to_dict('records')  # Преобразуем DataFrame в список словарей
    return istina_dict

# Функция поиска по словарю
def istina_search(words, istina_dict):
    result = []
    
    # Нормализация слов для регистронезависимого поиска
    words = [(' ' + word.lower() + ' ') for word in words]

    # Итерация по каждому элементу в словаре
    for row in istina_dict:
        title = row['Title']
        author = row['Authors']
        # Проверка, является ли title строкой
        if isinstance(title, str):
            # Подсчет количества совпадений слов в заголовке
            matches = sum(1 for word in words if word in title.lower())
            
            # Добавление в результат, если найдено не менее двух совпадений
            if matches >= 2:
                for char in ['[',']',"'"]:
                    author = author.replace(char,'')
                author = author.replace(';',',')
                result.append((title, author, 'Istina'))
    
    return result

"""
def istina_search(words, csv_file='istina_data/data_upd.csv'):
    result = []
    
    # Normalize the words to lowercase for case-insensitive matching
    words = [(' ' + word.lower() + ' ') for word in words]

    # Read the CSV file using pandas
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        title = row['Title']
        author = row['Authors']
        # Check if title is a string
        if isinstance(title, str):
            # Count how many words from the list are in the title
            matches = sum(1 for word in words if word in title.lower())
            
            # Add to result if at least two words match
            if matches >= 2:
                for char in ['[',']',"'"]:
                    author = author.replace(char,'')
                author = author.replace(';',',')
                result.append((title, author,'Istina'))
    
    return result
"""
def write_list_to_file(file_path, elements):
    try:
        # Open the file in write mode
        with open(file_path, 'w', encoding='utf-8') as file:
            # Write each element in the list to the file
            for element in elements:
                file.write(f"{element}\n")  # Write each element on a new line
        print(f"Successfully written to {file_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


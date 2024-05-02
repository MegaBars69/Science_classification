from g4f.client import Client

client = Client()
def ask(promt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": promt}],
    )
    print(response.choices[0].message.content)

file_path = 'input.txt'
 
with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()
    
promt = 'find and return 4 lists (in structure: (nuber of list) : [only data in list, without any artifacts]): first includes only: "topics", second  includes only: "speakers", third includes only: "time and data",  fourth includes only:"place" ' + file_content
ask(promt)
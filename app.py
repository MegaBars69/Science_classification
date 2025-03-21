from flask import Flask, render_template, request, jsonify, redirect, url_for
import main  # Assuming your search logic is in main.py
from search_in_csv import load_istina_data 
import asyncio
from main import get_people_from_query_eng
from main import get_people_from_query_rus
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    query_title = '' 
    query_content = ''

    if request.method == 'POST':
        # Only process the search when the form is submitted
        query_title = request.form['query_title']
        query_content = request.form['query_content']
 
        # Redirect to the results page with the query parameters
        return redirect(url_for('results', query_title=query_title, query_content=query_content))

    # Render the page with empty or previously filled fields
    return render_template('index.html', results_eng=[], results_rus=[], query_title=query_title, query_content=query_content)

@app.route('/results')
async def results():
    query_title = request.args.get('query_title', '')
    query_content = request.args.get('query_content', '')

    results_eng = await get_people_from_query_eng(query_title, query_content, dict_data)
    results_rus = await asyncio.to_thread(get_people_from_query_rus, query_title, query_content, dict_data)

    return render_template('index.html', results_eng=results_eng, results_rus=results_rus)

"""    # Call the search logic for both English and Russian
    results_eng = await main.get_people_from_query_eng(query_title, query_content, dict_data)
    results_rus = await main.get_people_from_query_rus(query_title, query_content, dict_data)
    
    return render_template('index.html', results_eng=results_eng, results_rus=results_rus, query_title=query_title, query_content=query_content)
"""
@app.route('/get_references', methods=['POST'])
def get_references():
    # Get the author from the request
    author = request.json.get('author')

    # Get the references based on the author
    references = main.find_social_media_profiles(author)

    # Return the references as a JSON response  
    return jsonify(references)  

if __name__ == '__main__': 
    dict_data = load_istina_data()
    app.run(debug=True, host='0.0.0.0')

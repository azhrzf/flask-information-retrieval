from flask import Flask, render_template, url_for, redirect, request
import pandas as pd
from static.code.docs_list import documents_list, labels_list, docs_key_list, docs_list, add_docs, scores

app = Flask(__name__)

dataset = pd.read_csv('static/assets/datasetTKI.csv')
docs_key = dataset.iloc[:, 0].values  # d1, d2, d3, etc
docs = dataset.iloc[:, -2].values  # buat yang feature / independent
label = dataset.iloc[:, -1].values  # politic olahraga etc

docs_list(docs_key, docs, label)


@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['input_query']
        cos_similarity, class_hasil = scores(query)

        return render_template('home.html', documents=[docs_key_list, documents_list, labels_list], cos_similarity=cos_similarity, class_hasil=class_hasil)

    input_term = request.args.get('input_term')
    input_class = request.args.get('input_class')

    if input_term and input_class:
        add_docs(input_term, input_class)

    return render_template('home.html', documents=[docs_key_list, documents_list, labels_list], cos_similarity=None, class_hasil=None)


@app.route("/add", methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        input_term = request.form['input_term']
        input_class = request.form['input_class']

        return redirect(url_for('home', input_term=input_term, input_class=input_class))

    return render_template('add.html')

# debug=True

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

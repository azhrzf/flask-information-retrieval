import nltk
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

documents_list = []
labels_list = []
docs_key_list = []
processed_documents_list = []

def docs_list(docs_key, docs, label):
    for i in range(len(docs)):
        documents_list.append(docs[i])

    for i in range(len(label)):
        labels_list.append(label[i])

    for i in range(len(docs_key)):
        docs_key_list.append(docs_key[i])

def add_docs(input_doc, input_label):
    documents_list.append(input_doc.lower())
    docs_key_list.append("d{}".format(len(docs_key_list)+1))
    labels_list.append(input_label.upper())

def scores(query):
    processed_documents_list = []

    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()

    for i in range(len(documents_list)):
        tmp = stopword.remove(documents_list[i])
        processed_documents_list.append(tmp)

    vectorizer = TfidfVectorizer()
    vectorized_docs = vectorizer.fit_transform(documents_list)

    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_vector, vectorized_docs)

    if len(labels_list) != 0:
        dict_data = {'document': docs_key_list, 'term yang mewakili dokumen': processed_documents_list,
                     'CLASS': labels_list, 'nilai kesamaan': similarity_scores[0]}

        df_dokumen = pd.DataFrame(dict_data)

        dokumen_sorted = df_dokumen.sort_values(
            by=['nilai kesamaan'], ascending=False)

        sorted_class = dokumen_sorted['CLASS'].values[:-1]

        if len(labels_list) % 2 == 0:
            counter_class = Counter(sorted_class)
            keys = counter_class.keys()
            num_values = len(keys)
        else:
            counter_class = Counter(dokumen_sorted['CLASS'])
            keys = counter_class.keys()
            num_values = len(keys)

        category_query = max(counter_class, key=counter_class.get)

        return [similarity_scores, category_query]
    else:
        return [similarity_scores, None]
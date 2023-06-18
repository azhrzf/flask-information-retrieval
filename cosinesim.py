from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

dataset = pd.read_csv('static/assets/datasetTKI.csv')
docs_key = dataset.iloc[:, 0].values  # d1, d2, d3, etc
docs = dataset.iloc[:, -2].values  # buat yang feature / independent
label = dataset.iloc[:, -1].values  # politic olahraga etc

print(docs)

print(docs_key)  # list
print(docs)  # list
print(label)

documents_list = []  # rubah dari numpy array ke list
for i in range(len(docs)):
    documents_list.append(docs[i])

labels_list = []  # rubah dari numpy array ke list
for i in range(len(label)):
    labels_list.append(label[i])

docs_key_list = []  # rubah dari numpy array ke list
for i in range(len(docs_key)):
    docs_key_list.append(docs_key[i])

documents_list, labels_list, docs_key_list

# fungsi untuk nambahin document lagi ke dalam kumpulan document


def tambahDocument():

    while True:
        input_doc = input("Masukkan document baru (ketik 'done' jika sudah): ")
        input_label = input(
            "Masukkan kategori untuk melabeli document barusan :")
        if input_doc == "done" and input_label == "done":
            break
        else:
            documents_list.append(input_doc.lower())
            docs_key_list.append("d{}".format(len(docs_key_list)+1))
            labels_list.append(input_label.upper())
    return documents_list, labels_list


# Run cell ini jika mau menambahkan document
tambahDocument()

print(type(documents_list))

documents_list

"""metode ini dilakukan dengan menghilangkan stop words sepereti but, not, to, the dan lain lain."""


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')


def preprocess(sentence):
    # Create an instance of the stop word remover
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()

    # Remove the stop words from the sentence
    sentence_without_stopwords = stopword.remove(sentence)

    return sentence_without_stopwords


processed_documents_list = []

for i in range(len(documents_list)):
    tmp = preprocess(documents_list[i])
    processed_documents_list.append(tmp)

print(type(documents_list))

documents_list

processed_documents_list

"""## mamasukkan querry yang ingin dicari

"""

query = input("Masukkan querry : ")

"""## Melakukan Cosine similarity"""


# preprocessing dan vektorisasi documents
vectorizer = TfidfVectorizer()
vectorized_docs = vectorizer.fit_transform(documents_list)


# preprocessing dan vektorisasi querry
query_vector = vectorizer.transform([query])

# Calculate cosine similarity between the query and each document
similarity_scores = cosine_similarity(query_vector, vectorized_docs)

# Print the similarity scores for each document

for doc_idx, score in enumerate(similarity_scores[0]):
    document = documents_list[doc_idx]
    print(
        f"Similarity with d{doc_idx+1}: {score:.5f}\t Document: {document} \t Class : {labels_list[doc_idx]}")

query

processed_documents_list

similarity_scores[0]

labels_list

docs_key_list

"""# menyimpulkan kategori dari querry yang dimasukkan"""

len(docs_key_list)

# dictionary of lists
dict = {'document': docs_key_list, 'term yang mewakili dokumen': processed_documents_list,
        'CLASS': labels_list,  'nilai kesamaan': similarity_scores[0]}

df_dokumen = pd.DataFrame(dict)

dokumen_sorted = df_dokumen.sort_values(by=['nilai kesamaan'], ascending=False)
dokumen_sorted

dokumen_sorted['CLASS'][:-1]

# Use Counter from collections to count unique values in a Python list


if len(labels_list) % 2 == 0:
    counter_class = Counter(dokumen_sorted['CLASS'][:-1])
    keys = counter_class.keys()
    num_values = len(keys)

elif len(labels_list) % 2 != 0:
    counter_class = Counter(dokumen_sorted['CLASS'])
    keys = counter_class.keys()
    num_values = len(keys)

keys

num_values

counter_class

"""# menemukan kategori dari querry yang dimasukkan"""

category_querry = max(counter_class, key=counter_class.get)
category_querry

print("'{qry}' termasuk kedalam CLASS : {cat}".format(
    qry=query, cat=category_querry))

count_vect = CountVectorizer()

Document1 = "tanding sepakbola persebaya kampanye pemilu 2009 tunda"
Document2 = "	partai golkar demokrat tanding kampanye 2009	"
corpus = [Document1, Document2]

X_train_counts = count_vect.fit_transform(corpus)

pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names_out(
), index=['Document 1', 'Document 2'])


# Example texts
text1 = "tanding sepakbola persebaya kampanye pemilu 2009 tunda"
text2 = "partai golkar demokrat tanding kampanye 2009"
text3 = "tanding pertama persema persebaya malang"

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the texts
tfidf_matrix = vectorizer.fit_transform([text1, text2, text3])

# Compute the cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# The cosine similarity matrix
print(cosine_similarities)

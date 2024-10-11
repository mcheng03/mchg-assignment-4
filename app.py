from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from numpy.linalg import svd
import re


nltk.download('stopwords')

app = Flask(__name__)

newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data


def clean_document(doc):
    cleaned = re.sub(r'\W{20,}', ' ', doc)
    return cleaned

documents = [clean_document(doc) for doc in newsgroups.data]

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=4, max_df=0.7, ngram_range=(1, 2))

tfidf_matrix = vectorizer.fit_transform(documents)
tfidf_matrix = tfidf_matrix.toarray()
tfidf_mean = np.mean(tfidf_matrix, axis=0)
tfidf_matrix_centered = tfidf_matrix - tfidf_mean

def perform_svd(matrix, k):
    U, S, Vt = svd(matrix, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    return U_k, S_k, Vt_k

U_k, S_k, Vt_k = perform_svd(tfidf_matrix_centered, 80)

documents_lsa = np.dot(U_k, S_k)

def preprocess_query(query):
    query_tfidf = vectorizer.transform([query]).toarray() 
    query_tfidf_centered = query_tfidf - tfidf_mean
    query_lsa = np.dot(query_tfidf_centered, Vt_k.T)
    return query_lsa.flatten() 

def cosine_similarity(matrix, vector):
    dot_product = np.dot(matrix, vector)
    norm_matrix = np.linalg.norm(matrix, axis=1)
    norm_vector = np.linalg.norm(vector)
    return dot_product / (norm_matrix * norm_vector + 1e-10)

def search_engine(query):
    query_lsa = preprocess_query(query)
    similarities = cosine_similarity(documents_lsa, query_lsa)
    top_indices = similarities.argsort()[-5:][::-1]
    top_documents = [documents[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]
    return top_documents, top_similarities, top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)

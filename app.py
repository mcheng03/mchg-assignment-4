from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from numpy.linalg import svd

nltk.download('stopwords')

app = Flask(__name__)


newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data
stop_words = stopwords.words('english')  

vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
tfidf_matrix = vectorizer.fit_transform(documents)
tfidf_matrix = tfidf_matrix.toarray()


def perform_svd(matrix, k=100):
    U, S, VT = svd(matrix, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    return U_k, S_k, VT_k

k = 100  
U_k, S_k, VT_k = perform_svd(tfidf_matrix, k)

documents_lsa = np.dot(U_k, S_k)

def preprocess_query(query):
    query_tfidf = vectorizer.transform([query]).toarray()
    query_lsa = np.dot(query_tfidf, VT_k.T)
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
    try:
        documents, similarities, indices = search_engine(query)
        return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices})
    except Exception as e:
        app.logger.error(f"Error processing query '{query}': {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
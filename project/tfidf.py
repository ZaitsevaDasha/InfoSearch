import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
morph = MorphAnalyzer()
vectorizer = TfidfVectorizer()
sw = stopwords.words('russian')
import json


def read_corpus(path):
    with open(path, encoding = 'utf-8') as f:
        corpus = f.readlines()
    X = vectorizer.fit_transform(corpus)
    return X    

def get_answers(path):
    with open(path, encoding = 'utf-8') as f:
        text = f.read()
        answers = text.split('\n')
    return answers

def make_vector(query):
    tokenized = nltk.word_tokenize(query)
    lemmatized = []
    for word in tokenized:
        w = word.lower()
        if w.isalpha() and w not in sw:
            nf = morph.normal_forms(w)[0]           
            lemmatized.append(nf)
    query_prep = ' '.join(lemmatized)
    q = [query_prep]
    q_vec = vectorizer.transform(q)
    return q_vec

def count_sim(a, q_vec):
    a = a.reshape(1, -1)
    return cosine_similarity(a, q_vec)[0][0]
        
def make_list_of_sims(q_vec, X):
    X = X.toarray()
    q_vec = q_vec.toarray()
    cos_sims = np.apply_along_axis(count_sim, 1, X, q_vec)
    return cos_sims

def search(X, query, filenames):
    q_vec = make_vector(query)
    cos_sims = cosine_similarity(X, q_vec)
    print(type(cos_sims[0]))
    print(type(cos_sims))
    sims = np.array(cos_sims).flatten()
    indx = np.argsort(sims)
    indx = np.flip(indx)
    s_files = [filenames[i] for i in indx]
    return s_files[:10]

def main(query):
    X = read_corpus('data\\prep_ans.txt')
    filenames = get_answers('data\\answers.txt')
    return search(X, query, filenames)


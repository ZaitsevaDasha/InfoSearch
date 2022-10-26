import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import time

morph = MorphAnalyzer()
count_vectorizer = CountVectorizer(analyzer='word')
sw = stopwords.words('russian')

k = 2
b = 0.75

def read_matrix(path_mat):
    matrix = sparse.load_npz(path_mat)
    return matrix

# получаем тексты ответов на каждый вопрос с наибольшим value
def get_answers(path):
    with open(path, encoding = 'utf-8') as f:
        text = f.read()
        answers = text.split('\n')
    return answers

# создаем вектор запроса
def make_vector(query):
    with open('data\prep_ans.txt', encoding = 'utf-8') as f:
        answers = f.read().split('\n')
    count_vectorizer.fit(answers)
    tokenized = nltk.word_tokenize(query)
    lemmatized = []
    for word in tokenized:
        w = word.lower()
        if w.isalpha() and w not in sw:
            nf = morph.normal_forms(w)[0]           
            lemmatized.append(nf)
    query_prep = ' '.join(lemmatized)
    q = [query_prep]
    q_vec = count_vectorizer.transform(q)
    return q_vec

# считаем близости запроса с каждым текстом корпуса, сортируем и выбираем 10 наиболее похожих
def search(bm, query, answers):
    vec = make_vector(query)
    matrix_sims = np.dot(bm, vec.T)
    sims = matrix_sims.toarray().flatten()
    indx = np.argsort(sims)
    indx = np.flip(indx)
    s_files = [answers[i] for i in indx][:10]
    return s_files

def main(query):
    bm_25 = read_matrix('data\\bm.npz')
    answers = get_answers('data\\answers.txt')
    texts = search(bm_25, query, answers)
    return texts
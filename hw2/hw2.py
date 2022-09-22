import nltk
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from pymystem3 import Mystem
from pymorphy2 import MorphAnalyzer
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
morph = MorphAnalyzer()
import csv
vectorizer = TfidfVectorizer()
sw = stopwords.words('russian')


def read_corpus(path):
    with open('corpus.txt', encoding = 'utf-8') as f:
        corpus = f.readlines()
    X = vectorizer.fit_transform(corpus)
    return X    

def get_filenames(path):
    all_filenames = []
    for root, dirs, files in os.walk(path):
        for name in files:
            all_filenames.append(name)
            file_path = os.path.join(root, name)
    return all_filenames

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
    cos_sims = make_list_of_sims(q_vec, X)
    indx = np.argsort(cos_sims)
    indx = np.flip(indx)
    print(indx)
    print(filenames)
    s_files = [filenames[i] for i in indx]
    return s_files[:10]

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('path', type=str, help='Path to corpus')
    parser.add_argument('path_fr', type=str, help='Path to friends')
    parser.add_argument('query', type=str, help='Query')
    args = parser.parse_args()
    X = read_corpus(args.path)
    filenames = get_filenames(args.path_fr)
    print(search(X, args.query, filenames))

if __name__ == '__main__':
	main()


import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

morph = MorphAnalyzer()
count_vectorizer = CountVectorizer(analyzer='word')
sw = stopwords.words('russian')

k = 2
b = 0.75

# Создаем матрицу bm-25 для всех документов корпуса:
def make_matrix(path):
    with open(path, encoding='utf-8') as f:
        texts = f.readlines()
    tf = count_vectorizer.fit_transform(texts)
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf_vectorizer.fit_transform(texts)
    idf = tfidf_vectorizer.idf_   
    len_d = tf.sum(1)
    avdl = len_d.mean()
    B_1 = (k * (1 - b + b * len_d / avdl))
    for en, (i, j) in enumerate(zip(*tf.nonzero())):
        tf.data[en] = tf.data[en] * idf[j] * (k + 1) / (tf.data[en] + B_1[i])
    bm_25 = tf
    return bm_25

# получаем тексты ответов на каждый вопрос с наибольшим value
def get_answers(path):
    all_answers = []
    with open(path, 'r', encoding = 'utf-8') as f:
        corpus = list(f)[:50000]
    for line in corpus:
        d = json.loads(line)
        answers = d['answers']
        if not answers:
            continue
        ans = max(filter(lambda a: a['author_rating']['value'] != '', answers), key = lambda i : int(i['author_rating']['value']))
        all_answers.append(ans['text'])
    return all_answers

# создаем вектор запроса
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
    q_vec = count_vectorizer.transform(q)
    return q_vec

# считаем близости запроса с каждым текстом корпуса, сортируем и выбираем 10 наиболее похожих
def search(bm, query, answers):
    vec = make_vector(query)
    vals, rows, cols = [], [], []
    for en, (i, j) in enumerate(zip(*bm.nonzero())):
        val = bm.data[en] * vec[0, j]
        vals.append(val)
        rows.append(i)
        cols.append(j)
    matrix_sims = sparse.csr_matrix((vals, (rows, cols)))
    sims = matrix_sims.sum(axis = 1)
    sims = np.array(sims).flatten()
    indx = np.argsort(sims)
    indx = np.flip(indx)
    s_files = [answers[i] for i in indx][:10]
    return s_files

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('path', type=str, help='Path to corpus')
    parser.add_argument('path_ans', type=str, help='Path to answers')
    parser.add_argument('query', type=str, help='Query')
    args = parser.parse_args()
    bm_25 = make_matrix(args.path)
    answers = get_answers(args.path_ans)
    texts = search(bm_25, args.query, answers)
    for text in texts:
        print(text + '\n')

if __name__ == '__main__':
	main()

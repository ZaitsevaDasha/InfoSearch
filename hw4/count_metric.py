from re import A
import numpy as np
import json
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
count_vectorizer = CountVectorizer(analyzer='word')
from transformers import AutoTokenizer, AutoModel 
import torch
from scipy import sparse
import bert_search

count_vectorizer = CountVectorizer(analyzer='word')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

k = 2
b = 0.75

# читаем матрицу
def read_matrix(path):
    my_embeddings = np.load(path)
    return my_embeddings


# считаем близости запросов и ответов корпуса
def count_similarity(vec, matrix):
    cos_sims = cosine_similarity(matrix, vec)
    return cos_sims


# получаем топ-5 результатов для каждого запроса
def get_top5(mat_sims):
    s_mat_sims = mat_sims.argsort(axis = 1)
    indx = np.flip(s_mat_sims, axis = 1)
    s_mat_sims5 = indx[:, :5]
    return s_mat_sims5

# считаем итоговую метрику
def count_metric(s_mat_sims5):
    corr_answers = 0
    for ind, line in enumerate(s_mat_sims5):
        if ind in line:
            corr_answers += 1
    return corr_answers / s_mat_sims5.shape[0]


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('path_json', type=str, help='Path to json')
    args = parser.parse_args()
    bert_matrix_a = np.load('answers.npy')
    bert_matrix_q = np.load('questions.npy')
    sims = count_similarity(bert_matrix_q, bert_matrix_a)
    s_mat_sims5 = get_top5(sims)
    print('Bert metric:{}'.format(count_metric(s_mat_sims5)))

if __name__ == '__main__':
	main()


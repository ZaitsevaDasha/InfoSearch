import json
import torch
import transformers
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel 
from sklearn.metrics.pairwise import cosine_similarity
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru").to(device=device)

# получаем тексты ответов
def get_answers(path):
    with open(path, encoding = 'utf-8') as f:
        text = f.read()
        answers = text.split('\n')
    return answers

# читаем матрицу с векторизированными ответами
def read_matrix(path):
    my_embeddings = np.load(path)
    return my_embeddings

# преобразуем эмбеддинги токенов в эмбеддинг текста
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# создаем вектор запроса
def make_vector(query):
    encoded_input = tokenizer(query, padding=True, truncation=True, max_length=24, return_tensors='pt').to(device=device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

# считаем близость каждого документа с запросом
def count_similarity(vec, matrix):
    cos_sims = cosine_similarity(matrix, vec)
    return cos_sims


# получаем топ-10 наиболее близких ответов к нашему запросу
def search(matrix, query, answers):
    vec = make_vector(query).cpu()
    cos_sims = count_similarity(vec.cpu(), matrix)
    sims = np.array(cos_sims).flatten()
    indx = np.argsort(sims)
    indx = np.flip(indx)
    s_files = [answers[i] for i in indx]
    return s_files[:10]


def main(query):
    answers = get_answers('data\\answers.txt')
    bert_matrix = np.load('data\\answers.npy')
    texts = search(bert_matrix, query, answers)
    return texts
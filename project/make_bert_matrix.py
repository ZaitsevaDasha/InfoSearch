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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_answers(path):
    all_answers = []
    all_questions = []
    with open(path, 'r', encoding = 'utf-8') as f:
        corpus = list(f)[:50000]
    for line in corpus:
        d = json.loads(line)
        question = d['question']
        answers = d['answers']
        if not answers:
            continue
        ans = max(filter(lambda a: a['author_rating']['value'] != '', answers), key = lambda i : int(i['author_rating']['value']))
        all_questions.append(question)
        all_answers.append(ans['text'])
    return all_answers, all_questions


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def make_embeddings(batches):
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru").to(device=device)

    all_s_embeds = []
    for batch in batches:
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=24, return_tensors='pt').to(device=device)
        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        all_s_embeds.append(sentence_embeddings)
    return all_s_embeds

def embs_to_numpy(embeddings):
    f_batch = embeddings[0]
    f_b = [em.cpu().numpy() for em in f_batch]
    matrix = np.array(f_b)
    for batch in embeddings[1:]:
        b = [em.cpu().numpy() for em in batch]
        n_b = np.array(b)
        matrix = np.concatenate((matrix, n_b), axis=0)
    return matrix


def make_matrix(texts):
    f = lambda A, n=500: [A[i:i+n] for i in range(0, len(A), n)]
    batches = f(texts)
    embeddings = make_embeddings(batches)
    bert_matrix = embs_to_numpy(embeddings)
    return bert_matrix


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('path_json', type=str, help='Path to json')
    args = parser.parse_args()
    answers, questions = get_answers(args.path_json)
    m_ans = make_matrix(answers)
    np.save('answers.npy', m_ans)


if __name__ == '__main__':
	main()
import nltk
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from pymystem3 import Mystem
from pymorphy2 import MorphAnalyzer
import numpy as np
import argparse
morph = MorphAnalyzer()
vectorizer = CountVectorizer(analyzer='word')
sw = stopwords.words('russian')


# создание списка предобработанных серий
def preprocessing(path):
    corpus = []
    for root, dirs, files in os.walk(path):
        for name in tqdm(files):
            file_path = os.path.join(root, name)
            with open(file_path, encoding = 'utf-8') as f:
                text = f.read()
                tokenized = nltk.word_tokenize(text)
                lemmatized = []
                for word in tokenized:
                    w = word.lower()
                    if w.isalpha() and w not in sw:
                        nf = morph.normal_forms(w)[0]
                        lemmatized.append(nf)
                doc = ' '.join(lemmatized)
                corpus.append(doc)
    return corpus

# подсчет частоты встречаемости для каждого героя
def count_char_freq(names, matrix_freq):
    freq = 0
    for name in names:
        ind = vectorizer.vocabulary_.get(name)
        freq += matrix_freq[ind]
    return freq

# создание обратного индекса в виде словаря
def make_dict(corpus):
    dic = {}
    prev_words = 0
    for i, doc in enumerate(corpus):
        for ind, word in enumerate(set(doc.split(' '))):
            if word not in dic:
                dic[word] = [prev_words + ind]
            else:
                dic[word].append(prev_words + ind)
        prev_words += len(doc)
    return dic

# создание обратного индекса в виде матрицы
def make_matrix(corpus):
    M = vectorizer.fit_transform(corpus)
    return M

# поиск наиболее и наименее частотного слова, 
# поиск слов, которые встречаются во всех документах
def count_freq_words(M, dic):
    matrix_freq = np.asarray(M.sum(axis=0)).ravel()
    names = vectorizer.get_feature_names()
    most_freq_ind = np.argmax(matrix_freq)
    less_freq_ind = np.argmin(matrix_freq)
    most_freq_word = names[most_freq_ind]
    less_freq_word = names[less_freq_ind]
    print('Самое частое слово: {}'.format(most_freq_word))
    print('Самое редкое слово: {}'.format(less_freq_word))
    freq_words = {key: val for key, val in dic.items() if len(val) == 165}.keys()
    print('Слова, встречающиеся во всех документах: {}'.format(freq_words))
    
# поиск героя, который упоминается чаще всего
def count_freq_chars(M, dic):
    matrix_freq = np.asarray(M.sum(axis=0)).ravel()
    all_chars = [['моника', 'мона'], ['рейчел', 'рейч'], ['чендлер', 'чэндлер', 'чен'], ['фиби', 'фибс'], 
           ['росс'], ['джо', 'джой', 'джоуя']]
    freq_chars = {}
    for char in all_chars:
        freq = count_char_freq(char, matrix_freq)
        freq_chars[char[0]] = freq
    most_freq = max(freq_chars, key=freq_chars.get)
    print('Самый частый герой: {}'.format(most_freq))


def main():
    parser = argparse.ArgumentParser(description='Path to friends')
    parser.add_argument('path', type=str, help='Path to friends')
    args = parser.parse_args()
    print(args.path)
    corpus = preprocessing(args.path)
    dic = make_dict(corpus)
    matrix = make_matrix(corpus)
    count_freq_words(matrix, dic)
    count_freq_chars(matrix, dic)


if __name__ == '__main__':
	main()
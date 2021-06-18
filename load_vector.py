import io
import pickle
import time

import numpy as np


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])

    return data


def load_vectors_v2(file_name):
    word2vector_dic = {}
    with open(file_name, "r") as file:
        word2vector = file.readlines()
        for word in word2vector:
            key = word.split()[0]
            val = [float(v) for v in word.split()[1:]]
            val = np.asarray(val)
            word2vector_dic.update({key: val})
    return word2vector_dic


if __name__ == '__main__':
    print("Begin to load word2vector raw data")
    start = time.time()
    my_words = load_vectors_v2('./word2vector/glove.twitter.27B.200d.txt')
    end = time.time()
    print("Cost", end - start, "s to load!")

    with open('./word2vector/words_vector.pk', "wb") as fp:
        pickle.dump(my_words, fp)

    print("Save word2vector dictionary successfully!")

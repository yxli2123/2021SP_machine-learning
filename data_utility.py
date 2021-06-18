import pandas as pd
import os
import time
import numpy as np
import torch
import pickle


def load_data(data_dir='./train_data_all/'):
    """
    :param data_dir: the directory that contains all file.csv
    :return: X, the sample; y, the label
    """

    # use os.listdir to obtain a list whose element is a file name
    file_list = sorted(os.listdir(data_dir))

    # create 2 huge lists X and y to record comments and stars respectively
    X = []
    y = []

    # traverse all csv files
    for file in file_list:
        # read a csv file
        df = pd.read_csv(os.path.join(data_dir, file))

        # delete a row that contains NAN
        df = df.dropna()

        # remove non-alpha-numeric characters from strings within a dataframe column
        df['CommentsTitle'] = df['CommentsTitle'].str.replace('[^a-zA-Z0-9]', ' ', regex=True)
        df['CommentsContent'] = df['CommentsContent'].str.replace('[^a-zA-Z0-9]', ' ', regex=True)

        for index, row in df.iterrows():
            # if encounter invalid input, skip it
            try:
                # append the score or star to the huge list y; row['CommentsStars'] is an integer
                # !!! some element may be str type, e.g., '5.0', so we first need to convert it into float
                # type, i.e., 5.0 and then convert it into int type, i.e., 5
                string = str(row['CommentsStars']).split()[0]
                y.append(int(float(string)))
                # concatenate 'Comments Title' and 'Comments Content'
                # convert all letters to lowercase
                concatenated_comments = row['CommentsTitle'].lower() + row['CommentsContent'].lower()

                # split sentence into word list and append the list into the huge list X
                concatenated_comments = concatenated_comments.split()
                X.append(concatenated_comments)
            finally:
                pass
    return X, np.asarray(y)


def embedding_v2(sentence: list, words_vector: dict, dim=200):
    """
    :param words_vector: map a word to a tuple (25, )
    :param sentence: a word list, e.g., ['i', 'love', 'ustc', '.']
    :param dim: the dimension of a word vector
    :return: a truncated sentence with its embedding
    """
    # create a list with potential shape (K, )
    sentence_embedding = []

    # embedding word
    for word in sentence:
        # if the word is in the word vector
        if word in words_vector.keys():
            sentence_embedding.append(words_vector[word])
        # else we embed this word by a default zero vector
        else:
            sentence_embedding.append(np.zeros(dim))

    return sentence_embedding


if __name__ == '__main__':
    start = time.time()
    with open("./word2vector/words_vector.pk", "rb") as fp1:
        my_words = pickle.load(fp1)
    end = time.time()
    print(end - start)
    comments, label = load_data(data_dir='./train_data_all')
    comments_embedding = []
    for comment in comments:
        comments_embedding.append(embedding_v2(comment, my_words))

    with open("./word2vector/data.pk", "wb") as fp2:
        pickle.dump((comments_embedding, label), fp2)

    print(len(comments_embedding))
    print("Finished")

import pandas as pd
import os
import torch
import time


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
            # concatenate 'Comments Title' and 'Comments Content'
            # convert all letters to lowercase
            concatenated_comments = row['CommentsTitle'].lower() + row['CommentsContent'].lower()

            # split sentence into word list and append the list into the huge list X
            X.append(concatenated_comments.split())
            # append the score or star to the huge list y; row['CommentsStars'] is an integer
            y.append(row['CommentsStars'])

    return X, y


def embedding(sentence: list, words_vector: dict, length=100):
    """
    :param length: the length of final sentence, probably being truncated or padded
    :param words_vector: map a word to a tuple (300, )
    :param sentence: a word list, e.g., ['i', 'love', 'ustc', '.']
    :return: a truncated sentence with its embedding
    """
    # create a list with potential shape (K, )
    sentence_embedding = []

    # truncate the sentence if it is greater than 'length'
    if len(sentence) > length:
        sentence = sentence[0: length]
    # else pad '.' to a sentence
    else:
        sentence = sentence + ['.' for _ in range(length - len(sentence))]

    # embedding word
    for word in sentence:
        # if the word is in the word vector
        if word in words_vector.keys():
            sentence_embedding.append(words_vector[word])
        # else we embed this word by a default zero vector
        else:
            sentence_embedding.append([0 for _ in range(300)])

    cc = torch.as_tensor(sentence_embedding)
    return cc


if __name__ == '__main__':
    start = time.time()
    my_words = torch.load('./words_vector.pt')
    end = time.time()
    print(end - start)
    comments, label = load_data()
    comments_embedding = torch.zeros((len(comments), 100, 300))
    for i, comment in enumerate(comments):
        comments_embedding[i] = embedding(comment, my_words)
    torch.save(comments_embedding, './comments_embedding.pt')
    print("Finished")

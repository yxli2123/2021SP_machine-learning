import time
import numpy as np
import pickle
import random


def classifier(sentence: list, probability_parameter: list, prior_probability: list):
    """
    :param sentence: a word list that contains embedded word, each element in this list is a ndarray
                     with shape ( , c) where c is the word vector dimension
    :param probability_parameter: probability parameters (e.g. mean vector, inverse and determinant of
                                  covariance matrix) given a class. Every element in this list is a tuple,
                                  i.e., (mean, cov, inv, det)
    :param prior_probability: each element in this list is the prior probability of each class
    :return: the class of the input sentence
    """
    score = []
    num_class = len(probability_parameter)  # the number of class, i.e, 5
    for i in range(num_class):

        # fetch parameter from the input
        mean, _, inv, det = probability_parameter[i]
        pp = prior_probability[i]

        # calculate the posterior probability of each word
        word_p = [probability_given_class(word, mean, inv, det) + np.log(pp) for word in sentence]

        # use the average posterior probability as the score of this sentence
        if not word_p:  # in case that word_p is an empty list and it doesn't have mean
            score.append(0)
        else:
            score.append(np.mean(word_p))

    # choose the class that has the greatest score
    c = np.argmax(score) + 1

    return c


def probability_given_class(x, mean, inv, det):
    """
    :param    x: a piece of test data, i.e., a word (embedded into word vector)
    :param mean: the mean vector of class i
    :param  inv: the inverse of covariance matrix of class i
    :param  det: the determinant of covariance matrix of class i
    :return: the probability of x given class i
    """

    # regulate the shape of x
    c = x.size
    x = x.reshape(c, 1)

    # calculate the numerator of the Gaussian expression
    log_exp = (-0.5 * (x - mean).T.dot(inv).dot(x - mean)).item()

    # calculate the denominator of the Gaussian expression
    log_den = c * np.log(2*np.pi) / 2 + np.log(det) / 2

    return log_exp - log_den


def get_Gaussian_parameter(X: np.ndarray):
    """
    :param X: Training data that belongs to class i, with shape (N, C),
              where N is the number of samples and C is the feature length, aka, length of word vector
    :return:  mean vector, covariance matrix, and its inverse and determinant
    """

    # calculate the mean of each columns, axis=0 means operate on the column
    mean_vector = np.mean(X, axis=0)
    # after the operation above, mean_vector has shape (, C),
    # and we expand it into shape (C, 1)
    mean_vector = np.expand_dims(mean_vector, axis=1)

    # calculate the covariance matrix
    covariance_matrix = np.cov(X.T)

    # calculate its inverse and determinant
    inverse = np.linalg.inv(covariance_matrix)
    determinant = np.linalg.det(covariance_matrix)

    return mean_vector, covariance_matrix, inverse, determinant


def train(comments: list, labels: np.ndarray, num_class=5):
    """
    :param comments:  a list, where each element is a sentence that is also a list containing word vector
    :param labels:    the class, {1, 2, 3, 4, 5}
    :param num_class: the number of class, here we set 5
    :return:          the probability parameter
    """

    probability_parameter = []
    prior_probability = []
    num_set = [len(np.where(labels == i+1)[0]) for i in range(num_class)]
    min_num = min(num_set)
    print(num_set)

    # train probability parameter for each class
    for i in range(num_class):
        # fetch the index of each comment that belongs to class i
        star_index = np.where(labels == i+1)
        star_index = (star_index[0]).tolist()
        star_index = random.sample(star_index, min_num)
        # create a list that contains words from all of comment belonging to class i
        comments_i = []

        # concatenate all the sentence that belongs to class i
        for j in star_index:
            comments_i += comments[j]

        # transfer the list into numpy array, with shape (N, C)
        # N is the number of words from sentence that belongs to class i
        # C is the dimension of word vector
        comments_i = np.asarray(comments_i)
        probability_parameter.append(get_Gaussian_parameter(comments_i))
        prior_probability.append(num_set[i] / len(comments))

    return probability_parameter, prior_probability


def infer(comments: list, labels: np.ndarray, probability_parameter, prior_probability):
    """
    :param comments:
    :param labels:
    :param probability_parameter:
    :param prior_probability:
    :return:
    """

    result = []
    num_class = len(prior_probability)
    for i in range(num_class):
        start = time.time()
        c = []
        star_index = np.where(labels == i + 1)
        for j in star_index[0]:
            c.append(classifier(comments[j], probability_parameter, prior_probability))
        result.append(c)
        end = time.time()
        print("class", i + 1, "cost", end - start, "s")
    return result


def rater(result):
    """
    :param result:
    :return:
    """
    """
    Precision = correct number of class i / correct number of class i + number class i in other classes
    Recall = correct number of class i / total number of class i
    F1-score = 2 * Precision * Recall / (Precision + Recall)
    """
    F1_score = []
    num_class = len(result)
    for i in range(num_class):
        true_positive = result[i].count(i + 1)
        false_negative = len(result[i]) - true_positive
        false_positive = sum([result[j].count(i + 1) for j in range(num_class)]) - true_positive
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        F1_score.append(2 * precision * recall / (precision + recall))
        print("\nClass ", i+1)
        print("tp:", true_positive)
        print("fn:", false_negative)
        print("fp:", false_positive)
        print("precision:", precision)
        print("recall:", recall)
        print("F1-score:", 2 * precision * recall / (precision + recall))

    return sum(F1_score) / len(F1_score)


if __name__ == '__main__':
    with open('./word2vector/data.pk', "rb") as fp:
        comments, labels = pickle.load(fp)
        fp.close()

    print("Start to train!")
    prob_para, prior_prob = train(comments, labels, 5)

    with open('./parameter.pk', "wb") as fp:
        pickle.dump((prob_para, prior_prob), fp)
        fp.close()

    print("Start to infer!")
    ans = infer(comments, labels, prob_para, prior_prob)

    print("Start to evaluate!")
    Macro_F1_Score = rater(ans)
    print("\nMacro F1-Score: ", Macro_F1_Score)

import io
import torch


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])

    return data


if __name__ == '__main__':

    my_words = load_vectors('/Users/yxli/Downloads/wiki-news-300d-1M.vec')
    for key in my_words:
        my_words[key] = [val for val in my_words[key]]
    torch.save(my_words, './words_vector.pt')

    # my_words = torch.load('./words_vector.pt')
    # print(my_words)

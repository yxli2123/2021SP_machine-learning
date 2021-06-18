# 2021SP Machine Learning

## 2021.06.18 Gaussian Bayes

新增Bayes.py，内含train、infer、rate函数

修改data_utility.py，取消原来的对齐embedding，改成变长的embedding，方便后面的Bayes

修改load_vector.py，读取更轻量的word2vector，下载方式

```
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
```

## 2021.05.21 have finished text embedding
one sentence --> (100, 300) tensor

100 is truncated number if the words in a sentence is grater than 100 or is padded if the words in a sentencex is less than 100

300 is the dimesion of the words vector, downloaded from https://fasttext.cc/docs/en/english-vectors.html

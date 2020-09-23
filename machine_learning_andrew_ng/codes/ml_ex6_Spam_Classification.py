import csv
import re
import nltk, nltk.stem.porter  # 可用的英文分词算法

import numpy as np
from scipy.io import loadmat

#   1. Lower-casing: 把整封邮件转化为小写。
#   2. Stripping HTML: 移除所有HTML标签，只保留内容。
#   3. Normalizing URLs: 将所有的URL替换为字符串 “httpaddr”.
#   4. Normalizing Email Addresses: 所有的地址替换为 “emailaddr”
#   5. Normalizing Dollars: 所有dollar符号($)替换为“dollar”.
#   6. Normalizing Numbers: 所有数字替换为“number”
#   7. Word Stemming(词干提取): 将所有单词还原为词源。例如，“discount”, “discounts”, “discounted” and “discounting”都替换为“discount”。
#   8. Removal of non-words: 移除所有非文字类型，所有的空格(tabs, newlines, spaces)调整为一个空格.
from sklearn import svm

path1 = '../files/emailSample1.txt'
path2 = '../files/vocab.txt'
path3 = '../files/spamTrain.mat'
path4 = '../files/spamTest.mat'


# read csv file and save, para:file name, return x, y
def read_data(filename):
    with open(filename, 'r') as f:
        email = f.read()
    return email


# read vocab
def read_vocab(filename):
    vocab_dict = {}  # a dict to save index:value
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            index = int(row[0].replace("\t", " ").split()[0])
            value = row[0].replace("\t", " ").split()[1]
            vocab_dict[index] = value
    return vocab_dict


def pre_process(content):
    content = re.sub('<[^<>]>', ' ', content)  # 去掉html
    content = re.sub('(http|https)://[^\s]*', 'httpaddr', content)  # Normalizing URLs
    content = re.sub('[^\s]+@[^\s]+', 'emailaddr', content)  # Normalizing email
    content = re.sub('[\$]+', 'number', content)  # Look for one or more characters between 0-9
    content = re.sub('[\d]+', 'dollar', content)
    return content


def email_to_list(mail):
    stemmer = nltk.stem.porter.PorterStemmer()
    # 将邮件分割为单个单词，re.split() 可以设置多种分隔符
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', mail)

    # 遍历每个分割出来的内容
    list_content = []
    for token in tokens:
        # 删除任何非字母数字的字符
        token = re.sub('[^a-zA-Z0-9]', '', token)
        # Use the Porter stemmer to 提取词根
        stemmed = stemmer.stem(token)
        # 去除空字符串‘’，里面不含任何字符
        if not len(token):
            continue
        list_content.append(stemmed)
    return list_content


def email_to_index(token_list, vocabs):
    token = list(token_list)
    index = [i for i in vocabs if vocabs[i] in token]
    return index


def email_to_feature_vector(v_indices, v_list):
    # 将email转化为词向量，n是vocab的长度。存在单词的相应位置的值置为1，其余为0
    vector = np.zeros(len(v_list))  # init vector
    for i in v_indices:
        vector[i] = 1
    return vector


def train_model():  # 读取已经训提取好的特征向量以及相应的标签。分训练集和测试集
    # Training set
    data = loadmat(path3)
    X, y = data['X'], data['y']
    # Test set
    data1 = loadmat(path4)
    X_t, y_t = data1['Xtest'], data1['ytest']
    clf = svm.SVC(C=0.1, kernel='linear')
    clf.fit(X, y.ravel())
    pre_train = clf.score(X, y.ravel())
    pre_test = clf.score(X_t, y_t.ravel())
    print(pre_train, pre_test)


if __name__ == '__main__':
    vocab_list = read_vocab(path2)  # vocab list
    mail_content = read_data(path1).lower()  # lower case
    mail_content_pre = pre_process(mail_content)  # pre work
    final_list = email_to_list(mail_content_pre)
    indices_index = email_to_index(final_list, vocab_list)
    fea_vector = email_to_feature_vector(indices_index, vocab_list)
    train_model()

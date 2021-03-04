import string
import re
import nltk
from nltk.stem import PorterStemmer
porter = PorterStemmer()


def ProcessEmail(file_contents):
    # initiate values
    word_indices = []
    # load VocabList
    vocab_list = getVocabList()
    # 列表->string
    word_contents = ''.join(file_contents)
    # Find the header and delete it
    index = word_contents.find('\n\n')
    word_contents = word_contents[index+1:len(word_contents)-1]

    #  ----preprocess--------

    word_contents = word_contents.lower() #转小写
    word_contents = re.sub('[0-9]+', 'number', word_contents)  #使用正则表达式将数字替换为number
    word_contents = re.sub('[><]', ' ', word_contents)  #stripping HTML
    word_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', word_contents)  #http -> "httpaddr"
    word_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', word_contents)  #email address
    word_contents = re.sub('[$]+', 'dollar', word_contents)
    for i in string.punctuation:
        word_contents = word_contents.replace(i, "")
    word_contents = re.sub('[\s]+', ' ', word_contents)


    #  -----process---------
    for word in word_contents.split():
        word = re.sub('[^0-9a-zA-Z]', '', word)
        word = porter.stem(word)
        if len(word) < 1:
            continue
        for num in range(len(vocab_list)):  # list's index starts from 0
            if re.search(word, str(vocab_list[num])):
                word_indices.append(num)
                break
    return word_indices


def getVocabList():
    with open('../vocab.txt', 'r') as vocab:
        vocab_list = []
        line = vocab.readline()
        while line:
            a = line.split()
            b = a[1:2]
            vocab_list.append(b)
            line = vocab.readline()
    vocab.close()
    # print(type(vocab_list))
    return vocab_list






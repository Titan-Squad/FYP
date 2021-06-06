import json

import nltk
import string
import os
import io
import sys
english_words = set(nltk.corpus.words.words())
sys.setrecursionlimit(10**7)

def PreprocessForEnglishWords(InputText):
    UniqueEnglishWordList = []
    tokenized_sent = [list(map(str.lower, nltk.word_tokenize(sent)))
                      for sent in nltk.sent_tokenize(InputText)]
    for sent in tokenized_sent:
        for word in sent:
            if word not in UniqueEnglishWordList and word in english_words:
                UniqueEnglishWordList.append(word)
    return UniqueEnglishWordList

if __name__ == '__main__':
    if os.path.isfile('combined.txt'):
        with io.open('combined.txt', encoding='utf8') as fin:
            text = (fin.read())
    UniqueEnglishWordList = PreprocessForEnglishWords(text)
    a_file = open('english_words_in_corpus.txt', 'w', encoding='utf-8', errors='ignore')
    json.dump(UniqueEnglishWordList,a_file)
    a_file.close()

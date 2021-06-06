# -*- coding: utf-8 -*-
import json
import os
import io
import string
import sys
from math import sqrt

import nltk
import re

sys.setrecursionlimit(10 ** 6)
# special characters
special_chars = string.punctuation
PermutationList = []


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def Preprocess(InputText):
    UniqueWordListSinhala = []
    tokenized_sent = [list(map(str.lower, nltk.word_tokenize(sent)))
                      for sent in nltk.sent_tokenize(text)]
    for sent in tokenized_sent:
        for word in sent:
            if word not in UniqueWordListSinhala and not word.isnumeric() and word not in special_chars:
                UniqueWordListSinhala.append(word)
    return UniqueWordListSinhala


def GeneratePermutationsUsingEditDistance(word):
    UniqueWordList = []
    with open('UniqueWords.txt', 'r', encoding='utf-8', errors='ignore') as file1:
        UniqueWordList = json.load(file1)
    for uniqueword in UniqueWordList:
        if not hasNumbers(uniqueword):
            distance = nltk.edit_distance(uniqueword, word)
            if distance <= sqrt(len(word)):
                PermutationList.append(uniqueword)


if __name__ == '__main__':
    if os.path.isfile('adaderana.txt'):
        with io.open('adaderana.txt', encoding='utf8') as fin:
            text = fin.read()
    UniqueWordListSinhala = Preprocess(text)
    for word in UniqueWordListSinhala:
        PermutationList = []
        GeneratePermutationsUsingEditDistance(word)
        print(PermutationList)
# Generating a permutation list using replacing letters
import os
import io
import nltk
import string
import json
import sys
import nltk
from nltk.util import ngrams
from nltk.lm import NgramCounter
from itertools import chain
from math import sqrt
import re
import csv

sys.setrecursionlimit(10 ** 9)
# special characters
special_chars = string.punctuation
PermutationList = []
PermutationDic = {'a': ['a', 'A']}
UniqueWordListInCorpus = []
SimilarLetterGroup = [{'\u0DD0', '\u0DD1'},  # ැ , ෑ replacement
                      {'\u0DD2', '\u0DD3'},  # ි ‍‍, ී replacement
                      {'\u0DD4', '\u0DD6'},  # ු , ූ replacement
                      {'\u0DD9', '\u0DDA'},  # ෙ, ේ replacement
                      {'\u0DDC', '\u0DDD'},  # ො , ෝ  replacement
                      {'\u0DD8', '\u0DF2'},  # ෘ , ෲ replacement
                      {'\u0DD0', '\u0DD9'},  # ැ , ෙ replacement
                      {'\u0DD0','\u0DCF'},
                      {'අ', 'ආ', 'ඇ', 'ඈ'},
                      {'ඉ', 'ඊ', 'යි'},
                      {'උ', 'ඌ'},
                      {'එ', 'ඇ'},
                      {'එ', 'ඒ'},
                      {'ඔ', 'ඕ'},
                      {'ක', 'ඛ'},
                      {'ග', 'ඝ', 'ඟ', 'ජ'},
                      {'ච', 'ඡ'},
                      {'ජ', 'ඣ'},
                      {'ඤ', 'ඥ'},
                      {'ට', 'ඨ'},
                      {'ඩ', 'ඪ', 'ඬ', 'ද', 'ධ', 'ඳ'},
                      {'ත', 'ථ'},
                      {'න', 'ණ'},
                      {'ප', 'ඵ'},
                      {'බ', 'භ', 'ඹ'},
                      {'ල', 'ළ'},
                      {'ස', 'ශ', 'ෂ'}
                      ]
prepo_singhlish_word = []
prepo_alt_word = []
prepo_sin_word = []


def findlen(word):
    counter = 0
    for letter in word:
        if u'\u0dc7' <= letter <= u'\u0dff':
            continue
        else:
            counter += 1
    return counter


def Preprocess(InputText):
    UniqueWordList = []
    tokenized_sent = [list(map(str.lower, nltk.word_tokenize(sent)))
                      for sent in nltk.sent_tokenize(text)]
    for sent in tokenized_sent:
        for word in sent:
            if word not in UniqueWordList and not word.isnumeric() and word not in special_chars:
                UniqueWordList.append(word);
    return UniqueWordList


def oneDArray(x):
    return list(chain(*x))


def GeneratePermutationsByReplacing(word):
    if 'න්' in word:
        newWord = word.replace('න්', 'ං')
        if newWord not in PermutationList:
            PermutationList.append(newWord)
    LetterList = list(word)
    OneDSimilar = oneDArray(SimilarLetterGroup)
    for i in range(len(LetterList)):
        if LetterList[i] in OneDSimilar:
            for similar_l in SimilarLetterGroup:
                if LetterList[i] in similar_l:
                    for l in similar_l:
                        LetterList[i] = l
                        new_word = "".join(LetterList)
                        if new_word not in PermutationList:
                            PermutationList.append(new_word)
                            GeneratePermutationsByReplacing(new_word)


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def GeneratePermutationsUsingEditDistance(word, uniqueWordListInCorpus):
    for uniqueword in uniqueWordListInCorpus:
        if not hasNumbers(uniqueword):
            distance = nltk.edit_distance(uniqueword, word)
            if distance < sqrt(findlen(word)) and findlen(word) > 3:
                PermutationList.append(uniqueword)


def getPermutationsList(word, uniqueWordListInCorpus, sinhala_prepositions):
    PermutationList.clear()
    PermutationList.append(word)
    if word not in sinhala_prepositions:
        GeneratePermutationsByReplacing(word)
        GeneratePermutationsUsingEditDistance(word, uniqueWordListInCorpus)
    # print(PermutationList)
    return PermutationList


def loadPrepositions():
    prepositions_file = open('sinhala_preposition.csv', 'r', encoding="utf-8-sig")
    with prepositions_file:
        dataReader = csv.DictReader(prepositions_file)
        for i, row in enumerate(dataReader):
            prepo_sin_word.append(row['sin'].lower())


if __name__ == '__main__':
    if os.path.isfile('adaderana.txt'):
        with io.open('adaderana.txt', encoding='utf8') as fin:
            text = fin.read()
    with open('UniqueWords.txt', 'r', encoding='utf-8', errors='ignore') as file1:
        UniqueWordListInCorpus = json.load(file1)
        file1.close()
    loadPrepositions()
    UniqueWordListInTest = Preprocess(text)
    for word in UniqueWordListInTest:
        PermutationList = []
        if word not in UniqueWordListInCorpus:
            # GeneratePermutationsByReplacing(word)
            GeneratePermutationsUsingEditDistance(word, UniqueWordListInCorpus)
            PermutationListCopy = PermutationList.copy()
            PermutationDic[word] = PermutationListCopy
        else:
            PermutationDic[word] = word
        print(PermutationDic)

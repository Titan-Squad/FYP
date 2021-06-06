# -*- coding: utf-8 -*-
import os
import io
import json
import sys
import nltk
from itertools import chain
from string import punctuation
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

sys.setrecursionlimit(10 ** 8)
TrigramN = 4
BigramN = 3
ThreeSyllableChunksList = {}
TwoSyllableChunksList = {}
UniqueWordList = []
WordList = []
english_words = set(nltk.corpus.words.words())
sinhala_alphabets = [chr(alpha) for alpha in range(3456, 3583)]

def DivideTokenIntoNSyllableChunks(token, n):
    chunks = [token[i:i + n] for i in range(0, len(token))]
    return chunks


def GetTrigramCount(word):
    a_file = open('threeSyllable.txt', 'r', encoding='utf-8', errors='ignore')
    x = a_file.read()
    ThreeSyllableChunksList = json.loads(x)


def GetBigramCount(word):
    a_file = open('twoSyllable.txt', 'r', encoding='utf-8', errors='ignore')
    x = a_file.read()
    y = json.loads(x)
    if word in y:
        return y[word]
    else:
        return 0


def GenarateThreeSyllableChunks(tokenized_sent):
    for sent in tokenized_sent:
        for token in sent:
            chunks = DivideTokenIntoNSyllableChunks(token, TrigramN)
            for chunk in chunks:
                if chunk in ThreeSyllableChunksList:
                    ThreeSyllableChunksList[chunk] += 1
                else:
                    ThreeSyllableChunksList[chunk] = 1


def GenarateTwoSyllableChunks(tokenized_sent):
    for sent in tokenized_sent:
        for token in sent:
            chunks = DivideTokenIntoNSyllableChunks(token, BigramN)
            for chunk in chunks:
                if chunk in TwoSyllableChunksList:
                    TwoSyllableChunksList[chunk] += 1
                else:
                    TwoSyllableChunksList[chunk] = 1

def is_sinhala(x):
    return x in sinhala_alphabets


def GenarateUniqueWordList(tokenized_sent):
    for sent in tokenized_sent:
        for word in sent:
            word = ''.join((x for x in word if not x.isdigit() and is_sinhala(x)))
            if word not in UniqueWordList and word not in english_words:
                UniqueWordList.append(word)

def GenarateWordList(tokenized_sent):
    for sent in tokenized_sent:
        for word in sent:
            WordList.append(word)

def oneDArray(x):
    return list(chain(*x))

def trainNGramModelForWords():
    newsListOne = []
    text = ''
    with open("combined.txt", 'r', encoding='utf-8', errors='ignore') as outfile:
        newslist = json.load(outfile)
    for news in newslist:
        newsListOne.extend(news)
    text = ' '.join([str(elem) for elem in newsListOne])
    tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                      for sent in sent_tokenize(text)]
    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

    model = MLE(n)  # Lets train a 3-grams maximum likelihood estimation model.
    model.fit(train_data, padded_sents)
    return model

def genarateNGramModel():
    newsListOne = []
    text = ''
    with open("combined.txt", 'r', encoding='utf-8', errors='ignore') as outfile:
        newslist = json.load(outfile)
    for news in newslist:
        newsListOne.extend(news)
    text = ' '.join([str(elem) for elem in newsListOne])
    tokenized_text = [list(map(str.lower, nltk.word_tokenize(sent)))
                      for sent in nltk.sent_tokenize(text)]
    print("Text is tokenized..")
    print("Generating Two syllable Chunks..")
    GenarateTwoSyllableChunks(tokenized_text)
    print("Generating Three syllable Chunks..")
    GenarateThreeSyllableChunks(tokenized_text)
    print("Generating Unique word list..")
    GenarateUniqueWordList(tokenized_text)
    print("Generating Tokenized text..")
    GenarateWordList(tokenized_text)
    print("Creating files..")
    with open('WordsList.txt', 'w', encoding='utf-8', errors='ignore') as file1:
        json.dump(WordList,file1)
    with open('UniqueWords.txt', 'w', encoding='utf-8', errors='ignore') as file2:
        json.dump(UniqueWordList,file2)
    a_file = open('twoSyllable.txt', 'w', encoding='utf-8', errors='ignore')
    json.dump(TwoSyllableChunksList, a_file)
    a_file.close()
    a_file = open('threeSyllable.txt', 'w', encoding='utf-8', errors='ignore')
    json.dump(ThreeSyllableChunksList,a_file)
    a_file.close()
    print("Generating Tokenized..")

if __name__ == '__main__':
    genarateNGramModel()
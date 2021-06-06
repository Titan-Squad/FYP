# this used a pre created
# -*- coding: utf-8 -*-
import json
import os
import io
import nltk
import string
import sys
import nltk
from nltk.util import ngrams
from nltk.lm import NgramCounter
from itertools import chain

sys.setrecursionlimit(10 ** 8)
TrigramN = 4
BigramN = 3
# special characters
special_chars = string.punctuation

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
                      {'අ', 'ආ', 'ඇ', 'ඈ'},
                      {'ඉ', 'ඊ', 'යි'},
                      {'උ', 'ඌ'},
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
                        if new_word not in PermutationList and new_word in UniqueWordListInCorpus:
                            PermutationList.append(new_word)
                            GeneratePermutationsByReplacing(new_word)


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def GeneratePermutationsUsingEditDistance(word):
    UniqueWordList = []
    with open('UniqueWords.txt', 'r', encoding='utf-8', errors='ignore') as file1:
        UniqueWordList = json.load(file1)
    for uniqueword in UniqueWordList:
        if not hasNumbers(uniqueword):
            distance = nltk.edit_distance(uniqueword, word)
            if len(word) > 2 and distance <= sqrt(len(word)):
                PermutationList.append(uniqueword)


def GetTrigramCount(word, Trigram_counter_model):
    if word in Trigram_counter_model:
        return Trigram_counter_model[word]
    else:
        return 0


def GetBigramCount(word, Bigram_counter_model):
    if word in Bigram_counter_model:
        return Bigram_counter_model[word]
    else:
        return 0


def DivideTokenIntoNSyllableChunks(token, n):
    chunks = [token[i:i + n] for i in range(0, len(token))]
    return chunks


def Preprocess(InputText):
    UniqueWordList = []
    tokenized_sent = [list(map(str.lower, nltk.word_tokenize(sent)))
                      for sent in nltk.sent_tokenize(InputText)]
    for sent in tokenized_sent:
        for word in sent:
            if word not in UniqueWordList and not word.isnumeric() and word not in special_chars:
                UniqueWordList.append(word)
    return UniqueWordList


def oneDArray(x):
    return list(chain(*x))


def SetUpUnigramModel():
    newsListOne = []
    with open("combined.txt", 'r', encoding='utf-8', errors='ignore') as outfile:
        newslist = json.load(outfile)
    for news in newslist:
        newsListOne.extend(news)
    text = ' '.join([str(elem) for elem in newsListOne])
    tokenized_text = [list(map(str.lower, nltk.word_tokenize(sent))) for sent in nltk.sent_tokenize(text)]
    text_unigrams = [ngrams(sent, 1) for sent in tokenized_text]
    unigram_counter_model = NgramCounter(text_unigrams)
    return unigram_counter_model


def selectBestSuggestionLetterBased(unigram_model, Bigram_counter_model, Trigram_counter_model, permutationList, word):
    highestUnigramFrequency = 0
    bestWord = word
    bestWordsList = []
    bestWordsList.append(word)
    if len(permutationList) > 1:
        # Unigram count
        for w in permutationList:
            wordUnigramFrequency = unigram_model[w]
            if (wordUnigramFrequency > highestUnigramFrequency):
                # highestUnigramFrequency = wordUnigramFrequency
                bestWord = w
                bestWordsList.append(w)
        # trigram count
        if bestWord == word:
            HighestTrigramScore = 0
            for w in PermutationList:
                ThreeSyllableChunks = []
                if len(w) > TrigramN:
                    ThreeSyllableChunks = DivideTokenIntoNSyllableChunks(w, TrigramN)
                    WordTrigramScore = 0
                    for ThreeSyllableChunk in ThreeSyllableChunks:
                        WordTrigramScore += GetTrigramCount(ThreeSyllableChunk, Trigram_counter_model)
                    if WordTrigramScore > HighestTrigramScore:
                        # HighestTrigramScore = WordTrigramScore
                        bestWord = w
                        bestWordsList.append(w)
        # bigram count
        if bestWord == word:
            HighestBigramScore = 0
            for w in PermutationList:
                TwoSyllableChunks = []
                if len(w) > BigramN:
                    TwoSyllableChunks = DivideTokenIntoNSyllableChunks(w, BigramN)
                    WordBigramScore = 0
                    for TwoSyllableChunk in TwoSyllableChunks:
                        WordBigramScore += GetBigramCount(TwoSyllableChunk, Bigram_counter_model)
                    if WordBigramScore > HighestBigramScore:
                        # HighestBigramScore = WordBigramScore
                        bestWord = w
                        bestWordsList.append(w)
        return bestWordsList
    else:
        return bestWordsList


def getBestSuggestionLetterBased(unigram_model, Bigram_counter_model, Trigram_counter_model, permutationList):
    bestWordsSet = {}
    for permutation in permutationList:
        bestWordsSet[permutation] = selectBestSuggestionLetterBased(unigram_model, Bigram_counter_model,
                                                                    Trigram_counter_model, permutationList[permutation],
                                                                    permutation)
    return bestWordsSet


def GenarateThreeSyllableChunks():
    a_file = open('threeSyllable.txt', 'r', encoding='utf-8', errors='ignore')
    x = a_file.read()
    return json.loads(x)


def GenarateTwoSyllableChunks():
    a_file = open('threeSyllable.txt', 'r', encoding='utf-8', errors='ignore')
    x = a_file.read()
    return json.loads(x)


if __name__ == '__main__':
    if os.path.isfile('adaderana.txt'):
        with io.open('adaderana.txt', encoding='utf8') as fin:
            text = fin.read()
    with open('UniqueWords.txt', 'r', encoding='utf-8', errors='ignore') as file1:
        UniqueWordListInCorpus = json.load(file1)
        file1.close()

    UniqueWordList = Preprocess(text)
    Unigram_counter_model = SetUpUnigramModel()
    Bigram_counter_model = GenarateThreeSyllableChunks()
    Trigram_counter_model = GenarateTwoSyllableChunks()
    for word in UniqueWordList:
        PermutationList = []
        if word not in UniqueWordListInCorpus:
            PermutationList.append(word)
            GeneratePermutationsByReplacing(word)
            GeneratePermutationsUsingEditDistance(word)
        else:
            PermutationList.append(word)
        print(PermutationList)
        BestSuggestion = selectBestSuggestionLetterBased(Unigram_counter_model, Bigram_counter_model,
                                                         Trigram_counter_model, PermutationList, word)
        print(BestSuggestion)
        text = text.replace(word, BestSuggestion[0])
    print(text)
    with open('myfile.txt', 'w', encoding='utf-8', errors='ignore') as f_out:
        f_out.write(text)

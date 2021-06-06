#best word suggestion using Trigram m
import io
import json
import os
from string import punctuation
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
#Generating a permutation list using replacing letters
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

sys.setrecursionlimit(10**9)
# special characters
special_chars = string.punctuation
PermutationList = []
PermutationDic = {'a':['a','A']}
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
                        if new_word not in PermutationList:
                            PermutationList.append(new_word)
                            GeneratePermutationsByReplacing(new_word)

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def GeneratePermutationsUsingEditDistance(word):
    for uniqueword in UniqueWordListInCorpus:
        if not hasNumbers(uniqueword):
            distance = nltk.edit_distance(uniqueword, word)
            if len(word) > 2 and distance <= sqrt(len(word)):
                PermutationList.append(uniqueword)

def TrainNGramModelForWords():
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


def selectBestSuggestionContextBasedBigram(word1, word, ngramModel, permulationList):
    bestword = word
    highestProbablity = 0
    if len(permulationList[word]) > 1:
        for perWord in permulationList[word]:
            probability = ngramModel.score(perWord, [word1])
            if (probability > highestProbablity):
                highestProbablity = probability
                bestword = perWord
        return bestword
    else:
        return bestword

def selectBestSuggestionContextBasedUniigram(word, ngramModel, permulationList):
    bestword = word
    highestProbablity = 0
    if len(permulationList[word]) > 1:
        for perWord in permulationList[word]:
            probability = ngramModel.score(perWord)
            if (probability > highestProbablity):
                highestProbablity = probability
                bestword = perWord
        return bestword
    else:
        return bestword

def Padsent():
    newsListOne = []
    text = ''
    if os.path.isfile('adaderana.txt'):
        with io.open('adaderana.txt', encoding='utf8') as fin:
            text = fin.read()
    tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                      for sent in sent_tokenize(text)]
    opensent = ['<s>','<s>']
    closeSent = ['</s>','</s>']
    padded_sent = tokenized_text.copy()
    for i, sent in enumerate(padded_sent):
        padded_sent[i] = opensent + sent + closeSent

    return padded_sent

def genarateBestWordsList(pad_sents,ngramModel,permutationList):
    bestWordList = {}
    for sent in pad_sents:
        for i, word in enumerate(sent[2:-2]):
            if word not in punctuation and not word.isnumeric():
                bestWordList[word] = selectBestSuggestionContextBasedBigram(sent[i - 1], word, ngramModel, permutationList)
                if bestWordList[word] == word:
                    bestWordList[word] = selectBestSuggestionContextBasedUniigram(word, ngramModel, permutationList)
                print(bestWordList)
            else:
                bestWordList[word] = word
    return bestWordList


if __name__ == '__main__':
    if os.path.isfile('adaderana.txt'):
        with io.open('adaderana.txt', encoding='utf8') as fin:
            text = fin.read()
    with open('UniqueWords.txt', 'r', encoding='utf-8', errors='ignore') as file1:
        UniqueWordListInCorpus = json.load(file1)
        file1.close()

    UniqueWordList = Preprocess(text)
    TrigramModel = TrainNGramModelForWords()
    pad_sent = Padsent()
    for word in UniqueWordList:
        PermutationList = []
        if word not in UniqueWordListInCorpus:
            GeneratePermutationsByReplacing(word)
            #GeneratePermutationsUsingEditDistance(word)
            PermutationListCopy = PermutationList.copy()
            PermutationDic[word] = PermutationListCopy
        else:
            PermutationDic[word] = word
        genarateBestWordsList(pad_sent, TrigramModel,PermutationList)

    #bestWordList = GenarateBestWordsList(pad_sent,TrigramModel)
    #print(bestWordList)
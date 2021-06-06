#Generating a permutation list using replacing letters
import os
import io
import nltk
import string
import sys
import nltk
from nltk.util import ngrams
from nltk.lm import NgramCounter
from itertools import chain
sys.setrecursionlimit(10**9)
# special characters
special_chars = string.punctuation
PermutationList = []
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

if __name__ == '__main__':
    if os.path.isfile('adaderana.txt'):
        with io.open('adaderana.txt', encoding='utf8') as fin:
            text = fin.read()
    UniqueWordList = Preprocess(text)
    for word in UniqueWordList:
        PermutationList = []
        GeneratePermutationsByReplacing(word)
        print(PermutationList)


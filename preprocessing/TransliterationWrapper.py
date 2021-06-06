# -*- coding: utf-8 -*-
import Transliteration as tr
import GeneratePermutations as gp
import FindBestSuggestionLetterBased as bestword_letter_based
import FindBestWordContextBased as bestword_context_based
import json
import nltk
import csv
import string
import sys
from nltk.util import ngrams
from nltk.lm import NgramCounter
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from itertools import chain
sys.setrecursionlimit(10**9)
special_chars = string.punctuation
import re


class TranslterationWrapper:
    def __init__(self,data):
        self.data = data
        self.prepo_sin_words = []
        self.trigram_counter_model = []
        self.bigram_counter_model = []
        self.unigram_counter_model = []
        self.ngram_model_for_word  = []

    def transliterate(self):
        with self.data:
            tr.transliteratecsvFile(myFile)
            print("transliterated_l1.csv file Created.")

    def loadUniquewordsInCorpus(self):
        with open('UniqueWords.txt', 'r', encoding='utf-8', errors='ignore') as file1:
            self.uniqueWordListInCorpus = json.load(file1)
            file1.close()
        print("Unique words in Corpus loaded.")

    def loadUniqueWordsInTest(self,text):
        UniqueWordList = []
        tokenized_sent = [list(map(str.lower, nltk.word_tokenize(sent)))
                          for sent in nltk.sent_tokenize(text)]
        for sent in tokenized_sent:
            for word in sent:
                if word not in UniqueWordList and not word.isnumeric() and word not in special_chars:
                    UniqueWordList.append(word);
        return UniqueWordList,tokenized_sent

    def getPermutationList(self,word):
        return gp.getPermutationsList(word,self.uniqueWordListInCorpus, self.prepo_sin_words)


    def loadPrepositions(self):
        prepositions_file = open('sinhala_preposition.csv', 'r', encoding="utf-8-sig")
        with prepositions_file:
            dataReader = csv.DictReader(prepositions_file)
            for i, row in enumerate(dataReader):
                self.prepo_sin_words.append(row['sin'])
        print(self.prepo_sin_words)
        print("Sinhala preposition list loaded.")

    def loadThreeSyllableChunks(self):
        a_file = open('threeSyllable.txt', 'r', encoding='utf-8', errors='ignore')
        x = a_file.read()
        self.trigram_counter_model = json.loads(x)
        print("Trigram counter model loaded.")

    def loadTwoSyllableChunks(self):
        a_file = open('twoSyllable.txt', 'r', encoding='utf-8', errors='ignore')
        x = a_file.read()
        self.bigram_counter_model = json.loads(x)
        print("Bigram counter model loaded.")

    def setUpUnigramModel(self):
        newsListOne = []
        with open("combined.txt", 'r', encoding='utf-8', errors='ignore') as outfile:
            newslist = json.load(outfile)
        for news in newslist:
            newsListOne.extend(news)
        text = ' '.join([str(elem) for elem in newsListOne])
        tokenized_text = [list(map(str.lower, nltk.word_tokenize(sent))) for sent in nltk.sent_tokenize(text)]
        text_unigrams = [ngrams(sent, 1) for sent in tokenized_text]
        self.unigram_counter_model = NgramCounter(text_unigrams)
        print("Unigram counter model loaded.")

    def generateNGramModelForWords(self):
        newsListOne = []
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
        self.ngram_model_for_word = model
        print("Trigram model for words loaded.")

    def genaratePadsent(self,text):
        tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                          for sent in sent_tokenize(text)]
        opensent = ['<s>', '<s>']
        closeSent = ['</s>', '</s>']
        padded_sent = tokenized_text.copy()
        for i, sent in enumerate(padded_sent):
            padded_sent[i] = opensent + sent + closeSent
        return padded_sent

    def selectBestSuggestionLetterBased(self,permutationList):
        return bestword_letter_based.getBestSuggestionLetterBased(self.unigram_counter_model, self.bigram_counter_model, self.trigram_counter_model, permutationList)

    def selectBestSuggestionContextBased(self,pad_sents, permutationList):
        return bestword_context_based.genarateBestWordsList(pad_sents,self.ngram_model_for_word,permutationList)

def oneDArray(x):
    return list(chain(*x))

def main():
    myFile = open('Part_5_401_420.csv', 'r+', encoding="utf-8")
    permutation_list ={}
    obj = TranslterationWrapper(myFile)
    obj.transliterate()
    obj.loadUniquewordsInCorpus()
    obj.loadPrepositions()
    obj.loadTwoSyllableChunks()
    obj.setUpUnigramModel()
    obj.generateNGramModelForWords()
    tranliterated_l1_file = open('transliterated_l1.csv','r',encoding='utf-8')
    tranliterated_l2_file = open('transliterated_l2.csv', 'w', encoding="utf-8", newline='')
    myFields = ['singlish_news', 'transliterated_l1', 'transliterated_l2','man_written']
    with tranliterated_l2_file:
        dataWriter = csv.DictWriter(tranliterated_l2_file, fieldnames=myFields)
        dataWriter.writeheader()
        with tranliterated_l1_file:
            dataReader = csv.DictReader(tranliterated_l1_file)
            print("Permutations are generating...")
            for row in dataReader:
                bestwordList = {}
                sent = row['transliterated_L1']
                man_written = row['man_written']
                uniquewordintest,tokenized_sents = obj.loadUniqueWordsInTest(sent)
                for word in uniquewordintest:
                    if word in obj.uniqueWordListInCorpus:
                        permutation_list[word] = [word]
                    elif word in obj.prepo_sin_words:
                        permutation_list[word] = [word]
                    else:
                        PermutationList = obj.getPermutationList(word)
                        PermutationListCopy = PermutationList.copy()
                        permutation_list[word] = PermutationListCopy
                #print(permutation_list)
                pad_sents = obj.genaratePadsent(sent)
                bestwordList = obj.selectBestSuggestionLetterBased(permutation_list)
                bestwordList = obj.selectBestSuggestionContextBased(pad_sents,bestwordList)
                #print(permutation_list)
                #print(bestwordList)
                print(sent)
                tokenized_sents = oneDArray(tokenized_sents)
                newSent = []
                for word_sent in tokenized_sents:
                    newSent.append(bestwordList[word_sent])
                text = ' '.join(w for w in newSent)
                dataWriter.writerow({'singlish_news': row['singlish_content'] , 'transliterated_l1': row['transliterated_L1'], 'transliterated_l2':text, 'man_written':man_written })
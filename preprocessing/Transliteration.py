# -*- coding: utf-8 -*-
import json
import sys
import csv
import nltk

sys.setrecursionlimit(10**9)
consonants= []
consonantsUni= []
vowels= []
vowelsUni= []
vowelModifiersUni= []
specialConsonants= []
specialConsonantsUni= []
specialCharUni= []
specialChar= []

vowelsUni.extend(['ඌ','ආ','ඊ','ඊ','ඊ','ඒ','ඌ','ඖ'])

vowels.extend(['oo','aa','ii','ie','ee','ea','uu','au'])

vowelModifiersUni.extend(['ූ','ා','ී','ී','ී','ේ','ූ','ෞ'])

vowelsUni.extend(['අ','යි','එ','උ','ඔ','ඓ'])

vowels.extend(['a','i','e','u','o','ai'])

vowelModifiersUni.extend(['','ි','ෙ','ු','ො','ෛ'])

nVowels=len(vowels)

# special characher Repaya
specialConsonantsUni.append('ර්'+'\u200D')
specialConsonantsUni.append('ර්'+'\u200D')

specialConsonants.append("/R")
specialConsonants.append("\r")

consonantsUni.extend(['ච','ත','ශ','ඥ','ඳ','ඹ','ඟ'])

consonants.extend(['ch','th','sh','gn', 'nd','mb', 'ng'])

consonantsUni.extend(['ක','ක','ග','ජ','ට','ඩ','න','ප','බ','ම','ය','ර','ල','ව','ව','ස','හ'])

consonants.extend(['k','c','g','j','t','d','n','p','b','m','y','r','l','v','w','s','h'])

consonantsUni.append('ර')
consonants.append('r')

specialCharUni.append('ෲ')
specialChar.append('ruu')
specialCharUni.append('ෘ')
specialChar.append('ru')

prepo_singhlish_word = []
prepo_alt_word = []
prepo_sin_word = []

def loadEnglishWordList():
    with open("english_words_in_corpus.txt","r",encoding='utf-8', errors='ignore') as f_en:
        english_words = json.load(f_en)
        return english_words

def translate(text):
    # special consonents
    for i in range (0,len(specialConsonants)):
        text = text.replace(specialConsonants[i], specialConsonantsUni[i])
    # consonents + special
    for i in range (0,len(specialCharUni)):
        for j in range(0,len(consonants)):
            s = consonants[j] + specialChar[i]
            v = consonantsUni[j] + specialCharUni[i]
            r = s
            text = text.replace(r, v)
    # consonants + Rakaransha + vowel modifiers
    for j in range(0,len(consonants)):
        for i in range(0,len(vowels)):
            s = consonants[j] + "r" + vowels[i]
            v = consonantsUni[j] + "්‍ර" + vowelModifiersUni[i]
            r = s
            # r = new RegExp(s, "g")
            text = text.replace(r, v)

        s = consonants[j] + "r"
        v = consonantsUni[j] + "්‍ර‍"
        r = v
        text = text.replace(r, v)


    # constants with vowels modifiers
    for i in range(0,len(consonants)):
        for j in range(0,nVowels):
            s = consonants[i]+vowels[j]
            v = consonantsUni[i] + vowelModifiersUni[j]
            r = s
            text = text.replace(r, v)

    # Hal kirima
    for i in range(0, len(consonants)):
        r = consonants[i]
        text = text.replace(r, consonantsUni[i]+"්")


    # adding vowels
    for i in range(0,len(vowels)):
        r = vowels[i]
        text = text.replace(r, vowelsUni[i])

    return text

def loadPrepositions():
    prepositions_file = open('sinhala_preposition.csv', 'r', encoding="utf-8-sig")
    with prepositions_file:
        dataReader = csv.DictReader(prepositions_file)
        for i, row in enumerate(dataReader):
            prepo_sin_word.append(row['sin'].lower())
            prepo_singhlish_word.append(row['singlish'].lower())
            prepo_alt_word.append(row['alt1'].lower())



def transliteratecsvFile(file):
    output = []
    loadPrepositions()
    print(prepo_singhlish_word)
    print(prepo_sin_word)
    print(prepo_alt_word)
    eng = loadEnglishWordList()
    print(eng)
    dataReader = csv.DictReader(file)
    transliterated_file = open('transliterated_l1.csv', 'w', encoding="utf-8", newline='')
    with transliterated_file:
        myFields = ['singlish_content', 'transliterated_L1','man_written']
        dataWriter = csv.DictWriter(transliterated_file, fieldnames=myFields)
        dataWriter.writeheader()
        for row in dataReader:
            sent = row['content'].lower()
            words = nltk.wordpunct_tokenize(sent)
            for i in range(len(words)):
                if words[i] not in eng :
                    if words[i] in prepo_singhlish_word:
                        index = prepo_singhlish_word.index(words[i])
                        words[i] = prepo_sin_word[index]
                    elif words[i] in prepo_alt_word:
                        index = prepo_alt_word.index(words[i])
                        words[i] = prepo_sin_word[index]
                    else:
                        words[i] = translate(words[i])
            text = ' '.join(word for word in words)
            dataWriter.writerow({'singlish_content': sent ,'transliterated_L1': text, 'man_written': row['man_written']})


if __name__ == '__main__':
    output = []
    eng = loadEnglishWordList()
    print(eng)
    with open('../data.csv', 'rt')as f:
        data = csv.DictReader(f)
        for row in data:
            sent = row['content'].lower()
            words = nltk.wordpunct_tokenize(sent)
            for i in range(len(words)):
                if words[i] not in eng :
                    words[i] = translate(words[i])
            text = ' '.join(word for word in words)
            print(text)
            with open('adaderana.txt', 'a', encoding='utf-8', errors='ignore') as file1:
                file1.write(text)
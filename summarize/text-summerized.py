#!/usr/bin/env python
# coding: utf-8
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from tkinter import *
from tkinter import filedialog

import ast
import networkx as nx
from nltk.tokenize import RegexpTokenizer


def countWords(list_of_words):
    counts_dict = {}
    for word in list_of_words:
        if word in counts_dict:
            counts_dict[word] = counts_dict[word] + 1
        else:
            counts_dict[word] = 1
    return counts_dict


def intersection(tweetdata_one, tweetdata_two):
    result_intesection = 0
    for word in tweetdata_one:
        while tweetdata_one[word] != 0 and word in tweetdata_two:
            if word in tweetdata_two:
                tweetdata_two[word] = tweetdata_two[word] - 1
                tweetdata_one[word] = tweetdata_one[word] - 1
                if tweetdata_two[word] == 0:
                    tweetdata_two.pop(word, None)
                result_intesection += 1
    return result_intesection


def union(tweetdata_one, tweetdata_two):
    result_union = 0
    for word in tweetdata_one:
        if word in tweetdata_two:
            result_union = result_union + max(tweetdata_one[word], tweetdata_two[word])
            tweetdata_two.pop(word, None)
        else:
            result_union = result_union + tweetdata_one[word]
    for word in tweetdata_two:
        result_union = result_union + tweetdata_two[word]
    return result_union


def jaccard_distance(tweetDataOne, tweetDataTwo):
    tweetDataOne_count = countWords(tweetDataOne)
    tweetDataTwo_count = countWords(tweetDataTwo)
    tweetdata_union = union(dict(tweetDataOne_count), dict(tweetDataTwo_count))
    tweetdata_intersect = intersection(dict(tweetDataOne_count), dict(tweetDataTwo_count))
    return 1.0 - tweetdata_intersect * 1.0 / tweetdata_union

sentence_list = []
def read_microblog():
    global sentence_list
    file = filedialog.askopenfilename(
        initialdir="C:/Users/MainFrame/Desktop/",
        title="Open Text file",
        filetypes=(("Text Files", "*.txt"),)
    )
    pathh.insert(END, file)
    file = open(file, "r", encoding="utf-8")
    file_data = file.read()
    txtarea.insert(END, file_data)
    dict_value = ast.literal_eval(file_data)

    # false_news = {586266658731388929:0,586260160462589954:1,586238751334125569:1}

    kys = []
    for key in dict_value:
        # if(false_news[key] == 1):
        kys.append(key)
    print(kys)

    data = []
    length = len(kys)
    for i in range(length):
        if kys[i] != 0:
            data_string = dict_value[kys[i]]
            for g in kys[i + 1:]:
                if g != 0:
                    jd = jaccard_distance(dict_value[kys[i]], dict_value[g])
                    if jd < 0.5:
                        data_string = data_string +"."+ dict_value[g]
                        kys.remove(g)
                        kys.append(0)
            data.append(data_string)
    sentence_list = data


def read_article(clustered_sentences):
    filedata = clustered_sentences
    tokenizer = RegexpTokenizer('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', gaps=True)
    article = tokenizer.tokenize(filedata)
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    # sentences.pop()
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

index = 1
list_data_1 = ''
def generate_summary(file_name, top_n):
    nltk.download("stopwords")
    stop_words = stopwords.words('sinhala')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))
    global list_data_1
    global index
    begining = str(index) + ". "
    list_data_1 = begining + list_data_1 + ". ".join(summarize_text) + "\n\n"
    index = index + 1

# let's begin


def create_data():
    print(sentence_list)
    for sentence in sentence_list:
        generate_summary(sentence, 1)
    txtarea.delete('1.0', 'end')

def open_result():
    txtarea.insert(END, list_data_1)
    print(list_data_1)


ws = Tk()
ws.title("PythonGuides")
ws.geometry("400x450")
ws['bg']='#fb0'

txtarea = Text(ws, width=40, height=20)
txtarea.pack(pady=20)

pathh = Entry(ws)
pathh.pack(side=LEFT, expand=True, fill=X, padx=20)



Button(
    ws,
    text="Open File",
    command=read_microblog
    ).pack(side=RIGHT, expand=True, fill=X, padx=20)

Button(
    ws,
    text="clean",
    command=create_data
    ).pack(side=LEFT, expand=True, fill=X, padx=20)

Button(
    ws,
    text="open",
    command=open_result
    ).pack(side=LEFT, expand=True, fill=X, padx=20)



ws.mainloop()
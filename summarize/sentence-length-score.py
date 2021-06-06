import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
from nltk.corpus import stopwords
from nltk.text import TextCollection
stop_words = stopwords.words('sinhala')
import pandas as pd
from scipy.spatial import distance
import ast

from tkinter import *
from tkinter import filedialog
from scipy.spatial import distance

print(stop_words)

sentence_length = []

apple = []

paragraph_locations = []
0
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
        if kys[i] !=0 :
            data_string = dict_value[kys[i]]
            for g in kys[i + 1:]:
                if g != 0:
                    jd = jaccard_distance(dict_value[kys[i]], dict_value[g])
                    if jd < 0.5:
                        data_string = data_string +". "+ dict_value[g]
                        kys.remove(g)
                        kys.append(0)
            data.append(data_string)
    sentence_list = data




def read_article(clustered_sentences):

    file_data = clustered_sentences
    print('{0} {1}'.format("this is the paragraph :", file_data))

    tokenizer = RegexpTokenizer('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', gaps=True)

    sentences = tokenizer.tokenize(file_data)

    print('{0} {1}'.format("sentences of tokenized paragraph :", len(sentences)))

    mean_length = np.sum([len(sentence) for sentence in sentences]) / len(sentences)



    article = []

    list_of_words = []
    for sentence in sentences:
        tokenizer = RegexpTokenizer('\s+', gaps=True)
        word_list = tokenizer.tokenize(sentence)
        tf_idf(word_list)
        [list_of_words.append(item) for item in word_list]


        sentence_length.append(sentence_length_scores(sentence,mean_length))
    print(list_of_words)

    # print('{0} {1}'.format("those are the counts",count))
    print(len(sentence_length))

    print('{0} {1}'.format("words of tokenized sentences :", article))

    print('{0} {1}'.format("sentence mean length", mean_length))


    # paragraph_location(sentences)

    return sentences

def sentence_length_scores(sentence,mean_length):

    return len(sentence)/mean_length


all_word_tf_idf = []




def tf_idf(article):
    word_list = {}

    word_tf_idf = []

    myArticle = TextCollection(article)

    for word in article:
        if word not in stop_words:
            if word not in word_list:
                word_tf_idf.append(myArticle.tf_idf(word, article))


        else:
            word_tf_idf.append(0)

    vector = np.asarray(word_tf_idf)

    print(np.sum(vector))
    all_word_tf_idf.append(np.sum(vector))

print("#### all tf idf")
print(all_word_tf_idf)

def paragraph_location(sentences):



    for l in range(1,len(sentences)+1):
        if l <=5:
            paragraph_locations.append(l/5)

        else:
            if len(sentences)-4<=l:
                paragraph_locations.append((l-(len(sentences)-5))/5)

            else:
                paragraph_locations.append(0.0)






# sentence_list = read_microblog("tweet1.txt")
index = 1
list_data_1 = ''
def create_data():
    for sentence in sentence_list:
        all_word_tf_idf.clear()
        cluster_sentences = read_article(sentence)
        if len(cluster_sentences) <= 3:
            continue
        print(all_word_tf_idf)

        all_word_tf_idf.sort()

        print(all_word_tf_idf)

        all_sentence_tf_idf = all_word_tf_idf/np.max(all_word_tf_idf)

        print('{0} {1}'.format("cluster sentences" , cluster_sentences))

        print(all_sentence_tf_idf)

        print(sentence_length)

        # print(paragraph_locations)

        all_feartures = []
        all_feartures.clear()
        for l in range(0,len(all_word_tf_idf)):
            vector = []
            vector.append(all_sentence_tf_idf[l])
            vector.append(sentence_length[l])
            # vector.append(paragraph_locations[l])

            all_feartures.append(vector)

        print(len(all_feartures))



        k = 3
        p = 2

        X = pd.DataFrame(all_feartures)




        # Print the number of data and dimension
        n = len(X)
        d = len(X.columns)
        addZeros = np.zeros((n, 1))
        X = np.append(X, addZeros, axis=1)
        print("The FCM algorithm: \n")
        print("The training data: \n", X)
        print("\nTotal number of data: ", n)
        print("Total number of features: ", d)
        print("Total number of Clusters: ", k)

        # Create an empty array of centers
        C = np.zeros((k, d + 1))
        # print(C)

        # Randomly initialize the weight matrix
        weight = np.random.dirichlet(np.ones(k), size=n)
        print("\nThe initial weight: \n", np.round(weight, 2))

        for it in range(3000):  # Total number of iterations

            # Compute centroid
            for j in range(k):
                denoSum = sum(np.power(weight[:, j], 2))

                sumMM = 0
                for i in range(n):
                    mm = np.multiply(np.power(weight[i, j], p), X[i, :])
                    sumMM += mm
                cc = sumMM / denoSum
                C[j] = np.reshape(cc, d + 1)

        # print("\nUpdating the fuzzy pseudo partition")
            for i in range(n):
                denoSumNext = 0
                for j in range(k):
                    denoSumNext += np.power(1 / distance.euclidean(C[j, 0:d], X[i, 0:d]), 1 / (p - 1))
                for j in range(k):
                    w = np.power((1 / distance.euclidean(C[j, 0:d], X[i, 0:d])), 1 / (p - 1)) / denoSumNext
                    weight[i, j] = w

        print("\nThe final weights: \n", np.round(weight, 2))

        for i in range(n):
            cNumber = np.where(weight[i] == np.amax(weight[i]))
            print(cNumber)
            X[i, d] = cNumber[0]

        print("\nThe data with cluster number: \n", X)

        # Sum squared error calculation
        SSE = 0
        for j in range(k):
            for i in range(n):
                SSE += np.power(weight[i, j], p) * distance.euclidean(C[j, 0:d], X[i, 0:d])

        print("\nSSE: ", np.round(SSE, 4))


        clusters = X[:,2]

        print(len(clusters))

        r_1 = 0
        r_2 = 0
        r_3 = 0
        y = []
        y.clear()
        concatenate_sentence  = ''

        #print('{0} {1}'.format("cluster_sentence", cluster_sentences))
        global index
        global list_data_1
        begining = str(index) + ". "
        d1 = 0
        mind1 = 1
        ind_1 = 0

        d2 = 0
        mind2 = 1
        ind_2 = 0

        d3 = 0
        mind3 = 1
        ind_3 = 0
        list_data_2 = weight.tolist()
        for i in clusters:
             if i == 2.0:
                 print(C[2])
                 print(distance.euclidean(list_data_2[r_1],C[2]))
                 d1 = distance.euclidean(list_data_2[r_1], C[2])
                 if d1 < mind1:
                     mind1 = d1
                     ind_1 = r_1
             r_1 += 1
        y.append(ind_1)


        for i in clusters:
             if i == 1.0:
                 print(C[1])
                 print(distance.euclidean(list_data_2[r_2],C[1]))
                 d2 = distance.euclidean(list_data_2[r_2], C[1])
                 if d2 < mind2:
                     mind2 = d2
                     ind_2 = r_2
             r_2 += 1
        y.append(ind_2)

        for i in clusters:
             if i == 0.0:
                 print(C[0])
                 print(distance.euclidean(list_data_2[r_3],C[0]))
                 d3 = distance.euclidean(list_data_2[r_3], C[0])
                 if d3 < mind3:
                     mind3 = d3
                     ind_3 = r_3
             r_3 += 1
        y.append(ind_3)

        for u in y:
            print(cluster_sentences[u])
            concatenate_sentence = concatenate_sentence + cluster_sentences[u]

        print(y)
        list_data_1 = list_data_1 + concatenate_sentence + "\n\n" + begining
        index = index + 1
        print(concatenate_sentence)

def open_data():
    txtarea.delete('1.0', 'end')
    print(list_data_1)
    txtarea.insert(END, list_data_1)

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
    command=open_data
    ).pack(side=LEFT, expand=True, fill=X, padx=20)

ws.mainloop()








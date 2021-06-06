# -*- coding: utf-8 -*-
# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
import nltk.translate.gleu_score as gleu
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
import csv

def evaluateUsingBLEU(transliterated_file,row_name):
    score = 0
    dataReader = csv.DictReader(transliterated_file)
    for i, row in enumerate(dataReader):
        sent_man_written = row['man_written']
        reference = [tokenizer.tokenize(sent_man_written)]
        sent_machine_gen = row[row_name].lower()
        candidate = tokenizer.tokenize(sent_machine_gen)
        score += sentence_bleu(reference, candidate, weights=(1.0, 0.0, 0.0, 0.0))
    avg_score = score / (i + 1)
    print(row_name,"| Score using BLEU : ",avg_score)

def evaluateUsingGLEU(transliterated_file,row_name):
    j, score = 0,0
    dataReader = csv.DictReader(transliterated_file)
    for j, row in enumerate(dataReader):
        sent_man_written = row['man_written']
        reference = [tokenizer.tokenize(sent_man_written)]
        sent_machine_gen = row[row_name].lower()
        candidate = tokenizer.tokenize(sent_machine_gen)
        score += gleu.sentence_gleu(reference, candidate)
    avg_score = score / (j + 1)
    print(row_name,"| Score using GLEU : ",avg_score)

def evaluateUsingWER(transliterated_file,row_name):
    j, score = 0, 0
    dataReader = csv.DictReader(transliterated_file)
    for j, row in enumerate(dataReader):
        sent_man_written = row['man_written']
        reference = sent_man_written
        sent_machine_gen = row[row_name].lower()
        candidate = sent_machine_gen
        score += wer(reference, candidate)
    avg_score = score / (j + 1)
    print(row_name, "| Score using WER : ", avg_score)

def evaluateUsingTER(transliterated_file,row_name):
    j, score = 0, 0
    dataReader = csv.DictReader(transliterated_file)
    for j, row in enumerate(dataReader):
        sent_man_written = row['man_written']
        reference = sent_man_written
        sent_machine_gen = row[row_name].lower()
        candidate = sent_machine_gen
        score += ter(reference, candidate)
    avg_score = score / (j + 1)
    print(row_name,"| Score using TER : ", avg_score)

def wer(ref, hyp ,debug=False):
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY=1 # Tact
    INS_PENALTY=1 # Tact
    SUB_PENALTY=1 # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("Ncor " + str(numCor))
        print("Nsub " + str(numSub))
        print("Ndel " + str(numDel))
        print("Nins " + str(numIns))
    return (numSub + 0.5 * numDel + 0.5* numIns) / (float) (len(r))
    wer_result = round( (numSub + 0.5* numDel + 0.5* numIns) / (float) (len(r)), 3)
    return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}

def ter(ref, hyp ,debug=False):
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY=1 # Tact
    INS_PENALTY=1 # Tact
    SUB_PENALTY=1 # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("Ncor " + str(numCor))
        print("Nsub " + str(numSub))
        print("Ndel " + str(numDel))
        print("Nins " + str(numIns))
    return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}

if __name__ == '__main__':
    transliterated_file = open('transliterated_l2.csv','r',encoding='utf-8')
    tokenizer = RegexpTokenizer('[\/\-\,\(\)\.\s+]', gaps=True)
    transliterated_file_2 = open('transliterated_l2g1.csv', 'r', encoding='utf-8')
    with transliterated_file_2:
        evaluateUsingBLEU(transliterated_file_2, 'transliterated_l1')
        transliterated_file_2.seek(0)
        evaluateUsingGLEU(transliterated_file_2, 'transliterated_l1')
        transliterated_file_2.seek(0)
        evaluateUsingWER(transliterated_file_2, 'transliterated_l1')
        transliterated_file_2.seek(0)
        evaluateUsingTER(transliterated_file_2, 'transliterated_l1')
        transliterated_file_2.seek(0)
        evaluateUsingBLEU(transliterated_file_2,'transliterated_l2')
        transliterated_file_2.seek(0)
        evaluateUsingGLEU(transliterated_file_2,'transliterated_l2')
        transliterated_file_2.seek(0)
        evaluateUsingWER(transliterated_file_2, 'transliterated_l2')
        transliterated_file_2.seek(0)
        evaluateUsingTER(transliterated_file_2, 'transliterated_l2')
        transliterated_file_2.seek(0)
        evaluateUsingBLEU(transliterated_file_2, 'google')
        transliterated_file_2.seek(0)
        evaluateUsingGLEU(transliterated_file_2, 'google')
        transliterated_file_2.seek(0)
        evaluateUsingWER(transliterated_file_2, 'google')
        transliterated_file_2.seek(0)
        evaluateUsingTER(transliterated_file_2, 'google')
        transliterated_file_2.seek(0)
    transliterated_file.close()
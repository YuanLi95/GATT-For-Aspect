# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import spacy

import sys
import scipy.sparse as sp
from spacy import displacy
from pathlib import Path
nlp = spacy.load('en_core_web_sm')

def dependency_adj_matrix(text):
    # text = "Great food but the service was dreadful !"
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    matrix= normalize(matrix)
    return matrix


def token_speech_weight(token):
    if token == "ADJ":  # 关注形容词
        simple_weigth = 1.0
    elif (token =="CCONJ"):
        simple_weigth = 1.0
    elif (token == "ADV"):  # 副词及其连接词
        simple_weigth = 1.0
    else:
        simple_weigth = 1.0

    return simple_weigth

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一个特征进行归一化
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# 获得词性的一个list
def  Part_of_speech_list(text):
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return  pos

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    part_of_speech = {}
    fout = open(filename + '.graph', 'wb')
    speech = open(filename + '.speech', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right)
        part_of_speech_verctor = Part_of_speech_list(text_left + ' '+aspect +' '+ text_right)

        idx2graph[i] = adj_matrix
        part_of_speech[i] = list(map(lambda  x :token_speech_weight(x),part_of_speech_verctor))

    pickle.dump(idx2graph, fout)
    pickle.dump(part_of_speech, speech)
    fout.close()
    speech.close()

if __name__ == '__main__':
    # process('./datasets/acl-14-short-data/train.raw')
    # process('./datasets/acl-14-short-data/test.raw')
    # process('./datasets/semeval14/restaurant_train.raw')
    # process('./datasets/semeval14/restaurant_test.raw')
    # process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    # process('./datasets/semeval15/restaurant_train.raw')
    # process('./datasets/semeval15/restaurant_test.raw')
    # process('./datasets/semeval16/restaurant_train.raw')
    # process('./datasets/semeval16/restaurant_test.raw')

    # doc = nlp("The price is reasonable although the service is poor .")
    # svg = spacy.displacy.render(doc, style="dep", jupyter=False)
    # file_name = '-'.join([w.text for w in doc if not w.is_punct]) + ".svg"
    # output_path = Path("./" + file_name)
    # output_path.open("w", encoding="utf-8").write(svg)
    # exit()

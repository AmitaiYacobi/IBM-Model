import os
import sys
import json
import pickle
import functools
import numpy as np
from collections import defaultdict


def dd(num_of_e_words):
    return float(1/num_of_e_words)

def get_sentence_pairs(e_file, f_file):
    sentence_pairs = []
    for e, f in zip(e_file, f_file):
        sentence_pairs.append((e.strip() ,f.strip()))
    return sentence_pairs

def init_translation_propabilities(sentence_pairs):
    num_of_e_words = len(set(e_word for (e_sentence, f_sentence) in sentence_pairs for e_word in e_sentence))
    translation_propabilities = defaultdict(functools.partial(dd, num_of_e_words))
    return translation_propabilities

def init_alignment_propabilities(sentence_pairs):
    num_of_f_words = len(set(f_word for (e_sentence, f_sentence) in sentence_pairs for f_word in f_sentence))
    alignment_propabilities = defaultdict(functools.partial(dd, num_of_f_words))
    return alignment_propabilities

def calc_distance(previous_table, current_table):
    result = 0
    for key in previous_table.keys():
        delta = (previous_table[key] - current_table[key]) ** 2
        result += delta

    return result ** 0.5

def is_conveged(previous_table, current_table, epsilon=0.001):
    delta = calc_distance(previous_table, current_table)
    print(f"delta is: {delta}")
    return delta < epsilon

def write_translation_probs(translation_propabilities):
    with open ("./results/propabilities.pkl", "wb") as file:
        pickle.dump(translation_propabilities, file)

def load_translation_probs(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data

def align_sentences(sentence_pairs, translation_propabilities):
    print("Start aligning...")
    with open("./results/alignments.txt", "w") as file:
        for e_sentence, f_sentence in sentence_pairs:
            e_sentence = e_sentence.split()
            f_sentence = f_sentence.split()
            for (i, f) in enumerate(f_sentence): 
                maximum = 0
                for (j, e) in enumerate(e_sentence):
                    if translation_propabilities[(e,f)] >= maximum:
                        max_i = i
                        max_j = j
                        maximum  = translation_propabilities[(e,f)]
                file.write("%i-%i " % (max_i,max_j))
            file.write("\n")
    print("Alignment finished!")

def evaluate():
    os.system("./given/eval.py -e ./given/data/hansards.e -f ./given/data/hansards.f -a ./given/data/hansards.a < results/alignments.txt -n 0")
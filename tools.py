import os
import sys
import math
import json
import pickle
import random
import functools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def dd(x):
    return float(1/x)

def get_sentence_pairs(e_file, f_file, num_of_sentences=100000):
    return [[sentence.strip().split() for sentence in pair] for pair in list(zip(f_file, e_file))[:num_of_sentences]]

def init_translation_propabilities(sentence_pairs):
    num_of_f_words = len(set(f_word for (f_sentence, e_sentence) in sentence_pairs for f_word in f_sentence))
    translation_propabilities = defaultdict(functools.partial(dd, num_of_f_words))
    return translation_propabilities

def init_alignment_propabilities(sentence_pairs):
    return init_translation_propabilities(sentence_pairs)

def write_translation_probs(translation_propabilities):
    with open ("./results/propabilities.pkl", "wb") as file:
        pickle.dump(translation_propabilities, file)

def load_translation_probs(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data

def align_sentences(sentence_pairs, translation_propabilities, model):
    print("Start aligning...")
    with open("./results/alignments_"+model+".txt", "w") as file:
        for f_sentence, e_sentence in sentence_pairs:
            for (j, e) in enumerate(e_sentence): 
                maximum = 0
                for (i, f)in enumerate(f_sentence):
                    if translation_propabilities[(e,f)] >= maximum:
                        max_i = i
                        max_j = j
                        maximum  = translation_propabilities[(e,f)]
                file.write("%i-%i " % (max_i,max_j))
            file.write("\n")
    print("Alignment finished!")

def evaluate(model):
    os.system("./given/eval.py -e ./given/data/hansards.e -f ./given/data/hansards.f -a ./given/data/hansards.a < results/alignments_"+model+".txt -n 0 > results.txt")
    with open("./results.txt") as f:
        results = f.readlines()
    return results[2].split()[2]

def create_aer_per_epochs_graph():
    rates = {}
    i_model1 = open("./based_on_model1.txt", "r").readlines()
    i_uniform = open("./uniform.txt", "r").readlines()

    epochs = [int(i) for i in i_model1[0][1:-2].strip().split(",")]
    rates["model1"] = [float(i) for i in i_model1[1][1:-2].strip().split(",")]
    rates["uniform"] = [float(i) for i in i_uniform[1][1:-2].strip().split(",")]
    plt.plot(epochs, rates["model1"], color='r', label='Based on model 1')
    plt.plot(epochs, rates["uniform"], color='b', label="1/(Number of french words)")
    plt.xlabel("Epochs")
    plt.ylabel("AER")
    plt.legend()
    plt.savefig("aer_per_init_model2.png")



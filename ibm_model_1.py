import sys
import json
import pickle
import functools
import numpy as np
from collections import defaultdict

def dd(num_of_e_words):
    return defaultdict(float(1/num_of_e_words))

def get_sentence_pairs(e_file, f_file):
    sentence_pairs = {}
    for e, f in zip(e_file, f_file):
        sentence_pairs[e.strip()] = f.strip()
    return sentence_pairs

def init_translation_propabilities(sentence_pairs):
    num_of_e_words = len(set(e_word for (e_sentence, f_sentence) in sentence_pairs for e_word in e_sentence))
    translation_propabilities = defaultdict(functools.partial(dd, num_of_e_words))
    for (e_sentence, f_sentence) in sentence_pairs:
        for e in e_sentence:
            for f in f_sentence:
                translation_propabilities[(e,f)] = 1/num_of_e_words
    return translation_propabilities

def em_iteration(sentence_pairs, translation_propabilities):
    s_total = defaultdict(float)
    total = defaultdict(float)
    count = defaultdict(float)

    for e_sentence, f_sentence in sentence_pairs:
        for e in e_sentence:
            s_total[e] = 0
            for f in f_sentence:
                s_total[e] += translation_propabilities[(e,f)]
        
        for e in e_sentence:
            for f in f_sentence:
                count[(e,f)] += count[(e,f)] / s_total[e]
                total[f] += translation_propabilities[(e,f)] / s_total[e]

    f_words = total.keys()
    e_words = s_total.keys()

    for f in f_words:
        for e in e_words:
            translation_propabilities[(e,f)] = count[(e,f)] / total[f]
    
    return translation_propabilities

def calc_distance(previous_table, current_table):
    row_keys = previous_table.keys()
    cols = list(current_table.values())
    col_keys = cols[0].keys()
    result = 0

    for (row_key, col_key) in zip(row_keys, col_keys):
        delta = (previous_table[row_key][col_key] -
                 current_table[row_key][col_key]) ** 2
        result += delta

    return result ** 0.5

def is_conveged(previous_table, current_table, epsilon=0.001):
    delta = calc_distance(previous_table, current_table)
    return delta < epsilon

def train(sentence_pairs):
    converged = False
    epoch = 0
    prev_translation_propabilities = init_translation_propabilities(sentence_pairs)
    while not converged:
        print(f"Epoch number {epoch}", end="\r")
        new_translation_propabilities = em_iteration(sentence_pairs, prev_translation_propabilities)
        print("start writing")
        write_translation_props(new_translation_propabilities)
        converged = is_conveged(prev_translation_propabilities, new_translation_propabilities)
        prev_translation_propabilities = new_translation_propabilities

    final_translation_propabilities = prev_translation_propabilities
    return final_translation_propabilities

def write_translation_props(translation_propabilities):
    with open ("./propabilities.pkl", "wb") as file:
        pickle.dump(translation_propabilities, file)

def load_translation_props(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data
            
def align_sentences(sentence_pairs, translation_propabilities):
    pass

def evaluate(alignments_file):
    pass

def main():
    # e_file = open("./word-alignment/data/hansards.e")
    # f_file = open("./word-alignment/data/hansards.f")
    # a_file = open("./word-alignment/data/hansards.a")
    # sentence_pairs = get_sentence_pairs(e_file, f_file)
    # props = init_translation_propabilities(sentence_pairs.items())
    # print(props.keys())
    # write_translation_props_to_txt(props)

    # final_translation_propabilities = train(sentence_pairs.items())
    # alignments_file = align_sentences(sentence_pairs, final_translation_propabilities)
    # evaluate(alignments_file)
    # write_translation_props_to_txt(final_translation_propabilities)
    # write_translation_props_to_txt(props)

if __name__ == "__main__":
    main()

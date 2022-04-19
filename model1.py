import matplotlib.pyplot as plt

from tools import *
from collections import defaultdict

def em_iteration(sentence_pairs, translation_propabilities):
    total = defaultdict(float)
    count = defaultdict(float)
    for f_sentence, e_sentence in sentence_pairs:
        s_total = defaultdict(lambda: 0.0)
        for e in e_sentence:
            for f in f_sentence:
                s_total[e] += translation_propabilities[(e,f)]
        
        for e in e_sentence:
            for f in f_sentence:
                count[(e,f)] += translation_propabilities[(e,f)] / s_total[e]
                total[f] += translation_propabilities[(e,f)] / s_total[e]

    print("Start updating translation probs...")
    c = 0
    for (e,f) in count.keys():
        print(f"{c}", end="\r")
        c += 1        
        translation_propabilities[(e,f)] = (count[(e,f)] + 1)  / (total[f] + 10000) 
    print("Update finished!")
    return translation_propabilities


def train(sentence_pairs, iternum=25):
    prev_translation_propabilities = init_translation_propabilities(sentence_pairs)
    for epoch in range(iternum):
        print(f"Epoch number {epoch}")
        new_translation_propabilities = em_iteration(sentence_pairs, prev_translation_propabilities)
        align_sentences(sentence_pairs, new_translation_propabilities, "model1")
        aer = evaluate("model1")
        print(f"AER = {aer}")
        prev_translation_propabilities = new_translation_propabilities

    final_translation_propabilities = prev_translation_propabilities
    return final_translation_propabilities





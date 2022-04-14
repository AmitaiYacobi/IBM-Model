from tools import *
from collections import defaultdict


def em_iteration(sentence_pairs, translation_propabilities, alignment_propabilities):
    s_total = defaultdict(float)
    total = defaultdict(float)
    count = defaultdict(float)
    
    total_a = defaultdict(float)
    count_a = defaultdict(float)
    
    for e_sentence, f_sentence in sentence_pairs:
        e_sentence = e_sentence.split()
        f_sentence = f_sentence.split()
        le = len(e_sentence)
        lf = len(f_sentence)

        for j, e in enumerate(e_sentence):
            s_total[e] = 0.0
            for i, f in enumerate(f_sentence):
                s_total[e] += translation_propabilities[(e,f)] * alignment_propabilities[(i,j,le,lf)]
        
        for j, e in enumerate(e_sentence):
            for i, f in enumerate(f_sentence):
                c = translation_propabilities[(e,f)] * alignment_propabilities[(i,j,le,lf)] / s_total[e]
                count[(e,f)] += c
                count_a[(i,j,le,lf)] += c
                total[f] += c
                total_a[(j,le,lf)] += c

    f_words = total.keys()
    e_words = s_total.keys()

    for f in f_words:
        for e in e_words:
            if total[f] != 0:
                translation_propabilities[(e,f)] = count[(e,f)] / total[f]
    
    for e_sentence, f_sentence in sentence_pairs:
        e_sentence = e_sentence.split()
        f_sentence = f_sentence.split()
        le = len(e_sentence)
        lf = len(f_sentence)
        for i in range(lf):
            for j in range(le):
                if total_a[(j,le,lf)] != 0:
                    alignment_propabilities[(i,j,le,lf)] = count_a[(i,j,le,lf)] / total_a[(j,le,lf)]

    return translation_propabilities


def train(sentence_pairs, iternum=30):
    prev_translation_propabilities = load_translation_probs(sentence_pairs)
    prev_alignment_propabilities = init_alignment_propabilities(sentence_pairs)
    for epoch in range(iternum):
        print(f"Epoch number {epoch}", end="\r")
        new_translation_propabilities, new_alignment_propabilities = em_iteration(sentence_pairs, 
                                                                                  prev_translation_propabilities,
                                                                                  prev_alignment_propabilities)
        if epoch % 5 == 0:
            align_sentences(sentence_pairs, new_translation_propabilities)
        prev_translation_propabilities = new_translation_propabilities

    final_translation_propabilities = prev_translation_propabilities
    return final_translation_propabilities
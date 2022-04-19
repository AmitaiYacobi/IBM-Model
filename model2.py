from tools import *
from collections import defaultdict


def em_iteration(sentence_pairs, translation_propabilities, alignment_propabilities):
    total = defaultdict(float)
    count = defaultdict(float)
    
    total_a = defaultdict(float)
    count_a = defaultdict(float)
    
    for f_sentence, e_sentence in sentence_pairs:

        le = len(e_sentence)
        lf = len(f_sentence)
        s_total = defaultdict(lambda: 0.0)

        for j, e in enumerate(e_sentence):
            for i, f in enumerate(f_sentence):
                s_total[e] += translation_propabilities[(e,f)] * alignment_propabilities[(i,j,le,lf)]
        
        for j, e in enumerate(e_sentence):
            for i, f in enumerate(f_sentence):
                c = translation_propabilities[(e,f)] * alignment_propabilities[(i,j,le,lf)] / s_total[e]
                count[(e,f)] += c
                count_a[(i,j,le,lf)] += c
                total[f] += c
                total_a[(j,le,lf)] += c

    print("Start updating translation probs...")
    c = 0
    for (e,f) in count.keys():
        print(f"{c}", end="\r")
        c += 1        
        translation_propabilities[(e,f)] = count[(e,f)] / total[f]
    print("Update finished!")
    
    print("Start updating alignments probs...")
    for f_sentence, e_sentence in sentence_pairs:
        le = len(e_sentence)
        lf = len(f_sentence)
        for i in range(lf):
            for j in range(le):
                if total_a[(j,le,lf)] != 0:
                    alignment_propabilities[(i,j,le,lf)] = count_a[(i,j,le,lf)] / total_a[(j,le,lf)]

    print("Update finished!")

    return translation_propabilities, alignment_propabilities


def train(sentence_pairs, iternum=25):
    # prev_translation_propabilities = load_translation_probs("./results/propabilities.pkl")
    prev_translation_propabilities = init_translation_propabilities(sentence_pairs)
    prev_alignment_propabilities = init_alignment_propabilities(sentence_pairs)
    for epoch in range(iternum):
        print(f"Epoch number {epoch}")
        new_translation_propabilities, new_alignment_propabilities = em_iteration(sentence_pairs, 
                                                                                  prev_translation_propabilities,
                                                                                  prev_alignment_propabilities)
        align_sentences(sentence_pairs, new_translation_propabilities, "model2")
        aer = evaluate("model2")
        print(f"AER = {aer}")
        prev_translation_propabilities = new_translation_propabilities
        prev_alignment_propabilities = new_alignment_propabilities

    final_translation_propabilities = prev_translation_propabilities
    return final_translation_propabilities
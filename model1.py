from tools import *
from collections import defaultdict

def em_iteration(sentence_pairs, translation_propabilities):
    s_total = defaultdict(float)
    total = defaultdict(float)
    count = defaultdict(float)

    for e_sentence, f_sentence in sentence_pairs:
        e_sentence = e_sentence.split()
        f_sentence = f_sentence.split()
        for e in e_sentence:
            s_total[e] = 0.0
            for f in f_sentence:
                s_total[e] += translation_propabilities[(e,f)]
        
        for e in e_sentence:
            for f in f_sentence:
                count[(e,f)] += translation_propabilities[(e,f)] / s_total[e]
                total[f] += translation_propabilities[(e,f)] / s_total[e]

    print("Start updating translation probs...")
    c = 0
    for e_sentence, f_sentence in sentence_pairs:
        e_sentence = e_sentence.split()
        f_sentence = f_sentence.split()
        for f in f_sentence:
            print(f"{c}", end="\r")
            c += 1        
            for e in e_sentence:
                translation_propabilities[(e,f)] = count[(e,f)] / total[f]
    print("Update finished!")
    return translation_propabilities


def train(sentence_pairs, iternum=30):
    prev_translation_propabilities = init_translation_propabilities(sentence_pairs)
    for epoch in range(iternum):
        print(f"Epoch number {epoch}")
        new_translation_propabilities = em_iteration(sentence_pairs, prev_translation_propabilities)
        # is_conveged(prev_translation_propabilities, new_translation_propabilities)
        if epoch % 5 == 0:
            align_sentences(sentence_pairs, new_translation_propabilities)
            # write_translation_probs(new_translation_propabilities)
            # evaluate()
        prev_translation_propabilities = new_translation_propabilities

    final_translation_propabilities = prev_translation_propabilities
    return final_translation_propabilities






import sys

def get_sentence_pairs(e_file, f_file):
    sentence_pairs = {}
    for e, f in zip(e_file, f_file):
        sentence_pairs[e] = f
    return sentence_pairs

def get_words(sentences):
    words = []
    for sentence in sentences:
        for word in sentence.split():
            if word not in words:
                words.append(word)
    return words

def init_translation_propabilities(e_words, f_words):
    return {
                e_word: {
                    f_word: 1/ len(e_words) for f_word in f_words
                } for e_word in e_words 
            }


def em_iteration(sentence_pairs, e_words, f_words, translation_propabilities):
    s_total = {}
    total = {word: 0 for word in f_words}
    count = {e_word: {f_word: 0 for f_word in f_words} for e_word in e_words}

    for e_sentence, f_sentence in sentence_pairs:
        for e in e_sentence:
            s_total[e] = 0
            for f in f_sentence:
                s_total[e] += translation_propabilities[e][f]
        for e in e_sentence:
            for f in f_sentence:
                count[e][f] += translation_propabilities[e][f] / s_total[e]
                total[f] += translation_propabilities[e][f] / s_total[e]
    for f in f_words:
        for e in e_words:
            translation_propabilities[e][f] = count[e][f] / total[f]

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
    e_words = get_words(sentence_pairs.keys())
    f_words = get_words(sentence_pairs.values())
    prev_translation_propabilities = init_translation_propabilities(e_words, f_words)

    converged = False
    while not converged:
        new_translation_propabilities = em_iteration(sentence_pairs, e_words, f_words, prev_translation_propabilities)
        converged = is_conveged(prev_translation_propabilities, new_translation_propabilities)
        prev_translation_propabilities = new_translation_propabilities
    
    final_translation_propabilities = prev_translation_propabilities
    return final_translation_propabilities

def write_translation_props_to_txt(translation_propabilities):
    pass

def align_sentences(sentence_pairs, translation_propabilities):
    pass

def evaluate(alignments_file):
    pass

def main():
    e_file = open("./word-alignment/data/hansards.e")
    f_file = open("./word-alignment/data/hansards.f")
    a_file = open("./word-alignment/data/hansards.a")
    sentence_pairs = get_sentence_pairs(e_file, f_file)
    final_translation_propabilities = train(sentence_pairs)
    # alignments_file = align_sentences(sentence_pairs, final_translation_propabilities)
    # evaluate(alignments_file)
    # write_translation_props_to_txt(final_translation_propabilities)

if __name__ == "__main__":
    main()

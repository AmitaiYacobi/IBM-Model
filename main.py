import model1
import model2
import optparse
from tools import *

model  = {
    "model1": model1,
    "model2": model2,
}

def main():
    optparser = optparse.OptionParser()
    optparser.add_option("-e", dest="e_file", default="./given/data/hansards.e", help="E filename")
    optparser.add_option("-f", dest="f_file", default="./given/data/hansards.f", help="F filename")
    optparser.add_option("-i", dest="iterations", default="30", help="number of iterations")
    optparser.add_option("-n", dest="sentences", default="100000", help="number of sentences to read from the files")
    optparser.add_option("-m", dest="model", default="model1", help="choose which model you want to run (model1 or model2)")
    
    (opts, args) = optparser.parse_args()

    e_file = open(opts.e_file)
    f_file = open(opts.f_file)
    iterations = opts.iterations

    sentence_pairs = get_sentence_pairs(e_file, f_file, int(opts.sentences))
    final_translation_propabilities = model[opts.model].train(sentence_pairs, int(iterations))
    write_translation_probs(final_translation_propabilities)
    align_sentences(sentence_pairs, final_translation_propabilities, opts.model)
    print("\n")
    print("############################")
    print("Final evaluation:")
    evaluate(opts.model)
    print("############################")
    # create_aer_per_epochs_graph()

if __name__ == "__main__":
    main()
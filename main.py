import model1
import model2
import optparse
from tools import *

model  = {
    "model1": model1,
    "model2": model2,
}

def main():
    # optparser = optparse.OptionParser()
    # optparser.add_option("-e", dest="e_file", default="./given/data/hansards.e", help="E filename")
    # optparser.add_option("-f", dest="f_file", default="./given/data/hansards.e", help="F filename")
    # optparser.add_option("-m", dest="model", default="model1", help="choose which model you want to run (model1 or model2)")
    # (opts, args) = optparser.parse_args()

    e_file = open("./given/data/hansards.e")
    f_file = open("./given/data/hansards.f")

    sentence_pairs = get_sentence_pairs(e_file, f_file)
    final_translation_propabilities = model1.train(sentence_pairs)

if __name__ == "__main__":
    main()
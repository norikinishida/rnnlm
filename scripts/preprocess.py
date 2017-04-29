# -*- coding: utf-8 -*-

import argparse

import nlppreprocess.lowercase
import nlppreprocess.tokenizer
import nlppreprocess.convert_textlines_to_characters
import nlppreprocess.replace_digits
import nlppreprocess.append_eos
import nlppreprocess.split_corpus
import nlppreprocess.create_dictionary
import nlppreprocess.replace_rare_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    args = parser.parse_args()
    
    raw = args.raw
    train = args.train
    val = args.val
    
    tmp = "tmp.txt"
    nlppreprocess.lowercase.run(raw, tmp + ".lowercase")
    nlppreprocess.tokenizer.run(tmp + ".lowercase", tmp + ".tokenize")
    # nlppreprocess.convert_textlines_to_characters.run(tmp + ".tokenize", tmp + ".tokenize.char")
    nlppreprocess.replace_digits.run(tmp + ".tokenize", tmp + ".replace_digits")
    nlppreprocess.append_eos.run(tmp + ".replace_digits", tmp + ".append_eos")
    nlppreprocess.split_corpus.run(tmp + ".append_eos", tmp + ".train", tmp + ".val", size=5000)
    nlppreprocess.create_dictionary.run(tmp + ".train", train + ".dictionary", prune_at=300000, min_count=5)
    nlppreprocess.replace_rare_words.run(tmp + ".train", train, path_dict=train + ".dictionary")
    nlppreprocess.replace_rare_words.run(tmp + ".val", val, path_dict=train + ".dictionary")


# -*- coding: utf-8 -*-

import argparse

import nlppreprocess.lowercase
import nlppreprocess.tokenizer
import nlppreprocess.convert_textlines_to_characters
import nlppreprocess.replace_digits
import nlppreprocess.append_eos
import nlppreprocess.split_corpus
import nlppreprocess.create_vocabulary
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
    nlppreprocess.lowercase.run(
            raw,
            tmp + ".lowercase")
    nlppreprocess.tokenizer.run(
            tmp + ".lowercase",
            tmp + ".lowercase.tokenize")
    # nlppreprocess.convert_textlines_to_characters.run(
    #         tmp + ".lowercase.tokenize",
    #         tmp + ".lowercase.tokenize.char")
    nlppreprocess.replace_digits.run(
            tmp + ".lowercase.tokenize",
            tmp + ".lowercase.tokenize.replace_digits")
    nlppreprocess.append_eos.run(
            tmp + ".lowercase.tokenize.replace_digits",
            tmp + ".lowercase.tokenize.replace_digits.append_eos")
    nlppreprocess.split_corpus.run(
            tmp + ".lowercase.tokenize.replace_digits.append_eos",
            tmp + ".lowercase.tokenize.replace_digits.append_eos.train",
            tmp + ".lowercase.tokenize.replace_digits.append_eos.val",
            size=5000)
    nlppreprocess.create_vocabulary.run(
            tmp + ".lowercase.tokenize.replace_digits.append_eos.train",
            train + ".vocab",
            prune_at=300000,
            min_count=5,
            special_words=["<EOS>"]) # or ["<EOS>", "<SPACE>", "<EOL>"] in case of char
    nlppreprocess.replace_rare_words.run(
            tmp + ".lowercase.tokenize.replace_digits.append_eos.train",
            train,
            path_vocab=train + ".vocab")
    nlppreprocess.replace_rare_words.run(
            tmp + ".lowercase.tokenize.replace_digits.append_eos.val",
            val,
            path_vocab=train + ".vocab")


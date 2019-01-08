import argparse

import textpreprocessor.replace_digits
import textpreprocessor.append_eos
import textpreprocessor.split_corpus
import textpreprocessor.create_vocabulary
import textpreprocessor.replace_rare_words

def main(args):
    path_tokenized = args.tokenized
    path_train = args.train
    path_dev = args.dev
    path_vocab = args.vocab

    # textpreprocessor.replace_digits.run(
    #         path_tokenized,
    #         path_tokenized + ".replace_digits")
    # textpreprocessor.append_eos.run(
    #         path_tokenized + ".replace_digits",
    #         path_tokenized + ".replace_digits.append_eos")
    # textpreprocessor.split_corpus.run(
    #         path_tokenized + ".replace_digits.append_eos",
    #         path_tokenized + ".replace_digits.append_eos.train",
    #         path_tokenized + ".replace_digits.append_eos.dev",
    #         size=5000)

    textpreprocessor.create_vocabulary.run(
            path_tokenized + ".replace_digits.append_eos.train",
            path_vocab,
            prune_at=300000,
            min_count=5,
            special_words=["<EOS>", "<BOS>"]) # or ["<EOS>", "<SPACE>", "<EOL>"] in case of char

    textpreprocessor.replace_rare_words.run(
            path_tokenized + ".replace_digits.append_eos.train",
            path_train,
            path_vocab=path_vocab)
    textpreprocessor.replace_rare_words.run(
            path_tokenized + ".replace_digits.append_eos.dev",
            path_dev,
            path_vocab=path_vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    args = parser.parse_args()
    main(args)

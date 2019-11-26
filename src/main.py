import warnings
warnings.filterwarnings("ignore")
from bert_dataset import BERTDataset
import os
from tqdm.auto import tqdm
import random
from bert import fit_bert
import logging


import wandb
wandb.init(project="treccar_fit")
config = wandb.config



def main():
    data_home = "/ssd2/arthur/TRECCAR/data"
    tsv_file = "/ssd2/arthur/TRECCAR/data/samples/triples_tokenized.bert"
    train_path = "/".join(tsv_file.split("/")[:-1]+["train-"+tsv_file.split("/")[-1]])
    test_path = "/".join(tsv_file.split("/")[:-1]+["test-"+tsv_file.split("/")[-1]])
    random.seed(config.seed)
    all_docs = []
    for l in tqdm(open(tsv_file)):
        all_docs.append(l)
        test_size = int(config.split_percentage * len(all_docs))
    random.shuffle(all_docs)
    test_set = all_docs[:test_size]
    train_set = all_docs[test_size:]
    assert (len(test_set) + len(train_set)) == len(all_docs)
    train_path = "/".join(tsv_file.split("/")[:-1]+["train-"+tsv_file.split("/")[-1]])
    test_path = "/".join(tsv_file.split("/")[:-1]+["test-"+tsv_file.split("/")[-1]])
    with open(train_path, 'w') as outf:
        for l in train_set:
            outf.write(l)
    with open(test_path, 'w') as outf:
        for l in test_set:
            outf.write(l)
    fit_bert(config, train_path, test_path)


if __name__=="__main__":
    level = logging.getLevelName(config.logging_level)
    Log = logging.getLogger()
    Log.setLevel(level)

    main()

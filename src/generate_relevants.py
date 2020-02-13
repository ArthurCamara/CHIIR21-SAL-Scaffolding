import warnings
warnings.filterwarnings("ignore")
import logging
import os

try:
    assert os.path.isfile("config-defaults.yaml")
    import wandb
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init()
    config = wandb.config
    assert "data_home" in config
except:
    import yaml
    try:
        config = yaml.load(open("config-defaults.yaml"), Loader=yaml.FullLoader)
    except:
        config = yaml.load(open("src/config-defaults.yaml"), Loader=yaml.FullLoader)
    config = {k:v['value'] for (k,v) in config.items()}
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    config = dotdict(config)


import numpy as np
import multiprocessing as mp
from multiprocessing import current_process
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from bert_dataset import BERTDataset
from collections import defaultdict
import pickle
from trec_car import read_data
from indri import generate_index, run_queries, generate_custom_index
from bert_dataset import BERTDataset
from bert import get_scores
import urllib

def process_chunk(docs_path, chunk_no, block_offset, no_lines, config):
    position=chunk_no
    # Load lines
    lines = []
    with open(docs_path, encoding="utf-8") as f:
        f.seek(block_offset[chunk_no])
        for _ in range(no_lines):
            lines.append(f.readline())
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    output_line_format = "{}\t{}\n"
    partial_doc_path = os.path.join(config["data_home"], "tmp", "docs-{}".format(chunk_no))
    partial_doc_path_bert = os.path.join(config["data_home"], "tmp", "docs-{}.bert".format(chunk_no))
    with open(partial_doc_path, 'w', encoding="utf-8") as outf, open(partial_doc_path_bert, 'w', encoding='utf-8') as outf_bert:  # noqa E501
        for line in tqdm(lines, desc="Running for chunk {}".format(chunk_no), position=position):
            try:
                doc_id, text = line.split("\t")
                text = text.replace("\n", "")
            except ValueError:
                continue
            bert_text = [x for x in tokenizer.tokenize(text)]
            tokenized_text = ' '.join(bert_text).replace("##", "")
            outf.write(output_line_format.format(doc_id, tokenized_text))
            outf_bert.write("{}\t{}\n".format(doc_id, bert_text))

def tokenize_docs(docs_path):
    """Tokenize a docs file in tsv format, outputting a tsv. Also generates offset file. Can take a LONG time"""

    assert os.path.isfile(docs_path), "Could not find documents file at {}".format(docs_path)
    if os.path.exists(os.path.join(config.data_home, "docs/full_docs.tsv")) and os.path.exists(os.path.join(config.data_home, "docs/full_docs.tokenized.bert")):
        logging.info("Skipping tokenization. File already exists.")
        return
    # Load in memory, split blocks and run in paralel.
    excess_lines = config.corpus_size % config.number_of_cpus
    number_of_chunks = config.number_of_cpus
    if excess_lines > 0:
        number_of_chunks = config.number_of_cpus - 1
    block_offset = {}
    lines_per_chunk = config.corpus_size // number_of_chunks
    logging.info("Number of lines per CPU chunk: %i", lines_per_chunk)
    if not os.path.isdir(os.path.join(config.data_home, "tmp")):
        os.mkdir(os.path.join(config.data_home, "tmp"))
    if config.number_of_cpus < 2:
        block_offset[0] = 0
    elif not os.path.isfile(os.path.join(config.data_home, "block_offset_{}.pkl".format(config.number_of_cpus))):
        with open(docs_path, encoding="utf-8") as inf:
            current_chunk = 0
            counter = 0
            line = True
            while line:
                if counter % lines_per_chunk == 0:
                    block_offset[current_chunk] = inf.tell()
                    current_chunk += 1
                line = inf.readline()
                counter += 1
        pickle.dump(block_offset, open(os.path.join(config.data_home, "block_offset_{}.pkl".format(config.number_of_cpus)), 'wb'))  # noqa E501
    else:
        block_offset = pickle.load(open(os.path.join(config.data_home, "block_offset_{}.pkl".format(config.number_of_cpus)), 'rb'))  # noqa E501
    assert len(block_offset) == config.number_of_cpus

    if config.number_of_cpus == 1:
        process_chunk(docs_path, 0, block_offset, lines_per_chunk, dict(config))
        return
    pool = mp.Pool(config.number_of_cpus)
    jobs = []
    for i in range(len(block_offset)):
        jobs.append(pool.apply_async(process_chunk, args=(
            docs_path, i, block_offset, lines_per_chunk, dict(config))))
    for job in jobs:
        job.get()
    pool.close()

    with open(os.path.join(config.data_home, "docs/full_docs.tsv"), 'w', encoding="utf-8") as outf:
        for i in tqdm(range(config.number_of_cpus), desc="Merging tsv file"):
            partial_path = os.path.join(config.data_home, "tmp", "docs-{}".format(i))
            for line in open(partial_path, encoding="utf-8"):
                outf.write(line)
            # os.remove(partial_path)

    with open(os.path.join(config.data_home, "docs/full_docs.tokenized.bert"), 'w', encoding="utf-8") as outf:
        for i in tqdm(range(config.number_of_cpus), desc="Merging BERT file"):
            partial_bert_path = os.path.join(config.data_home, "tmp", "docs-{}.bert".format(i))
            for line in open(partial_bert_path, encoding="utf-8"):
                outf.write(line)
            # os.remove(partial_bert_path)

def get_level2(headings, parents, hierarchy=[], depth=1):
    """extract all leaves or level2 headings from an hierarchy"""
    all_headings = []
    for i in headings:
        cannonical_name = parents+"/"+i[0].headingId
        if len(i[1]) < 1 or depth == 2:
            all_headings.append((cannonical_name, i[0].heading, hierarchy + [i[0].heading]))
        all_headings += (get_level2(i[1], cannonical_name, hierarchy + [i[0].heading], depth=depth+1))
    return all_headings

def get_content(doc_id, doc_file, offset_dict):
    offset = offset_dict[doc_id]
    with open(doc_file) as f:
        f.seek(offset)
        doc = f.readline()
    if doc_file.endswith("bert"):
        return eval(doc.split("\t")[1].strip())
    else:
        doc_text = "\t".join(doc.split("\t")[1:])
        return doc_text.strip()

def generate_docs_offset(doc_file):
    offset_path = doc_file + ".offset"
    if os.path.isfile(offset_path):
        logging.info("Already found offset dict at {}. skipping".format(offset_path))
        return pickle.load(open(offset_path, 'rb'))
    offset_dict = dict()
    pbar = tqdm(total=config.corpus_size, desc="Generating doc offset dictionary")
    empty_docs = 0
    with open(doc_file, encoding="utf-8", errors="surrogateescape") as inf:
        location = 0
        line = True
        while line:
            line = inf.readline()
            try:
                doc_id, _ = line.split("\t")
            except (IndexError, ValueError):
                print(line)
                empty_docs +=1
                continue
            offset_dict[doc_id] = location
            location = inf.tell()
            pbar.update()
    # assert len(offset_dict) == (config.corpus_size-empty_docs)
    pickle.dump(offset_dict, open(offset_path, 'wb'))
    print(len(offset_dict))
    return offset_dict

def generate_dataset():
    """Generate all possible triples of subtopic and document
    Add these to a file in the BERTDataset format (subtopic_id-doc_id\t[bert imput list])
    """    
    paragraphs_path = os.path.join(config.raw_data_home, "paragraphCorpus/dedup.articles-paragraphs.cbor")
    docs_path = os.path.join(config.data_home, "docs/docs.tsv")
    if not os.path.isfile(docs_path):
        with open(docs_path, 'w', encoding="utf-8") as outf:
            for paragraph in tqdm(read_data.iter_paragraphs(open(paragraphs_path, 'rb')), desc="Dumping paragraphs in tsv", total=config.corpus_size):
                text = paragraph.get_text().replace("\n", " ").replace("\t", " ")
                paragraph_id = paragraph.para_id
                outf.write("{}\t{}\n".format(paragraph_id, text))
    tokenize_docs(docs_path)

    # Load and tokenize topics
    topics_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-outlines.cbor")
    topics_to_use = []
    
    non_tokenized_queries = dict()
    for page in tqdm(read_data.iter_annotations(open(topics_path, 'rb')), total=65, desc="Reading and tokening topics"):
        # If the topic does not have hierarchy, we don't care about it
        if len(page.nested_headings()) == len(page.flat_headings_list()):
            continue
        topics_to_consider = get_level2(page.deep_headings_list(), page.page_id, hierarchy=[page.page_name])
        topics_to_use += topics_to_consider
        for topic_id, _, hierarchy in topics_to_consider:
            query = " ".join(hierarchy)
            non_tokenized_queries[topic_id] = query
    # Load relevants based on qrels
    relevant_per_doc = defaultdict(lambda:set())

    qrel_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-hierarchical.qrels")
    for line in open(qrel_path):
        topic, _, doc, _ = line.split(" ")
        hierarchy = urllib.parse.unquote(topic).split("/")
        if len(hierarchy) >= 2:
            cannonical_id = "/".join(topic.split("/")[:3])
            relevant_per_doc[doc].add(cannonical_id)

    doc_format = """<DOC>
    <DOCNO>{}</DOCNO>
    <text>{}</text>
    {}
    </DOC>\n"""
    
    
    full_docs_offset = generate_docs_offset(docs_path)
    final_docs_path = os.path.join(config.data_home, "docs/docs_with_relevance.trec")
    with open(final_docs_path, 'w') as outf:
        for _, doc in tqdm(enumerate(full_docs_offset), desc="dumping trec docs with scores", total = len(full_docs_offset)):
            # doc_scores = []
            relevant_docs_format ="<relevants>{}</relevants>"
            doc_text = get_content(doc,docs_path, full_docs_offset)
            relevant_topics = list(relevant_per_doc[doc])
            if len(relevant_topics) > 0:
                relevants = relevant_docs_format.format(",".join(relevant_topics))
            else:
                relevants  = relevant_docs_format.format(" ")
            doc_full = doc_format.format(doc, doc_text, relevants)

                # break
    generate_custom_index(final_docs_path, "docs_with_relevants")


def get_all_relevant_docs():
    """ Return relevant docs to all queries"""
    docs_path = os.path.join(config.data_home, "docs/clean_docs.tsv")
    dict_offset = generate_docs_offset(docs_path)
    qrel_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-hierarchical.qrels")
    relevants = []
    for line in open(qrel_path):
        _, _, doc_id, _ = line.split(" ")
        doc_text = get_content(doc_id, docs_path, dict_offset)
        relevants.append((doc_id, doc_text))
    return relevants
        

    





if __name__=="__main__":
    level = logging.getLevelName(config.logging_level)
    Log = logging.getLogger()
    Log.setLevel(level)
    generate_index()

    # generate_dataset()


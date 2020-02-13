import warnings
import logging
import os
from trec_car import read_data
from tqdm.auto import tqdm
from generate_relevants import get_level2, generate_docs_offset
from collections import defaultdict
import json
import urllib.parse
import numpy as np
from pprint import pprint as print

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

def main():
    topic_id_mapping = {}
    # topics_path  = os.path.join(config.raw_data_home, "benchmarkY2.public/benchmarkY2.cbor-outlines.cbor")
    topics_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-outlines.cbor")
    topics_to_use = []
    topic_id = 0
    topics_dict = {
            "0":  {
            "title" : "Sports",
            "description" : "Imagine you are taking an introductory course on Sports. For your term paper, you have decided to write about <b>Sports Development and Coaching </b>. ",
            "terms" : {
                "olympics": ["subtopic1", "subtopic2", "subtopic3"],
                "weight lifting": ["subtopic1", "subtopic2", "subtopic3"],
                "karate": ["subtopic1", "subtopic2", "subtopic3"],
                "martial art": ["subtopic1", "subtopic2", "subtopic3"],
                "aerobics": ["subtopic1", "subtopic2", "subtopic3"],
                "athletes": ["subtopic1", "subtopic2", "subtopic3"],
                "soccer": ["subtopic1", "subtopic2", "subtopic3"],
                "baseball": ["subtopic1", "subtopic2", "subtopic3"],
                "snowboarding": ["subtopic1", "subtopic2", "subtopic3"],
                "hockey": ["subtopic1", "subtopic2", "subtopic3"]
            }
        }
    }
    for page in tqdm(read_data.iter_annotations(open(topics_path, 'rb')), total=117):
        if len(page.nested_headings()) == len(page.flat_headings_list()):
            continue
        topic_id += 1
        if page.page_name not in config.topics:
            continue
        topics_dict[str(topic_id)] = {
                "title": page.page_name,
                "description": "Imagine you are taking an introductory course on {0} this term. For your final paper, you need to write about <b>{0} </b>".format(page.page_name),
                "youtube": ""
            }
        topic_id_mapping[page.page_name] = str(topic_id)
        terms = defaultdict(lambda:[])
        topics_to_consider = get_level2(page.deep_headings_list(), page.page_id, hierarchy=[page.page_name])
        topics_to_use += topics_to_consider
        for _, _, hierarchy in topics_to_consider:
            topic_name  = urllib.parse.unquote(hierarchy[0]).replace("enwiki:", "")
            if topic_name not in config.topics:
                continue
            if len(hierarchy) == 3:
                terms[hierarchy[1]].append(hierarchy[2])
            elif len(hierarchy) == 2:
                terms[hierarchy[1]] = []
            elif len(hierarchy) > 3:
                if hierarchy[2] not in terms[hierarchy[1]]:
                    terms[hierarchy[1]].append(hierarchy[2])
            else:
                print(hierarchy)
        terms = dict(terms)
        topics_dict[str(topic_id)]["terms"] = terms
        topics_dict[str(topic_id)]["docs"] = {}


    # Get relevant docs for each topic

    # qrel_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-hierarchical.qrels")
    qrel_path = os.path.join(config.data_home, "topics/expanded_relevants.qrel")
    for line in open(qrel_path):
        topic, _, _, _ = line.split(" ")
        hierarchy = topic.split("/")
        if len(hierarchy) >= 2: 
            topic_name  = urllib.parse.unquote(hierarchy[0]).replace("enwiki:", "")
            if topic_name not in config.topics:
                continue
            try:
                topic_id = topic_id_mapping[topic_name]
            except:
                continue
            try:
                unit_id = hierarchy[2]
            except IndexError:
                unit_id = None
            subtopic_id = hierarchy[1]
            if subtopic_id not in topics_dict[topic_id]["docs"]:
                topics_dict[topic_id]["docs"][subtopic_id] = {}
                topics_dict[topic_id]["docs"][subtopic_id]["_count"] = 0
            topics_dict[topic_id]["docs"][subtopic_id]["_count"] += 1
            if "_units" not in topics_dict[topic_id]["docs"][subtopic_id]:
                topics_dict[topic_id]["docs"][subtopic_id]["_units"] = {}
            if unit_id is not None:
                if unit_id not in topics_dict[topic_id]["docs"][subtopic_id]["_units"] :
                    topics_dict[topic_id]["docs"][subtopic_id]["_units"] [unit_id]  = 0
                topics_dict[topic_id]["docs"][subtopic_id]["_units"] [unit_id]  += 1




    with open(os.path.join(config.data_home, "topics/topics.json"), 'w') as outf:
        json.dump(topics_dict, outf, indent=2)
    
if __name__ == "__main__":
    main()
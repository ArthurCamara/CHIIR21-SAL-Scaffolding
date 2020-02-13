from bert_dataset import BERTDataset
import warnings
warnings.filterwarnings("ignore")
from transformers import DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score, accuracy_score, roc_auc_score
from collections import defaultdict
import subprocess
import os
import random
import torch
import math
import logging
import wandb
torch.multiprocessing.set_start_method('fork', force=True)
logging.getLogger("transformers").setLevel(logging.WARNING)
config = wandb.config

def init_optimizer(
        model: DistilBertForSequenceClassification,
        n_steps: int,
        lr: float,
        warmup_proportion: float = 0.1,
        weight_decay: float = 0.0):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    warmup_steps = n_steps * warmup_proportion
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=n_steps)
    return optimizer, scheduler


def fit_bert(config, train_triples_path, dev_triples_path):
    # Set dataset
    train_dataset = BERTDataset(train_triples_path, config.data_home, invert_label=True)
    dev_dataset = BERTDataset(dev_triples_path, config.data_home, invert_label=True)

    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Set CUDA
    model = DistilBertForSequenceClassification.from_pretrained(config.bert_class)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        visible_gpus = list(range(torch.cuda.device_count()))
        for _id in config.ignore_gpu_ids:
            visible_gpus.remove(_id)
        logging.info("Running with gpus {}".format(visible_gpus))
        device = torch.device("cuda:{}".format(visible_gpus[0]))
        model = torch.nn.DataParallel(model, device_ids=visible_gpus)
        model.to(device)
        train_batch_size = len(visible_gpus) * config.batch_per_device
        logging.info("Effective train batch size of %d", train_batch_size)
    else:
        device = torch.device("cpu")
        model.to(device)
    logging.info("Using device: %s", str(device))
    wandb.watch(model)

    data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=config.number_of_cpus,
        shuffle=True)
    num_train_optimization_steps = len(data_loader) * config.n_epochs
    optimizer, scheduler = init_optimizer(model, num_train_optimization_steps, config.learning_rate)
    logging.info("******Started trainning******")
    logging.info("   Total optmization steps %d", num_train_optimization_steps)

    global_step = 0
    tr_loss = logging_loss = 0.0
    model.zero_grad()
    for _ in tqdm(range(config.n_epochs), desc="Epochs"):
        for step, batch in tqdm(enumerate(data_loader), desc="Batches", total=len(data_loader)):
            model.train()
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'labels': batch[3].to(device)}
            outputs = model(**inputs)
            loss = outputs[0]
            loss = loss.mean()
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                wandb.log({
                    "Train Loss": loss.item(),
                    "Leaning Rate": scheduler.get_lr()[0]})

            global_step += 1
            if global_step % config.train_loss_print == 0:
                logits = outputs[1]
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy().flatten()
                logging.info("Train ROC: {}".format(roc_auc_score(out_label_ids, preds[:, 1])))
                preds = np.argmax(preds, axis=1)
                logging.info("Train accuracy: {}".format(
                    accuracy_score(out_label_ids, preds)))
                logging.info("Training loss: {}".format(
                    loss.item()))

            if global_step % config.eval_steps == 0:
                evaluate(dev_dataset,
                         config.data_home,
                         model,
                         device,
                         global_step,
                         eval_batchsize=config.eval_batchsize,
                         n_workers=config.number_of_cpus,
                         sample=config.eval_sample)
                # Save intermediate model
                output_dir = os.path.join(config.data_home, "checkpoints/checkpoint-{}".format(global_step))
                logging.info("Saving model checkpoint to %s", output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
    output_dir = os.path.join(config.data_home, "models/{}-treccar".format(config.bert_class))
    if not os.path.isdir(os.path.join(config.data_home, "models")):
        os.makedirs(os.path.join(config.data_home, "models"))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    return model


def evaluate(eval_dataset: BERTDataset,
             output_dir: str,
             model,
             device: str,
             global_step: int,
             eval_batchsize=32,
             n_workers=0,
             sample=1.0):
    starting_index = 0
    max_feasible_index = len(eval_dataset) - math.floor(len(eval_dataset) * sample)
    if max_feasible_index > 0:
        starting_index = random.choice(range(max_feasible_index))
    ending_index = starting_index + math.floor(len(eval_dataset) * sample)
    eval_dataloader = DataLoader(
        eval_dataset[starting_index:ending_index], batch_size=eval_batchsize, shuffle=False, num_workers=n_workers)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Eval batch"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[3].to(device)}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy().flatten()
            else:
                batch_predictions = logits.detach().cpu().numpy()
                preds = np.append(preds, batch_predictions, axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy().flatten(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
    results = {}
    results["ROC Dev"] = roc_auc_score(out_label_ids, preds[:, 1])
    preds = np.argmax(preds, axis=1)
    results["Acuracy Dev"] = accuracy_score(out_label_ids, preds)
    results["F1 Dev"] = f1_score(out_label_ids, preds)
    results["AP Dev"] = average_precision_score(out_label_ids, preds)
    logging.info("***** Eval results *****")
    wandb.log(results)
    for key in sorted(results.keys()):
        logging.info("  %s = %s", key, str(results[key]))
    
def get_scores(samples_path, config):
    preds_out_file = os.path.join(config.data_home, "predictions/")
    preds_out_file = os.path.join(preds_out_file, "trecccar-top{}.tensor".format(config.possible_relevants))
    if "bert" in config.force_steps or not os.path.isfile(preds_out_file): 
        dataset = BERTDataset(samples_path, config.data_home, invert_label=True, labeled=False)
        
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        model_path = os.path.join(config.data_home, "models/distilbert-base-uncased-treccar")

        # Set CUDA
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            visible_gpus = list(range(torch.cuda.device_count()))
            for _id in config.ignore_gpu_ids:
                visible_gpus.remove(_id)
            logging.info("Running with gpus {}".format(visible_gpus))
            device = torch.device("cuda:{}".format(visible_gpus[0]))
            model = torch.nn.DataParallel(model, device_ids=visible_gpus)
            model.to(device)
            train_batch_size = len(visible_gpus) * config.batch_per_device
            logging.info("Effective train batch size of %d", train_batch_size)
        else:
            device = torch.device("cpu")
            model.to(device)
        logging.info("Using device: %s", str(device))
        # wandb.watch(model)    
        
        preds_out_file = os.path.join(config.data_home, "predictions/")
        if not os.path.isdir(preds_out_file):
            os.makedirs(preds_out_file)
        preds_out_file = os.path.join(preds_out_file, "trecccar-top{}.tensor".format(config.possible_relevants))
        
        batch_size = len(visible_gpus) * config.batch_per_device * 16
        
        nb_eval_steps = 0
        preds = None
        _preds = None

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False)

        for  batch in tqdm(data_loader, desc="Batches", total=len(data_loader)):
            model.eval()
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device)}
                outputs = model(**inputs)
                logits = outputs[0]
                nb_eval_steps += 1
                if _preds is None:
                    _preds = logits.detach().cpu().numpy()
                else:
                    batch_predictions = logits.detach().cpu().numpy()
                    _preds = np.append(_preds, batch_predictions, axis=0)
        torch.save(_preds, preds_out_file)
        softmax = torch.nn.Softmax(dim=1)
        preds = softmax(torch.as_tensor(_preds))[:, 0].cpu().numpy()
        torch.save(preds, preds_out_file+".softmax")
    
    # Load QL scores and normalize
    indri_run_file = os.path.join(config.data_home, "runs/QL.run")
    current_topic = None    
    current_topic_scores = []
    QL_scores = []
    all_ids = []
    for line in tqdm(open(indri_run_file), desc="loading QL scores"):
        topic, _, doc, _, score, _ =  line.split()
        score = float(score)
        if topic != current_topic and current_topic is not None:
            QL_scores += list((current_topic_scores - np.min(current_topic_scores))/np.ptp(current_topic_scores))
            current_topic_scores = []
        current_topic_scores.append(score)
        all_ids.append((topic, doc))
        current_topic = topic
    # Do the last one
    QL_scores += list((current_topic_scores - np.min(current_topic_scores))/np.ptp(current_topic_scores))

    #Load BERT scores
    preds = torch.load(preds_out_file+".softmax")
    assert len(preds) == len(QL_scores)
    #Combine scores
    doc_scores = defaultdict(lambda:defaultdict(lambda:0.0))
    for (topic_id, doc_id), QL_score, BERT_score in tqdm(zip(all_ids, QL_scores, preds), desc="Computing final scores", total=len(QL_scores)):
        beta = 1 - config.bert_alpha
        final_score = config.bert_alpha * BERT_score + beta * QL_score
        doc_scores[doc_id][topic_id] = final_score   
    return doc_scores

    






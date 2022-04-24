from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
from typing import Dict, List
from datasets import DatasetDict, load_dataset, load_metric, Dataset

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, pipeline
from tqdm.auto import tqdm
import pandas as pd


def read_data(tag2idx:dict() = {}, idx2tag:dict() = {}):
    
    data_tokens = []
    data_tags = []
    data_id = []
    
    with open(args.train_file) as f:
        datas = json.load(f)
        for data in datas:
            data_tokens.append(data['tokens'])
            data_tags.append(data['tags'])
            data_id.append(data['id'])
    
    with open(args.valid_file) as f:
        datas = json.load(f)
        for data in datas:
            data_tokens.append(data['tokens'])
            data_tags.append(data['tags'])
            data_id.append(data['id'])
    
    idx = 0
    for tags in data_tags:
        for tag in tags:
            if tag in tag2idx:
                continue
            else:
                tag2idx[tag] = idx
                idx2tag[idx] = tag
                idx += 1
    
    for idx, tags in enumerate(data_tags):
        labels = []
        for tag in tags:
            labels.append(tag2idx[tag])
        data_tags[idx] = labels

    return tag2idx, idx2tag


def read_dataset(tag2idx:dict()):
    # load train and valid json
    dataset_dict = dict()
    with open(args.train_file, encoding="utf-8") as f:
        tmp_json = json.load(f)
        
        for idx, datas in enumerate(tmp_json):
            tag_ids = [tag2idx[tag] for tag in datas["tags"]]
            tmp_json[idx]["tags"] = tag_ids
        
        pd_dict_train = pd.DataFrame.from_dict(tmp_json)

    with open(args.valid_file, encoding="utf-8") as f:
        tmp_json = json.load(f)
        
        for idx, datas in enumerate(tmp_json):
            tag_ids = [tag2idx[tag] for tag in datas["tags"]]
            tmp_json[idx]["tags"] = tag_ids
        
        pd_dict_val = pd.DataFrame.from_dict(tmp_json)

    pd_dataset_train = Dataset.from_pandas(pd_dict_train)
    pd_dataset_val = Dataset.from_pandas(pd_dict_val)

    dataset_dict["train"] = pd_dataset_train
    dataset_dict["eval"] = pd_dataset_val
    dataset = DatasetDict(dataset_dict)
    return dataset


def read_dataset_test():
    # load train and valid json
    dataset_dict = dict()
    with open(args.test_file, encoding="utf-8") as f:
        tmp_json = json.load(f)
        pd_dict = pd.DataFrame.from_dict(tmp_json)

    pd_dataset = Dataset.from_pandas(pd_dict)

    # dataset_dict["test"] = pd_dataset
    # dataset = DatasetDict(dataset_dict)
    return pd_dataset


def get_labels(predictions, references, label_list):
    # Transform predictions and references tensos to numpy arrays
    if args.device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels


def compute_metrics(metric):
    results = metric.compute()
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def preprocess(datasets, tokenizer, label_all_tokens=True):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=args.max_length, padding=False)

        labels = []
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    processed_dataset = datasets.map(tokenize_and_align_labels, batched=True, remove_columns=datasets.column_names)
    return processed_dataset

def preprocess_test(dataset, tokenizer):
    class ADLDataset(Dataset):
        def __init__(self, tokenized_dict, id, length):
            self.input_ids = tokenized_dict['input_ids']
            self.token_type_ids = tokenized_dict['token_type_ids']
            self.attention_mask = tokenized_dict['attention_mask']
            self.id = id
            self.length = length
            
        def __getitem__(self, idx):
            input_id = self.input_ids[idx]
            tokentype = self.token_type_ids[idx]
            attentionmask = self.attention_mask[idx]
            id = self.id[idx]
            length = self.length[idx]
            
            return input_id, tokentype, attentionmask, id, length
        
        def __len__(self):
            return len(self.input_ids)
    
    tokenized_inputs = tokenizer(dataset["tokens"], truncation=True, is_split_into_words=True, max_length=args.max_length, padding=True, return_tensors = 'pt')
    # processed_dataset = ADLDataset(tokenized_inputs, )
    length = [len(token) for token in dataset["tokens"]]
    processed_dataset = ADLDataset(tokenized_inputs, dataset["id"], length)
    
    return processed_dataset


def train():
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tag2idx, idx2tag = read_data()
    datasets = read_dataset(tag2idx)
    
    train_raw_dataset = datasets["train"]
    eval_raw_dataset = datasets["eval"]
    
    train_dataset = preprocess(train_raw_dataset, tokenizer)
    eval_dataset = preprocess(eval_raw_dataset, tokenizer)
    

    # Dataloader
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)
    
    
    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, num_labels=9)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_step = len(train_dataset) * args.num_epoch // (args.batch_size * args.accum_steps)
    warmup_step = total_step * 0.06
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, total_step)
    
    metric = load_metric("seqeval")
    
    print("Start Training")
    best_dev_loss = float('inf')
    for epoch in range(args.num_epoch):
        model.train()
        print(f"\nEpoch: {epoch+1} / {args.num_epoch}")
        for batch_step, batch_datas in enumerate(tqdm(train_dataloader, desc="Train")):
            # input_ids, token_type_ids, attention_mask, labels = batch_datas.values()
            input_ids, token_type_ids, attention_mask, labels = [b_data.to(args.device) for b_data in batch_datas.values()]

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # train_loss += loss.detach().float()
            loss = loss / args.accum_steps
            loss.backward()

            if batch_step % args.accum_steps == 0 or batch_step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        dev_loss = 0.0
        for batch_step, batch_datas in enumerate(tqdm(eval_dataloader, desc="eval")):
            with torch.no_grad():
                # input_ids, token_type_ids, attention_mask, labels = batch_datas.values()
                input_ids, token_type_ids, attention_mask, labels = [b_data.to(args.device) for b_data in batch_datas.values()]

                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                dev_loss += loss.detach().item()
                
                preds = outputs.logits.argmax(dim=-1).cpu()
                labels = labels.cpu()
                
                preds, refs = get_labels(preds, labels, idx2tag)
                metric.add_batch(
                    predictions=preds,
                    references=refs,
                )
        
        eval_metric = compute_metrics(metric)
        print(eval_metric)
        print(dev_loss)
        
        if best_dev_loss > dev_loss:
            best_dev_loss = dev_loss
            if args.model_path != None:
                model.save_pretrained(args.model_path)


def test2():
    tag2idx, idx2tag = read_data()
    dataset = read_dataset_test()
    
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model = model.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    token_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer, device=0)
    ans = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    
    
    tokens = [''.join(data + " " for data in datas).strip() for datas in dataset["tokens"] ]
    ids = dataset["id"]
    
    print(tokens[:3])
    
    answers = token_classifier(tokens)
    pred_lbl = []
    for ans in answers:
        lbls = ""
        for a in ans:
            lbl = int(a["entity"].split("_")[-1])
            lbls += idx2tag[lbl] + " "
        lbls = lbls.strip()
        pred_lbl.append(lbls)
    print(pred_lbl[:3])
    
    ans_dict = {"id": ids, "tags": pred_lbl}
    pd.DataFrame.from_dict(ans_dict).to_csv(args.pred_file, index=False)


def test():
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tag2idx, idx2tag = read_data()
    dataset = read_dataset_test()
    
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model = model.to(args.device)
    model.eval()
    
    # preprocess
    dataset = preprocess_test(dataset, tokenizer)
    
    # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)
    
    pred_lbls = []
    pred_id = []
    with torch.no_grad():
        for batch_idx, batch_datas in enumerate(tqdm(test_dataloader)):
            input_ids, token_type_ids, attention_mask, id, length = batch_datas
            input_ids, token_type_ids, attention_mask = input_ids.to(args.device), token_type_ids.to(args.device), attention_mask.to(args.device)

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            
            # print(preds)
            # continue
            # break
            
            for idx, pred in enumerate(preds):
                pred_label = [idx2tag[p] + " " for i, p in enumerate(pred) if length[idx] > i]
                pred_lbl = ''.join(pred_label).strip()
                
                pred_lbls.append(pred_lbl)
                pred_id.append(id[idx])
    
    # write prediction to file (args.pred_file)
    ans_dict = {"id": pred_id, "tags": pred_lbls}
    pd.DataFrame.from_dict(ans_dict).to_csv(args.pred_file, index=False)
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="model name or path",
        default="bert-base-cased",
        # default="hfl/chinese-bert-wwm-ext",
        # default="hfl/chinese-macbert-base",
        # default="hfl/chinese-roberta-wwm-ext"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="Tokenizer name",
        default="bert-base-cased",
        # default="hfl/chinese-bert-wwm-ext",
        # default="hfl/chinese-macbert-base",
        # default="hfl/chinese-roberta-wwm-ext"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Directory to save the model.",
        default="./ckpt/slot/models/",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        help="Path to save the tokenizer.",
        default="./ckpt/tokenizer/QA",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        help="Context json file",
        default="./data/slot/train.json",
    )
    parser.add_argument(
        "--valid_file",
        type=Path,
        help="Validation json file",
        default="./data/slot/eval.json",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Test json file",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Test json file",
        default="./pred3.csv",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        help="Directory to store EM Loss Curve.",
        default="./plot/"
    )
        
    
    # data
    parser.add_argument("--max_length", type=int, default=27, help="The maximum length of a feature (question and context)")
    parser.add_argument("--doc_stride", type=int, default=192, help="The authorized overlap between two part of the context when splitting it is needed.")

    # model
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=10)
    
    # post processing
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="The threshold used to select the null answer: if the best answer has a score that is less than "
        "the score of the null answer minus this threshold, the null answer is selected for this example. "
        "Only useful when `version_2_with_negative=True`.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=50,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.model_path.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    
    # train()
    # test()
    test2()

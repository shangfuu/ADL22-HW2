from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Union
import os
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json
import math

from datasets import DatasetDict, load_dataset, Dataset
from accelerate import Accelerator
from transformers import (AutoTokenizer,
                          TrainingArguments, 
                          default_data_collator, 
                          get_cosine_schedule_with_warmup,
                          SchedulerType)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess(dataset, context):
    def preprocess_function(examples):
        """
        The preprocessing function needs to do:

        1. Make four copies of the sent1 field so you can combine each of them with sent2 to 
        recreate how a sentence starts.
        2. Combine sent2 with each of the four possible sentence endings.
        3. Flatten these two lists so you can tokenize them, and then unflatten them afterward 
        so each example has a corresponding input_ids, attention_mask, and labels field.
        """
        first_sentences = [[question]*4 for question in examples["question"]]
        # question_headers = examples["sent2"]
        # second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
        second_sentences = [[context[idx] for idx in idxs] for idxs in examples["paragraphs"]]
        labels = [paragraph.index(examples["relevant"][idx]) for idx, paragraph in enumerate(examples["paragraphs"])]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_examples = tokenizer(first_sentences, second_sentences, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=True, max_length=args.max_length)
        tokenized_inputs = {k:[v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    return tokenizer, tokenized_dataset


def read_dataset():
    # Get Context
    context_file = str((args.data_dir / "context.json").resolve())
    with open(context_file, encoding="utf-8") as f:
        context = json.load(f)
    
    # load train and valid json
    tmp_files = ["train.json", "valid.json"]
    dataset_dict = dict()
    for file in tmp_files:
        tmp = str((args.data_dir / file).resolve())
        with open(tmp, encoding="utf-8") as f:
            tmp_json = json.load(f)
            pd_dict = pd.DataFrame.from_dict(tmp_json)

        pd_dataset = Dataset.from_pandas(pd_dict)
        name = file.split(".")[0]
        dataset_dict[name] = pd_dataset
    
    dataset = DatasetDict(dataset_dict)
    return context, dataset


def train():
    # swag = load_dataset('swag', 'regular')
    
    context, dataset = read_dataset()
    
    # preprocess
    tokenizer, processed_datasets = preprocess(dataset, context)
    print(processed_datasets["train"][0])
    # print(len(tokenized_dataset["train"][0]["attention_mask"][0]))
    # print(len(tokenized_dataset["train"][0]["token_type_ids"][0]))
    # print(len(tokenized_dataset["train"][0]["input_ids"][0]))

    tokenizer.save_pretrained("./models/tokenizer/")
    # tokenizer2 = AutoTokenizer.from_pretrained("./models/tokenizer/")


    # Dataset and Dataloader
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["valid"]
    
    data_collator = default_data_collator
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)
    
    # Train
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path,)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    # optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # print(len(train_dataloader))
    print(len(train_dataset))
    total_step = len(train_dataset) * args.num_epoch // (args.batch_size * args.accum_steps)
    warmup_step = total_step * 0.06
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_step, total_step)
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, scheduler)
    
    best_dev_loss = 1e10
    print("Start Training")
    for epoch in range(args.num_epoch):
        model.train()
        
        print(f"\nEpoch: {epoch+1} / {args.num_epoch}")
        train_loss, train_acc = 0, 0
        for batch_step, batch_data in enumerate(tqdm(train_dataloader, desc="Train")):
            input_ids, token_type_ids, attention_mask, labels = batch_data.values()
            # outputs = model(**batch_data)
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            train_loss += loss.detach().float()
            loss = loss / args.accum_steps
            
            # accelerator.backward(loss)
            loss.backward()
            
            if batch_step % args.accum_steps == 0 or batch_step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            predictions = outputs.logits.argmax(dim=-1)
            train_acc += (predictions == labels).cpu().sum().item()
        
        train_loss /= (batch_step * args.accum_steps)
        train_acc /= len(train_dataset)

        model.eval()
        dev_acc, dev_loss = 0, 0
        for batch_step, batch_data in enumerate(tqdm(eval_dataloader, desc="Valid")):
            with torch.no_grad():
                input_ids, token_type_ids, attention_mask, labels = batch_data.values()

                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
                dev_loss += loss.detach().float()

                predictions = outputs.logits.argmax(dim=-1)
                print("pred", predictions)
                print("labels", labels)

                dev_acc += (predictions == labels).cpu().sum().item()

        dev_loss /= (batch_step * args.accum_steps)
        dev_acc /= len(eval_dataset)
        
        print(f"TRAIN LOSS:{train_loss} ACC:{train_acc}  | EVAL LOSS:{dev_loss} ACC:{dev_acc}")

        if dev_loss < best_dev_loss:
            best_dev_loss = best_dev_loss
            # best_state_dict = deepcopy(model.state_dict())
            if args.ckpt_path is not None:
                model.save_pretrained(args.ckpt_path)

    
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="model name or path",
        default="bert-base-chinese"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="Tokenizer name",
        default="bert-base-chinese"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the checkpoints.",
        default="./.ckpt/MC/models/",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        help="Path to save the tokenizer.",
        default="./.ckpt/MC/tokenizer/",
    )
    
    
    # data
    parser.add_argument("--max_length", type=int, default=384, help="Tokenize max length")

    # model
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    args.ckpt_path.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()
    
    train()
    
    
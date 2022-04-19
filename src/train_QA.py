from copy import deepcopy
from dataclasses import dataclass, field
from tabnanny import process_tokens
from typing import Optional, Union
import os
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json
import math

from datasets import DatasetDict, load_dataset, Dataset
from accelerate import Accelerator
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          default_data_collator, 
                          get_cosine_schedule_with_warmup,
                          SchedulerType, get_linear_schedule_with_warmup)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import pandas as pd


def preprocess(dataset, context):
    def preprocess_function(examples):

        questions = [q.lstrip() for q in examples["question"]]
        contexts = [context[rel] for rel in examples["relevant"]]
        
        tokenized_examples = tokenizer(
            questions,
            contexts,
            max_length=args.max_length,
            stride=args.doc_stride,
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            padding=True,
        )
        
        offset_mapping = tokenized_examples.pop("offset_mapping")
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # label those examples
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answer"][sample_index]
            # If no answers are given, set the cls_index as answer.
            # if len(answers["start"]) == 0:
            if not isinstance(answers["start"], int):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["start"]
                end_char = start_char + len(answers["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    
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
    # print(processed_datasets)

    tokenizer.save_pretrained(args.tokenizer_path)

    # Dataset and Dataloader
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["valid"]

    data_collator = default_data_collator
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)
    
    # Train
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path,)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    ## optimizer
    ## Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    total_step = len(train_dataset) * args.num_epoch // (args.batch_size * args.accum_steps)
    warmup_step = total_step * 0.06
    
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, total_step)
    # model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, scheduler)
    
    best_dev_loss = 1e10
    print("Start Training")
    for epoch in range(args.num_epoch):
        model.train()
        
        print(f"\nEpoch: {epoch+1} / {args.num_epoch}")
        train_loss, train_acc = 0, 0
        for batch_step, batch_datas in enumerate(tqdm(train_dataloader, desc="Train")):
            # input_ids, token_type_ids, attention_mask, labels = batch_datas.values()
            input_ids, token_type_ids, attention_mask, start_pos, end_pos = [b_data.to(args.device) for b_data in batch_datas.values()]

            # outputs = model(**batch_data)
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
            loss = outputs.loss
            
            train_loss += loss.detach().float()
            loss = loss / args.accum_steps
            
            # accelerator.backward(loss)
            loss.backward()

            # Choose the most probable start position / end position
            start_index = torch.argmax(outputs.start_logits, dim=1)
            end_index = torch.argmax(outputs.end_logits, dim=1)

            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == start_pos) & (end_index == end_pos)).float().sum()

            if batch_step % args.accum_steps == 0 or batch_step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        train_loss /= (batch_step * args.accum_steps)
        train_acc /= len(train_dataset)
        
        model.eval()
        dev_acc, dev_loss = 0, 0
        for batch_step, batch_datas in enumerate(tqdm(eval_dataloader, desc="Valid")):
            with torch.no_grad():
                # input_ids, token_type_ids, attention_mask, labels = batch_data.values()
                input_ids, token_type_ids, attention_mask, start_pos, end_pos = [b_data.to(args.device) for b_data in batch_datas.values()]

                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
                loss = outputs.loss
                
                dev_loss += loss.detach().float()

                # Choose the most probable start position / end position
                start_index = torch.argmax(outputs.start_logits, dim=1)
                end_index = torch.argmax(outputs.end_logits, dim=1)

                # Prediction is correct only if both start_index and end_index are correct
                dev_acc += ((start_index == start_pos) & (end_index == end_pos)).float().sum()
                # print(dev_acc)
                # print(((start_index == start_pos) & (end_index == end_pos)).float().mean())


        dev_loss /= (batch_step * args.accum_steps)
        dev_acc /= len(eval_dataset)
        
        print(f"TRAIN LOSS:{train_loss} ACC:{train_acc}  | EVAL LOSS:{dev_loss} ACC:{dev_acc}")

        if dev_loss < best_dev_loss:
            best_dev_loss = best_dev_loss
            # best_state_dict = deepcopy(model.state_dict())
            if args.model_path is not None:
                model.save_pretrained(args.model_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="model name or path",
        # default="bert-base-chinese",
        # default="hfl/chinese-bert-wwm-ext",
        # default="hfl/chinese-macbert-base",
        default="hfl/chinese-roberta-wwm-ext"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="Tokenizer name",
        # default="bert-base-chinese",
        # default="hfl/chinese-bert-wwm-ext",
        # default="hfl/chinese-macbert-base",
        default="hfl/chinese-roberta-wwm-ext"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Directory to save the model.",
        default="./ckpt/QA/models/",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        help="Path to save the tokenizer.",
        default="./ckpt/tokenizer/QA",
    )
    
    # data
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of a feature (question and context)")
    parser.add_argument("--doc_stride", type=int, default=192, help="The authorized overlap between two part of the context when splitting it is needed.")

    # model
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    # parser.add_argument(
    #     "--max_train_steps",
    #     type=int,
    #     default=None,
    #     help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    # )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    args.model_path.mkdir(parents=True, exist_ok=True)

    # accelerator = Accelerator()
    
    train()
    
    
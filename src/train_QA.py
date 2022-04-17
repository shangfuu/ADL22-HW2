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
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          TrainingArguments, 
                          default_data_collator, 
                          get_cosine_schedule_with_warmup,
                          SchedulerType, get_linear_schedule_with_warmup)
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


def preprocess(dataset, context):
    def preprocess_function(examples):

        questions = [q.strip() for q in examples["question"]]
        contexts = [context[rel] for rel in examples["relevant"]]
        inputs = tokenizer(
            questions,
            contexts,
            max_length=args.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding=True,
            stride=args.doc_stride,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answer"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["start"]
            end_char = answer["start"] + len(answer["text"])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
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
    # print(len(tokenized_dataset["train"][0]["attention_mask"][0]))
    # print(len(tokenized_dataset["train"][0]["token_type_ids"][0]))
    # print(len(tokenized_dataset["train"][0]["input_ids"][0]))

    tokenizer.save_pretrained(args.tokenizer_path)
    # tokenizer2 = AutoTokenizer.from_pretrained("./models/tokenizer/")

    # Dataset and Dataloader
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["valid"]

    # input_ids = tokenizer.decode(train_dataset[0]["input_ids"])
    # print(input_ids)
    # start = train_dataset[0]["start_positions"]
    # end = train_dataset[0]["end_positions"]
    # print("answer:", tokenizer.decode(train_dataset[0]["input_ids"][start:end+1]))
    # print(start, end)
    # return
    
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
            train_acc += ((start_index == start_pos) & (end_index == end_pos)).float().mean()
            
            
            if batch_step % args.accum_steps == 0 or batch_step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        print("ACC1:", train_acc / len(train_dataset))
        print("ACC2:", train_acc / (batch_step * args.accum_steps))
        train_loss /= (batch_step * args.accum_steps)
        train_acc /= len(train_dataset)
        
        model.eval()
        dev_acc, dev_loss = 0, 0
        for batch_step, batch_data in enumerate(tqdm(eval_dataloader, desc="Valid")):
            with torch.no_grad():
                # input_ids, token_type_ids, attention_mask, labels = batch_data.values()
                input_ids, token_type_ids, attention_mask, start_pos, end_pos = [b_data.to(args.device) for b_data in batch_datas.values()]

                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
                loss = outputs.loss
            
                dev_loss += loss.detach().float()

                # predictions = outputs.logits.argmax(dim=-1)
                # print("pred", predictions)
                # print("labels", labels)
                # dev_acc += (predictions == labels).cpu().sum().item()

                # Choose the most probable start position / end position
                start_index = torch.argmax(outputs.start_logits, dim=1)
                end_index = torch.argmax(outputs.end_logits, dim=1)

                # Prediction is correct only if both start_index and end_index are correct
                dev_acc += ((start_index == start_pos) & (end_index == end_pos)).float().mean()

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
        default="./.ckpt/QA/models/",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        help="Path to save the tokenizer.",
        default="./.ckpt/QA/tokenizer/",
    )
    
    
    # data
    parser.add_argument("--max_length", type=int, default=384, help="The maximum length of a feature (question and context)")
    parser.add_argument("--doc_stride", type=int, default=128, help="The authorized overlap between two part of the context when splitting it is needed.")

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
    parser.add_argument("--batch_size", type=int, default=8)

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

    # accelerator = Accelerator()
    
    train()
    
    
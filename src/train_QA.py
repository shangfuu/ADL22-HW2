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

from datasets import DatasetDict, load_dataset, Dataset, load_metric
from accelerate import Accelerator
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, EvalPrediction,
                          default_data_collator, 
                          get_cosine_schedule_with_warmup,
                          SchedulerType, get_linear_schedule_with_warmup)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils_qa import postprocess_qa_predictions, create_and_fill_np_array
# from transformers.data.processors.squad import squad_convert_examples_to_features


def preprocess(train_dataset, val_dataset, tokenizer):
    
    def preprocess_train(examples):
        # Train preprocessing
        questions = [q.lstrip() for q in examples["question"]]
        contexts = examples["context"]
        
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
    
    def prepare_validation(examples):
        # Validation preprocessing
        questions = [q.lstrip() for q in examples["question"]]
        contexts = examples["context"]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            questions,
            contexts,
            # examples[question_column_name if pad_on_right else context_column_name],
            # examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second",
            max_length=args.max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        offset_mapping = tokenized_examples["offset_mapping"]
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
        
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    tokenized_train_dataset = train_dataset.map(preprocess_train, batched=True, remove_columns=train_dataset.column_names)
    tokenized_valid_dataset = val_dataset.map(prepare_validation, batched=True, remove_columns=val_dataset.column_names)

    return tokenized_train_dataset, tokenized_valid_dataset

def plot_curve(curve, title, legend, output_name):
    plt.plot(curve)
    plt.legend([legend])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.xticks([epoch for epoch in range(args.num_epoch) if epoch % 10 == 0])
    plt.savefig(args.plot_dir / output_name)
    plt.clf()


def plot_EM_loss_curve(em_curve, loss_curve):
    plot_curve(em_curve, "EM curve", "exact match", "EM_CURVE.png")
    plot_curve(loss_curve, "Loss curve", "Loss", "LOSS_CURVE.png")
    

def read_dataset():
    # Get Context
    # context_file = str((args.data_dir / "context.json").resolve())
    with open(args.context_file, encoding="utf-8") as f:
        context = json.load(f)
    
    # load train and valid json
    dataset_dict = dict()
    with open(args.train_file, encoding="utf-8") as f:
        tmp_json = json.load(f)
        pd_dict_train = pd.DataFrame.from_dict(tmp_json)
        for idx, data in enumerate(tmp_json):
            tmp_json[idx]["context"] = context[data["relevant"]]

        pd_dict_train = pd.DataFrame.from_dict(tmp_json)
                
    with open(args.valid_file, encoding="utf-8") as f:
        tmp_json = json.load(f)
        for idx, data in enumerate(tmp_json):
            tmp_json[idx]["context"] = context[data["relevant"]]
            
        pd_dict_val = pd.DataFrame.from_dict(tmp_json)

    pd_dataset_train = Dataset.from_pandas(pd_dict_train)
    pd_dataset_val = Dataset.from_pandas(pd_dict_val)
    
    dataset_dict["train"] = pd_dataset_train
    dataset_dict["validation"] = pd_dataset_val
    dataset = DatasetDict(dataset_dict)
    
    return dataset


def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=args.null_score_diff_threshold,
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": {"text": [ex["answer"]["text"]], "answer_start": [ex["answer"]["start"]]}} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def train():
    # swag = load_dataset('swag', 'regular')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    dataset = read_dataset()
    train_raw_dataset = dataset["train"]
    eval_raw_dataset = dataset["validation"]
    
    # preprocess return datasets
    train_dataset, eval_dataset = preprocess(train_raw_dataset, eval_raw_dataset, tokenizer)
    tokenizer.save_pretrained(args.tokenizer_path)

    # Dataloader
    data_collator = default_data_collator
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(eval_dataset_for_model, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)
    
    # Train
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path,)
    # model = AutoModelForQuestionAnswering.from_pretrained("./ckpt/QA/best")
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
    metric = load_metric("squad")
    
    print("Start Training")
    best_exact_match = 0.0
    best_dev_loss = float('inf')
    EM_curve = []
    loss_curve = []
    for epoch in range(args.num_epoch):
        model.train()
        print(f"\nEpoch: {epoch+1} / {args.num_epoch}")
        for batch_step, batch_datas in enumerate(tqdm(train_dataloader, desc="Train")):
            # input_ids, token_type_ids, attention_mask, labels = batch_datas.values()
            input_ids, token_type_ids, attention_mask, start_pos, end_pos = [b_data.to(args.device) for b_data in batch_datas.values()]

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
            loss = outputs.loss
            
            # train_loss += loss.detach().float()
            loss = loss / args.accum_steps
            loss.backward()

            # # Choose the most probable start position / end position
            # start_index = torch.argmax(outputs.start_logits, dim=1)
            # end_index = torch.argmax(outputs.end_logits, dim=1)

            if batch_step % args.accum_steps == 0 or batch_step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        model.eval()
        dev_loss = 0.0
        all_start_logits = []
        all_end_logits = []
        for batch_step, batch_datas in enumerate(tqdm(eval_dataloader, desc="Valid")):
            with torch.no_grad():
                # input_ids, token_type_ids, attention_mask, labels = batch_data.values()
                input_ids, token_type_ids, attention_mask, start_pos, end_pos = [b_data.to(args.device) for b_data in batch_datas.values()]

                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, start_positions= start_pos, end_positions=end_pos)
                loss = outputs.loss
                dev_loss += loss.detach().item()
                
                # outputs start end
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                all_start_logits.append(start_logits.cpu().numpy())
                all_end_logits.append(end_logits.cpu().numpy())


        max_len = max([x.shape[1] for x in all_start_logits])
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)
        
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits
        outputs_numpy = (start_logits_concat, end_logits_concat)
        
        prediction = post_processing_function(eval_raw_dataset, eval_dataset, outputs_numpy)
        predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        dev_loss /= (batch_step * args.accum_steps)

        loss_curve.append(dev_loss)
        EM_curve.append(predict_metric["exact_match"])
        print("eval matrix:", predict_metric)
        print("loss:", dev_loss)
        
        plot_EM_loss_curve(em_curve=EM_curve, loss_curve=loss_curve)

        if best_exact_match < predict_metric["exact_match"]:
            best_exact_match = predict_metric["exact_match"]
            if args.model_path is not None:
                save_name = str(args.model_path) + "_em"
                model.save_pretrained(save_name)
        
        if best_dev_loss < dev_loss:
            best_dev_loss = dev_loss
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
    parser.add_argument(
        "--context_file",
        type=Path,
        help="Context json file",
        default="./data/context.json",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        help="Context json file",
        default="./data/train.json",
    )
    parser.add_argument(
        "--valid_file",
        type=Path,
        help="Validation json file",
        default="./data/valid.json",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        help="Directory to store EM Loss Curve.",
        default="./plot/"
    )
        
    
    # data
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of a feature (question and context)")
    parser.add_argument("--doc_stride", type=int, default=192, help="The authorized overlap between two part of the context when splitting it is needed.")

    # model
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
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

    # accelerator = Accelerator()
    
    train()
    
    
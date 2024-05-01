""" Converts a DPO file to a format that can be ingested by the TRL RewardTraining class """

import argparse
from pathlib import Path
import json

from transformers import AutoTokenizer
from datasets import Dataset

def load_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_dataset(dpo_data):
    model_id = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    data = {
        "input_chosen_tokens": [],
        "input_rejected_tokens": [],
        "attention_mask_chosen": [],
        "attention_mask_rejected": []
    }
    for dpo_data_point in dpo_data:
        input_chosen = dpo_data_point['instruction'] + ' ' + dpo_data_point['output'][0]
        input_rejected = dpo_data_point['instruction'] + ' ' + dpo_data_point['output'][1]

        # tokenize the inputs
        input_chosen_tokens = tokenizer.encode(input_chosen, return_tensors='pt')
        input_rejected_tokens = tokenizer.encode(input_rejected, return_tensors='pt')

        # create attention masks
        attention_mask_chosen = [1] * len(input_chosen_tokens[0])
        attention_mask_rejected = [1] * len(input_rejected_tokens[0])

        data["input_chosen_tokens"].append(input_chosen_tokens)
        data["input_rejected_tokens"].append(input_rejected_tokens)
        data["attention_mask_chosen"].append(attention_mask_chosen)
        data["attention_mask_rejected"].append(attention_mask_rejected)

    return Dataset.from_dict(data)


def main(args: argparse.Namespace):

    dpo_data = load_jsonl(args.dpo_input_dataset)
    trl_data = create_dataset(dpo_data)
    trl_data.save_to_disk(args.trl_input_dataset)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a DPO file to a format that can be ingested by the TRL RewardTraining class')
    parser.add_argument('--dpo-input-dataset', type=str, help='Path to the input DPO jsonl file')
    parser.add_argument('--trl-input-dataset', type=str, help='Path to the output TRL reward modeling dataset')
    args = parser.parse_args()
    main(args)

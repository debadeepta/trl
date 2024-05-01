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
    data = {
        "chosen": [],
        "rejected": [],
    }
    for dpo_data_point in dpo_data:
        input_chosen = dpo_data_point['instruction'] + ' ' + dpo_data_point['output'][0]
        input_rejected = dpo_data_point['instruction'] + ' ' + dpo_data_point['output'][1]

        data["chosen"].append(input_chosen)
        data["rejected"].append(input_rejected)

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

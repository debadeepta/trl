""" Converts a DPO file to a format that can be ingested by the TRL RewardTraining class """

import argparse
from pathlib import Path
import json

def load_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main(args: argparse.Namespace):

    # load the DPO jsonl file
    dpo_data = load_jsonl(args.dpo_input_dataset)
    print('dummy')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a DPO file to a format that can be ingested by the TRL RewardTraining class')
    parser.add_argument('--dpo-input-dataset', type=str, help='Path to the input DPO jsonl file')
    parser.add_argument('--trl-input-dataset', type=str, help='Path to the output TRL file')
    args = parser.parse_args()
    main(args)

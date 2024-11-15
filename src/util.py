import json
import random
from collections import defaultdict

import yaml
import pandas as pd


def load_questions():
    section = [
        "intent_classifier",
        "store_inquiry_handler",
        "product_inquiry_handler",
        "unwanted_topic_blocker",
    ]
    return {
        name: pd.read_excel("data/banana_punch.xlsx", sheet_name=i)
        for i, name in enumerate(section)
    }


def load_kmle(train: bool = False):
    if train:
        target_year = "2023"
    else:
        target_year = "2024"
    data = []
    with open(f"data/kmle_{target_year}.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    data = [x for x in data if not x["has_picture"] or x["has_picture"] == "no" or "비공개" not in x["answer"]]

    category_dict = defaultdict(list)
    for item in data:
        category = item["problem_category"]
        category_dict[category].append(item)

    random.seed(42)

    sampled_data = []
    for category, items in category_dict.items():
        sampled_items = random.sample(items, min(6, len(items)))
        sampled_data.extend(sampled_items)

    return sampled_data


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

import json
import os
import pandas as pd
import pickle as pkl

from . import pers_labels
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

PWD = os.environ["WORKSPACE_PATH"]

def get_dataset(data_type):
    return pd.read_json(f"{PWD}/data/model_datasets/{data_type}.jsonl", lines=True)

def create_prompt(data_type, text, char):
    label_mode = pers_labels.get_label_mode(data_type)
    _, scene = text.split("\n", 1)

    ((label1, label1_def), (label2, label2_def)) = pers_labels.get_labels_and_defs(data_type)

    return \
f"""Read the scenes below and then categorize {char}'s personality as "{label1}" for {label1_def} or "{label2}" for {label2_def}, according to the {label_mode.lower()} personality typology. Response with only one word.

scenes:
{scene}

"""

def add_prompts(df, data_type):
    with open(f"{PWD}/data/cornell_movies/speakers.json", "r+") as fp:
        fp_parsed = json.load(fp)
        chars_meta = {}
        chars_meta_rows = []
        for char in fp_parsed:
            meta = fp_parsed[char]["meta"]
            meta["character_name"] = meta["character_name"].lower()
            meta["char_id"] = char
            chars_meta[char] = meta
            chars_meta_rows.append(meta)
    prompts = []
    for row in df.iloc:
        char_name = chars_meta[row.char_id]["character_name"]
        scene = row.text
        prompts.append(create_prompt(data_type, scene, char_name))
    df["text"] = prompts

def split_data(dataset, test_size, valid_size, random_state=12):
    splitter1 = GroupShuffleSplit(test_size=test_size, random_state=random_state)

    non_test_idx, test_idx = next(splitter1.split(X=dataset, y=dataset["label"], groups=dataset["movie_id"]))

    test_df = dataset.iloc[test_idx]
    non_test_df = dataset.iloc[non_test_idx]

    splitter2 = GroupShuffleSplit(test_size=valid_size, random_state=random_state)

    train_idx, valid_idx = next(splitter2.split(X=non_test_df, y=non_test_df["label"], groups=non_test_df["movie_id"]))

    to_split_df = non_test_df

    train_df = to_split_df.iloc[train_idx]
    valid_df = to_split_df.iloc[valid_idx]
    return train_df, valid_df, test_df

def get_data_splits(data_type, test_size, valid_size, random_state=12):
    dataset = get_dataset(data_type)
    add_prompts(dataset, data_type)
    return split_data(dataset, test_size=test_size, valid_size=valid_size, random_state=random_state)


def row_to_msg(row, include_assistant=False):
    output = [{
            "role": "user",
            "content": row["text"]
        }]
    if include_assistant:
        output.append({
            "role": "assistant",
            "content": row["label"]
        })
    return output

def convert_to_message_fmt(df, tkr, include_assistant=True):
    messages = []
    df.to_dict(orient="records")
    for row in df.iloc:
        messages.append(row_to_msg(row, include_assistant))
    messages_fmt = []
    for msg in messages:
        fmt_msg = tkr.apply_chat_template(msg, tokenize=False, add_generation_prompt=(not include_assistant))
        messages_fmt.append(fmt_msg)
    final_df = pd.DataFrame(messages_fmt, columns=["text"])
    return final_df

def encode_labels(dataset, data_type):
    file_path = f"{PWD}/data/label_encoders/{data_type}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb+") as fp:
            label_enc = pkl.load(fp)
            dataset["label"] = label_enc.transform(dataset["label"])
    else:
        label_enc = LabelEncoder()
        dataset["label"] = label_enc.fit_transform(dataset["label"])
        with open(file_path, "wb+") as fp:
            pkl.dump(label_enc, fp)

def get_model_path(model, data_type):
    dt = str(datetime.now()).replace(" ", "|").split(".")[0]
    return f"{PWD}/models/{model}/{data_type}/{dt}/"

def get_most_recent_model_path(model, data_type):
    model_dir = f"{PWD}/models/{model}/{data_type}"
    models = sorted(os.listdir(model_dir))
    return f"{model_dir}/{(models)[-1]}"